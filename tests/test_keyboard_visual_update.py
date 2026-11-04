"""Synthetic JPEG evidence for the visual wakeup heuristic; no real services."""
from contextlib import redirect_stdout
from io import BytesIO
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from astra_robotics_agent.physical_keyboard.visual_update import (
    MAX_IMAGE_BYTES, compare_preview_images, validate_roi,
)


def jpeg(image, quality=90):
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def background(size=(1600, 1200), value=224):
    return Image.new("RGB", size, (value, value, value))


def declared_jpeg_size(raw, width, height):
    """Change only the SOF header to exercise bounds before pixel decoding."""
    index = raw.index(b"\xff\xc0")
    return raw[:index + 5] + struct.pack(">HH", height, width) + raw[index + 9:]


class VisualUpdateTests(unittest.TestCase):
    def test_identical_preview_stays_idle_and_metrics_are_json_safe(self):
        raw = jpeg(background())
        result = compare_preview_images(raw, raw)
        self.assertFalse(result["changed"])
        self.assertEqual(result["mean_abs_difference"], 0)
        self.assertEqual(result["changed_fraction"], 0)
        self.assertEqual(result["image_shape"], [1200, 1600])
        self.assertEqual(result["comparison_shape"], [240, 320])
        json.dumps(result, allow_nan=False)

    def test_small_new_message_wakes_inside_selected_chat_roi(self):
        before = background()
        after = before.copy()
        draw = ImageDraw.Draw(after)
        draw.rectangle((400, 350, 580, 420), fill=(110, 210, 90))
        draw.text((425, 370), "hi", fill=(0, 0, 0))
        result = compare_preview_images(jpeg(before), jpeg(after), [.15, .05, .85, .65])
        self.assertTrue(result["changed"])
        self.assertGreater(result["changed_fraction"], .002)
        self.assertGreater(result["mean_abs_difference"], .15)

    def test_cursor_outside_chat_roi_does_not_wake(self):
        before = background()
        after = before.copy()
        ImageDraw.Draw(after).rectangle((400, 1020, 408, 1090), fill=(0, 0, 0))
        result = compare_preview_images(jpeg(before), jpeg(after), [.15, .05, .85, .65])
        self.assertFalse(result["changed"])
        self.assertEqual(result["mean_abs_difference"], 0)

    def test_low_sensor_jpeg_noise_stays_idle(self):
        data = np.full((1200, 1600, 3), 224, dtype=np.int16)
        noise = np.random.default_rng(28).integers(-2, 3, size=data.shape)
        result = compare_preview_images(jpeg(Image.fromarray(data.astype(np.uint8))),
            jpeg(Image.fromarray((data + noise).astype(np.uint8))))
        self.assertFalse(result["changed"])
        self.assertLess(result["changed_fraction"], .002)

    def test_small_global_exposure_jitter_requires_pixel_fraction_too(self):
        result = compare_preview_images(jpeg(background(value=224)), jpeg(background(value=229)))
        self.assertGreater(result["mean_abs_difference"], .15)
        self.assertEqual(result["changed_fraction"], 0)
        self.assertFalse(result["changed"])

    def test_strong_uniform_change_wakes_but_is_not_semantic_classification(self):
        result = compare_preview_images(jpeg(background(value=224)), jpeg(background(value=180)))
        self.assertTrue(result["changed"])
        self.assertIn("not text", result["meaning"])

    def test_thresholds_are_explicit_and_can_suppress_a_detected_change(self):
        before = background((320, 240))
        after = before.copy()
        ImageDraw.Draw(after).rectangle((100, 100, 150, 130), fill=(0, 0, 0))
        first = compare_preview_images(jpeg(before), jpeg(after))
        second = compare_preview_images(jpeg(before), jpeg(after), minimum_changed_fraction=.8)
        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(first["changed_fraction"], second["changed_fraction"])

    def test_bad_roi_shapes_numbers_and_bounds_reject(self):
        for value in ([], [0, 0, 1], [0, 0, 1, 1, 1], "0,0,1,1", {},
                      [True, 0, 1, 1], [0, 0, float("nan"), 1],
                      [0, 0, float("inf"), 1], [-.1, 0, 1, 1],
                      [0, 0, 1.1, 1], [1, 0, 0, 1], [0, .5, 1, .5],
                      [0, 0, 10**1000, 1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_roi(value)

    def test_roi_is_copied_and_tiny_region_is_nonempty(self):
        roi = [0, 0, 1, 1]
        copied = validate_roi(roi)
        copied[0] = .2
        self.assertEqual(roi, [0, 0, 1, 1])
        self.assertEqual(validate_roi(), [0., 0., 1., 1.])
        raw = jpeg(background((10, 10)))
        result = compare_preview_images(raw, raw, [0., 0., .00001, .00001])
        self.assertEqual(result["comparison_shape"], [1, 1])

    def test_nonfinite_boolean_or_inactive_thresholds_reject(self):
        raw = jpeg(background((32, 32)))
        invalid = {
            "pixel_threshold": [True, None, float("nan"), float("inf"), -1, 255],
            "minimum_changed_fraction": [False, "0.1", 0, 1.1, float("nan")],
            "minimum_mean_difference": [True, 0, -1, 256, float("inf")],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    compare_preview_images(raw, raw, **{field: value})

    def test_invalid_nonjpeg_oversize_or_nonbytes_input_rejects(self):
        valid = jpeg(background((32, 32)))
        png = BytesIO(); background((32, 32)).save(png, format="PNG")
        for raw in (b"", b"not an image", bytearray(valid), "jpeg", None,
                    valid + b"a" * MAX_IMAGE_BYTES, png.getvalue(), valid[:20]):
            for first, second in ((raw, valid), (valid, raw)):
                with self.subTest(type=type(raw).__name__), self.assertRaises(ValueError):
                    compare_preview_images(first, second)

    def test_mismatched_dimensions_reject(self):
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            compare_preview_images(jpeg(background((32, 32))), jpeg(background((33, 32))))

    def test_decoded_dimension_and_pixel_limits_before_decode(self):
        raw = jpeg(background((32, 32)))
        for width, height in ((4097, 32), (32, 4097), (4096, 4096), (65535, 65535)):
            bad = declared_jpeg_size(raw, width, height)
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                compare_preview_images(bad, bad)

    def test_import_does_not_load_optional_decoders_or_run_io(self):
        path = Path(__file__).resolve().parents[1] / "src/astra_robotics_agent/physical_keyboard/visual_update.py"
        spec = importlib.util.spec_from_file_location("visual_update_import_probe", path)
        module = importlib.util.module_from_spec(spec)
        # Optional image libraries may already be loaded by this test module;
        # assert the freshly imported utility itself exposes no eager bindings.
        with patch("subprocess.Popen", side_effect=AssertionError("No process")), \
                patch("socket.socket", side_effect=AssertionError("No network")), \
                redirect_stdout(io.StringIO()) as stdout:
            spec.loader.exec_module(module)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("np", module.__dict__)
        self.assertNotIn("Image", module.__dict__)
        self.assertEqual(module.validate_roi(), [0., 0., 1., 1.])


if __name__ == "__main__":
    unittest.main()
