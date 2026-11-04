"""Pure image-change metrics for a cheap visual wakeup gate.

This module does not read files, access cameras, call models or infer messages.
It accepts already trusted JPEG bytes. The caller owns capture freshness,
temporal persistence and the decision to supply a new observation to its agent.
Pillow and NumPy are imported lazily, after byte/argument validation.
"""
from __future__ import annotations

import math


MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 12_000_000
MAX_COMPARISON_WIDTH = 320
MAX_COMPARISON_HEIGHT = 240


def _number(value, label):
    if type(value) not in (int, float):
        raise ValueError(label + " must be a finite number, not a boolean")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(label + " must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(label + " must be finite")
    return value


def validate_roi(roi=None):
    """Return a copied normalized [left, top, right, bottom] rectangle."""
    if roi is None:
        return [0.0, 0.0, 1.0, 1.0]
    if type(roi) not in (list, tuple) or len(roi) != 4:
        raise ValueError("roi must be four normalized coordinates")
    result = [_number(item, "roi coordinate") for item in roi]
    if (any(not 0.0 <= item <= 1.0 for item in result)
            or result[0] >= result[2] or result[1] >= result[3]):
        raise ValueError("roi must be nonempty and lie within [0, 1]")
    return result


def _validate_bytes(raw, label):
    if type(raw) is not bytes or not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(label + " must be nonempty JPEG bytes of at most 4 MiB")


def compare_preview_images(reference: bytes, current: bytes, roi=None, *,
                           pixel_threshold=10.0,
                           minimum_changed_fraction=0.002,
                           minimum_mean_difference=0.15,
                           local_change=False, full_frame_guard=False):
    """Compare bounded grayscale thumbnails; return JSON-safe wakeup metrics.

    The legacy global metric requires BOTH enough pixels above the threshold
    and enough mean difference after a small blur. ``local_change`` additionally
    checks overlapping thumbnail tiles; ``full_frame_guard`` compares outside
    the selected ROI too. Both options default to False. These are image
    heuristics, not message recognition or a guarantee against false wakes or
    misses. The caller should require change across successive captured frames.
    """
    region = validate_roi(roi)
    if type(local_change) is not bool or type(full_frame_guard) is not bool:
        raise ValueError("local_change and full_frame_guard must be explicit booleans")
    pixel_threshold = _number(pixel_threshold, "pixel_threshold")
    minimum_changed_fraction = _number(minimum_changed_fraction, "minimum_changed_fraction")
    minimum_mean_difference = _number(minimum_mean_difference, "minimum_mean_difference")
    if not 0.0 <= pixel_threshold < 255.0:
        raise ValueError("pixel_threshold must lie in [0, 255)")
    if not 0.0 < minimum_changed_fraction <= 1.0:
        raise ValueError("minimum_changed_fraction must lie in (0, 1]")
    if not 0.0 < minimum_mean_difference <= 255.0:
        raise ValueError("minimum_mean_difference must lie in (0, 255]")
    _validate_bytes(reference, "reference")
    _validate_bytes(current, "current")

    from contextlib import ExitStack
    from io import BytesIO
    import warnings

    import numpy as np
    from PIL import Image, ImageFilter

    def thumbnail(image, box, size):
        # The header limits are checked before conversion/crop triggers decoding.
        return np.asarray(
            image.convert("L").crop(box).resize(size, Image.Resampling.BILINEAR)
            .filter(ImageFilter.GaussianBlur(radius=0.6)), dtype=np.int16)

    def local_metric(difference):
        # Fixed overlapping thumbnail tiles prevent a small, strong change from
        # being diluted by the whole display. This is deliberately geometric:
        # a caret, exposure shift or hand can still cause a false wake.
        height, width = difference.shape
        tile_width, tile_height = min(32, width), min(24, height)
        def positions(total, tile):
            return sorted(set(range(0, max(1, total - tile + 1), max(1, tile // 2)))
                          | {total - tile})
        strongest = None
        changed = False
        for top in positions(height, tile_height):
            for left in positions(width, tile_width):
                tile = difference[top:top + tile_height, left:left + tile_width]
                count = int(np.count_nonzero(tile > pixel_threshold))
                fraction = count / tile.size
                mean = float(np.mean(tile))
                hit = bool(count >= 4 and fraction >= 0.04 and mean >= 1.0)
                changed = changed or hit
                score = fraction * mean
                if strongest is None or score > strongest[0]:
                    strongest = (score, dict(box=[left, top, left + tile_width, top + tile_height],
                        changed_fraction=fraction, mean_abs_difference=mean,
                        changed_pixels=count, changed=hit))
        return dict(changed=changed, strongest_tile=strongest[1],
                    tile_shape=[tile_height, tile_width], minimum_changed_fraction=0.04,
                    minimum_mean_difference=1.0, minimum_changed_pixels=4)

    def compare_region(images, width, height, selected):
        box = (math.floor(selected[0] * width), math.floor(selected[1] * height),
               math.ceil(selected[2] * width), math.ceil(selected[3] * height))
        crop_width, crop_height = box[2] - box[0], box[3] - box[1]
        ratio = min(1.0, MAX_COMPARISON_WIDTH / crop_width,
                    MAX_COMPARISON_HEIGHT / crop_height)
        size = (max(1, round(crop_width * ratio)), max(1, round(crop_height * ratio)))
        difference = np.abs(thumbnail(images[0], box, size) - thumbnail(images[1], box, size))
        mean = float(np.mean(difference))
        fraction = float(np.count_nonzero(difference > pixel_threshold) / difference.size)
        return dict(changed=bool(fraction >= minimum_changed_fraction and mean >= minimum_mean_difference),
                    mean_abs_difference=mean, changed_fraction=fraction,
                    comparison_shape=[size[1], size[0]],
                    local=local_metric(difference) if local_change else None)

    try:
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            images = []
            for raw in (reference, current):
                image = stack.enter_context(Image.open(BytesIO(raw)))
                width, height = image.size
                if (image.format != "JPEG" or width < 1 or height < 1
                        or max(width, height) > MAX_IMAGE_DIMENSION
                        or width * height > MAX_IMAGE_PIXELS):
                    raise ValueError("preview must be a JPEG within decoded dimension/pixel limits")
                images.append(image)
            if images[0].size != images[1].size:
                raise ValueError("preview image dimensions differ; re-establish the baseline")
            width, height = images[0].size
            selected = compare_region(images, width, height, region)
            full = (selected if region == [0.0, 0.0, 1.0, 1.0]
                    else compare_region(images, width, height, [0., 0., 1., 1.])) if full_frame_guard else None
    except (OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid or unsafe preview JPEG") from exc

    result = {
        "protocol": "physical-keyboard-visual-update-metrics-v1",
        "changed": selected['changed'],
        "mean_abs_difference": selected['mean_abs_difference'],
        "changed_fraction": selected['changed_fraction'],
        "roi": region,
        "image_shape": [height, width],
        "comparison_shape": selected['comparison_shape'],
        "pixel_threshold": pixel_threshold,
        "minimum_changed_fraction": minimum_changed_fraction,
        "minimum_mean_difference": minimum_mean_difference,
        "meaning": "visual wakeup heuristic; not text, recipient or message recognition",
    }
    if local_change or full_frame_guard:
        reasons = ['roi_global'] if selected['changed'] else []
        if local_change and selected['local']['changed']:
            reasons.append('roi_local')
        if full is not None:
            if full['changed']:
                reasons.append('full_frame_global')
            if local_change and full['local']['changed']:
                reasons.append('full_frame_local')
        result.update(changed=bool(reasons), wake_sources=reasons,
                      local_change=local_change, full_frame_guard=full_frame_guard)
        if local_change:
            result['local_metric'] = selected['local']
        if full is not None:
            result['full_frame_metric'] = full
    return result
