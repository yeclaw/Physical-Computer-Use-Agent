"""Derived wrist-camera playback from closed, timestamped keyboard demos.

No robot imports or raw-data edits. Playback is 30 FPS resampled from actual
sampler timestamps; cached camera frames are not acquisition-time guarantees.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

PROTOCOL = "keyboard-right-wrist-review-v1"
FPS = 30
CONFIG = {"camera": "right_wrist_camera", "fps": FPS, "codec": "libvpx",
          "pixel_format": "yuv420p", "crf": 10, "bitrate": "2M",
          "deadline": "good", "cpu_used": 4, "encoding_threads": 2,
          "scope": "full_episode_no_crop", "time_source": "sample.loop_anchor.monotonic_ns",
          "resampling": "causal hold at n/30 seconds; frame count=max(1, round(measured duration*30))",
          "last_frame_duration": "median positive sample interval; singleton uses recorded nominal fps"}


def camera_settings(camera):
    """Keep default right-view manifests byte-compatible in schema/content."""
    if camera not in ("right_wrist_camera", "left_wrist_camera"):
        raise ValueError("unsupported review camera: " + str(camera))
    config = dict(CONFIG, camera=camera)
    protocol = PROTOCOL if camera == "right_wrist_camera" else "keyboard-left-wrist-review-v1"
    return protocol, config, camera.removesuffix("_camera") + ".webm"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(root, relative):
    if not isinstance(relative, str) or "\n" in relative or "\r" in relative:
        raise ValueError("invalid relative source path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("source path escapes episode")
    current = root
    for component in part.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("source symlinks are not allowed")
    if not current.is_file():
        raise ValueError("missing source file: " + str(current))
    return current


def inspect_episode(episode, *, camera="right_wrist_camera"):
    """Validate a finalized episode and return immutable source/timing metadata."""
    camera_settings(camera)
    episode = Path(episode).expanduser()
    if episode.is_symlink():
        raise ValueError("episode symlink is not allowed")
    episode = episode.resolve(strict=True)
    summary_path = _file(episode, "keyboard-demo.json")
    summary = json.loads(summary_path.read_text())
    if summary.get("operator_invalid") is True or (episode / "false").exists():
        return {"episode": str(episode), "invalid": True}
    if summary.get("operator_invalid") is not False:
        raise ValueError("episode has no explicit normal completion marker")
    events_path = _file(episode, "keyboard-events.jsonl")
    events_sha = _sha(events_path)
    if events_sha != summary.get("events_sha256"):
        raise ValueError("events SHA mismatch")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if not events or events[0].get("type") != "begin" or events[-1].get("type") != "end":
        raise ValueError("episode is not closed")
    if sum(e.get("type") == "end" for e in events) != 1 or events[-1].get("operator_invalid") is not False:
        raise ValueError("invalid episode end marker")
    samples = [e for e in events if e.get("type") == "sample"]
    count = len(samples)
    if count == 0 or summary.get("sample_count") != count or events[-1].get("sample_count") != count:
        raise ValueError("sample count mismatch or empty recording")
    data_path = _file(episode, "data.json")
    data = json.loads(data_path.read_text())
    items = data.get("data", [])
    if not isinstance(data.get("summary"), dict) or len(items) != count:
        raise ValueError("writer not finalized or data count mismatch")
    frames = []
    previous = None
    for index, (sample, item) in enumerate(zip(samples, items)):
        if type(sample.get("item_index")) is not int or sample["item_index"] != index or type(item.get("idx")) is not int or item["idx"] != index:
            raise ValueError("non-contiguous sample/data index")
        stamp = sample.get("loop_anchor", {}).get("monotonic_ns")
        if type(stamp) is not int or stamp < 0 or (previous is not None and stamp <= previous):
            raise ValueError("sample timestamps must be strictly increasing integers")
        previous = stamp
        path = _file(episode, item.get("colors", {}).get(camera))
        frames.append({"index": index, "path": str(path.relative_to(episode)),
                       "sha256": _sha(path), "monotonic_ns": stamp,
                       "utc_ns": sample.get("loop_anchor", {}).get("utc_ns")})
    intervals = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e9 for a, b in zip(frames, frames[1:])]
    if intervals:
        tail = statistics.median(intervals)
    else:
        nominal = data.get("info", {}).get("image", {}).get("fps", FPS)
        if not isinstance(nominal, (int, float)) or not math.isfinite(nominal) or nominal <= 0:
            raise ValueError("invalid singleton nominal fps")
        tail = 1 / nominal
    result = {"episode": str(episode), "invalid": False,
              "inputs": {"keyboard-demo.json": _sha(summary_path), "keyboard-events.jsonl": events_sha,
                         "data.json": _sha(data_path)}, "frames": frames,
              "source_frame_count": count, "sample_elapsed_s": (frames[-1]["monotonic_ns"] - frames[0]["monotonic_ns"]) / 1e9,
              "last_frame_duration_s": tail, "expected_duration_s": sum(intervals) + tail}
    result["fingerprint"] = hashlib.sha256(_json(result).encode()).hexdigest()
    return result



def playback_frame_count(source):
    return max(1, round(source["expected_duration_s"] * FPS))


def playback_frames(source):
    """Yield causal source samples at n/FPS, with constant auxiliary memory."""
    frames = source["frames"]
    first = frames[0]["monotonic_ns"]
    cursor = 0
    for index in range(playback_frame_count(source)):
        target_ns = first + index * 1_000_000_000 // FPS
        while cursor + 1 < len(frames) and frames[cursor + 1]["monotonic_ns"] <= target_ns:
            cursor += 1
        yield frames[cursor]


def _run(command, timeout):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError("media command failed: " + completed.stderr[-4000:])
    return completed.stdout


def _probe(path, ffprobe, ffmpeg, timeout):
    data = json.loads(_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                            "-show_streams", "-show_format", "-of", "json", str(path)], timeout))
    if len(data.get("streams", [])) != 1:
        raise ValueError("review must contain exactly one video stream")
    stream = data["streams"][0]
    if stream.get("codec_name") != "vp8" or stream.get("pix_fmt") != "yuv420p" or abs(float(stream.get("start_time", "nan"))) > .001:
        raise ValueError("review codec/pixel-format/start-time mismatch")
    _run([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], timeout)
    return {"codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
            "width": stream["width"], "height": stream["height"],
            "encoded_frame_count": int(stream["nb_read_frames"]),
            "duration_s": float(data["format"]["duration"]), "start_time_s": float(stream["start_time"]),
            "full_decode": "passed"}


def export_episode(episode: Path, output_root: Path, *, camera="right_wrist_camera") -> dict:
    """Publish one derived WebM+manifest, or validate and return an existing one.

    Invalid segments return skipped_invalid and are never deleted here. The
    caller owns writer completion and X-only deletion. Errors leave raw intact.
    """
    protocol, config, filename = camera_settings(camera)
    source = inspect_episode(episode, camera=camera)
    result = {"episode": source["episode"], "video": None, "manifest": None}
    if source["invalid"]:
        return dict(result, status="skipped_invalid")
    episode = Path(source["episode"])
    output_root = Path(output_root).expanduser()
    if output_root.is_symlink():
        raise ValueError("output root symlink is not allowed")
    output_root = output_root.resolve()
    if output_root == episode or episode in output_root.parents:
        raise ValueError("derived output must be outside the raw episode")
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / episode.name
    if final.is_symlink():
        raise ValueError("review directory symlink is not allowed")
    lock = output_root / ("." + episode.name + ".export-lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("episode export already in progress (or stale lock): " + str(lock)) from exc
    staging = None
    try:
        video, manifest_path = final / filename, final / "manifest.json"
        result.update(video=str(video), manifest=str(manifest_path))
        if final.exists():
            manifest = json.loads(_file(final, "manifest.json").read_text())
            if manifest.get("protocol") != protocol or manifest.get("source") != source or manifest.get("config") != config or _sha(_file(final, filename)) != manifest.get("video_sha256"):
                raise ValueError("existing review differs; refusing overwrite")
            return dict(result, status="existing")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError("ffmpeg and ffprobe are required")
        staging = Path(tempfile.mkdtemp(prefix="." + episode.name + ".", dir=output_root))
        concat = staging / "frames.ffconcat"
        # Explicitly resample in Python: FFmpeg 4.2 does not understand the
        # ffconcat per-file `option framerate` syntax. A file-only list and an
        # input -r produce one evenly spaced frame per selected source image.
        with concat.open("w") as stream:
            stream.write("ffconcat version 1.0\n")
            for frame in playback_frames(source):
                path = str(episode / frame["path"]).replace("'", "'\\''")
                stream.write("file '" + path + "'\n")
        temp_video = staging / video.name
        command = [ffmpeg, "-nostdin", "-v", "error", "-r", str(FPS),
                   "-f", "concat", "-safe", "0", "-i", str(concat),
                   "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=tv,format=yuv420p",
                   "-vsync", "0", "-c:v", config["codec"], "-threads", str(config["encoding_threads"]),
                   "-deadline", config["deadline"], "-cpu-used", str(config["cpu_used"]),
                   "-crf", str(config["crf"]), "-b:v", config["bitrate"],
                   "-pix_fmt", config["pixel_format"], str(temp_video)]
        timeout = max(60, source["expected_duration_s"] * 5 + 30)
        _run(command, timeout)
        media = _probe(temp_video, ffprobe, ffmpeg, timeout)
        if media["encoded_frame_count"] != playback_frame_count(source):
            raise ValueError("review encoded frame count does not match resampling plan")
        if abs(media["duration_s"] - source["expected_duration_s"]) > 1 / FPS + .002:
            raise ValueError("review duration does not match sampler time")
        if inspect_episode(episode, camera=camera) != source:
            raise ValueError("raw episode changed during export")
        manifest = {"protocol": protocol, "generated_utc_ns": time.time_ns(), "source": source,
                    "config": config, "media": media, "video_sha256": _sha(temp_video),
                    "ffmpeg_version": _run([ffmpeg, "-version"], 10).splitlines()[0],
                    "command": command, "raw_modified": False, "skill_promoted": False,
                    "timing_meaning": "host sampler timeline; cached image timestamps not acquisition-calibrated"}
        (staging / manifest_path.name).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        concat.unlink()
        if final.exists():
            raise ValueError("review appeared during export; refusing overwrite")
        staging.rename(final)
        staging = None
        return dict(result, status="exported")
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.rmdir()
