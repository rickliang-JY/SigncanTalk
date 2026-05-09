from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import FRAME_RESIZE_LONG_EDGE, MAX_FRAMES, VIDEO_FPS


def extract_frames(
    video_path: str | Path,
    fps: int = VIDEO_FPS,
    max_frames: int = MAX_FRAMES,
    resize_long_edge: int | None = FRAME_RESIZE_LONG_EDGE,
) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / src_fps if src_fps > 0 else 0.0

    target_count = min(max_frames, max(1, int(round(duration * fps))))
    if target_count <= 0:
        cap.release()
        raise ValueError(f"Video appears empty: {video_path}")

    indices = np.linspace(0, max(total_frames - 1, 0), num=target_count, dtype=int)
    indices_set = set(int(i) for i in indices)

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            if resize_long_edge:
                img = _resize_long_edge(img, resize_long_edge)
            frames.append(img)
            if len(frames) >= target_count:
                break
        idx += 1

    cap.release()
    return frames


def _resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= long_edge:
        return img
    scale = long_edge / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
