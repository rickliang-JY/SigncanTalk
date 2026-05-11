"""Drop a video into this directory, then run: python run.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from src.translator import SignTranslator


def find_video() -> Path | None:
    here = Path(__file__).parent
    for ext in ("*.mp4", "*.mov", "*.webm", "*.avi", "*.mkv"):
        for p in sorted(here.glob(ext)):
            return p
    return None


def main() -> None:
    video = find_video()
    if video is None:
        print("No video found. Drop a .mp4 / .mov / .webm into this directory and rerun.")
        sys.exit(1)

    print(f"Using video: {video.name}")
    print("Loading model...")
    t0 = time.time()
    t = SignTranslator()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    print("Translating...")
    t0 = time.time()
    result = t.translate_video(video)
    print(f"Done in {time.time() - t0:.1f}s ({result['frame_count']} frames)\n")

    print("=" * 60)
    print("RAW OUTPUT:")
    print("=" * 60)
    print(result["raw"])
    print()
    if result["parsed"]:
        print("=" * 60)
        print("PARSED:")
        print("=" * 60)
        print(json.dumps(result["parsed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
