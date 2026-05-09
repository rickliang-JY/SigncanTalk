"""Quick offline smoke test: load model, run inference on a single video, print result.

Usage:
    python scripts/smoke_test.py path/to/video.mp4
    python scripts/smoke_test.py path/to/video.mp4 --hint "ASL fingerspelling"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.translator import SignTranslator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="Path to a sign language video")
    parser.add_argument("--hint", type=str, default=None)
    parser.add_argument(
        "--langs",
        type=str,
        default="English,Chinese",
        help="Comma-separated list of target languages",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    target_langs = [s.strip() for s in args.langs.split(",") if s.strip()]

    print(f"Loading model... (this may take a minute)")
    t0 = time.time()
    translator = SignTranslator()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    print(f"Running inference on {video_path} ...")
    t0 = time.time()
    result = translator.translate_video(
        video_path, target_languages=target_langs, hint=args.hint
    )
    elapsed = time.time() - t0

    print(f"\n=== RESULT (frames={result['frame_count']}, latency={elapsed:.1f}s) ===")
    print("\n--- raw ---")
    print(result["raw"])
    print("\n--- parsed ---")
    print(json.dumps(result["parsed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
