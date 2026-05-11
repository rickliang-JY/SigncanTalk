"""Test Gemma 4's pure-text cross-sign-language translation capability.

Runs 10 hand-crafted CSL gloss inputs through:
  CSL gloss -> Chinese -> English -> ASL gloss

Output goes to stdout for human evaluation. No SLR, no video, no fine-tune.
This validates whether Gemma 4 E4B has the language capability we're betting on.

Usage:
    python scripts/test_text_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_translator import TextCrossSignTranslator  # noqa: E402


TEST_CASES: list[tuple[str, list[str], str]] = [
    (
        "greeting",
        ["你", "好"],
        "Hello",
    ),
    (
        "thanks_for_help",
        ["谢谢", "你", "帮助", "我"],
        "Thank you for helping me",
    ),
    (
        "going_to_beijing",
        ["明天", "北京", "我", "去"],
        "I'm going to Beijing tomorrow",
    ),
    (
        "ate_at_mom",
        ["昨天", "妈妈", "家", "我", "吃饭"],
        "Yesterday I ate at my mother's house",
    ),
    (
        "what_time_now",
        ["现在", "几点"],
        "What time is it now?",
    ),
    (
        "weather_today",
        ["今天", "天气", "怎么样"],
        "How's the weather today?",
    ),
    (
        "negation_dont_want",
        ["我", "不", "想", "去"],
        "I don't want to go",
    ),
    (
        "school_tomorrow",
        ["明天", "学校", "我", "去"],
        "I'm going to school tomorrow",
    ),
    (
        "name_introduction",
        ["我", "名字", "李明"],
        "My name is Li Ming",
    ),
    (
        "where_bathroom",
        ["厕所", "在", "哪里"],
        "Where is the bathroom?",
    ),
]


def main() -> None:
    print("Loading Gemma 4...")
    t0 = time.time()
    t = TextCrossSignTranslator()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    print("=" * 80)
    print("CROSS-SIGN TEXT TRANSLATION TEST")
    print("=" * 80)

    summary = []
    for i, (name, csl_input, expected_meaning) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {name}")
        print(f"  Input CSL gloss : {csl_input}")
        print(f"  Expected meaning: {expected_meaning}")

        t0 = time.time()
        result = t.translate(csl_input)
        elapsed = time.time() - t0

        parsed = result["parsed"]
        if parsed:
            print(f"  Chinese  : {parsed.get('chinese', '???')}")
            print(f"  English  : {parsed.get('english', '???')}")
            print(f"  ASL gloss: {parsed.get('asl_gloss', '???')}")
            notes = parsed.get("asl_grammar_notes", "")
            if notes:
                print(f"  Notes    : {notes}")
            summary.append((name, "PARSED", elapsed))
        else:
            print(f"  [JSON parse failed]")
            print(f"  Raw: {result['raw'][:200]}")
            summary.append((name, "FAILED", elapsed))

        print(f"  ({elapsed:.1f}s)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, status, elapsed in summary:
        print(f"  [{status:6s}] {elapsed:5.1f}s  {name}")

    avg = sum(e for _, _, e in summary) / len(summary)
    parsed_rate = sum(1 for _, s, _ in summary if s == "PARSED") / len(summary)
    print(f"\n  Avg latency: {avg:.1f}s | JSON parse rate: {parsed_rate:.0%}")
    print("\nEvaluate manually: are the Chinese/English/ASL outputs reasonable?")


if __name__ == "__main__":
    main()
