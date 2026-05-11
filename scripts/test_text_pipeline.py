"""Validate the foundational primitive: sign language <-> natural language.

Tests 4 directions independently:
  1. CSL gloss  -> Chinese        (sign-to-natural)
  2. Chinese    -> CSL gloss      (natural-to-sign)
  3. ASL gloss  -> English        (sign-to-natural)
  4. English    -> ASL gloss      (natural-to-sign)

If all 4 work reliably, sign<->natural is "互通" and we can chain:
  CSL gloss -> Chinese -> English -> ASL gloss

Run: python scripts/test_text_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_translator import SignNaturalTranslator  # noqa: E402


CSL_CHINESE_PAIRS: list[tuple[str, list[str], str]] = [
    ("greeting",        ["你", "好"],                              "你好。"),
    ("thanks_help",     ["谢谢", "你", "帮助", "我"],              "谢谢你帮助我。"),
    ("to_beijing",      ["明天", "北京", "我", "去"],              "我明天去北京。"),
    ("ate_at_mom",      ["昨天", "妈妈", "家", "我", "吃饭"],      "昨天我在妈妈家吃饭。"),
    ("what_time",       ["现在", "几点"],                          "现在几点?"),
    ("weather",         ["今天", "天气", "怎么样"],                "今天天气怎么样?"),
    ("dont_want_go",    ["我", "不", "想", "去"],                  "我不想去。"),
    ("school_tomorrow", ["明天", "学校", "我", "去"],              "我明天去学校。"),
    ("my_name",         ["我", "名字", "李明"],                    "我叫李明。"),
    ("where_bathroom",  ["厕所", "在", "哪里"],                    "厕所在哪里?"),
]


ASL_ENGLISH_PAIRS: list[tuple[str, list[str], str]] = [
    ("greeting",        ["HELLO"],                                                 "Hello."),
    ("thanks_help",     ["THANK-YOU", "IX-2", "HELP", "IX-1"],                     "Thank you for helping me."),
    ("to_beijing",      ["TOMORROW", "FS:BEIJING", "IX-1", "GO"],                  "I'm going to Beijing tomorrow."),
    ("ate_at_mom",      ["YESTERDAY", "MOTHER", "POSS-3", "HOUSE", "IX-1", "EAT"], "Yesterday I ate at my mother's house."),
    ("what_time",       ["TIME", "NOW", "(Q)"],                                    "What time is it now?"),
    ("weather",         ["TODAY", "WEATHER", "HOW", "(Q)"],                        "How is the weather today?"),
    ("dont_want_go",    ["IX-1", "WANT", "GO", "NOT"],                             "I don't want to go."),
    ("school_tomorrow", ["TOMORROW", "SCHOOL", "IX-1", "GO"],                      "I'm going to school tomorrow."),
    ("my_name",         ["IX-1", "NAME", "FS:LI-MING"],                            "My name is Li Ming."),
    ("where_bathroom",  ["BATHROOM", "WHERE", "(Q)"],                              "Where is the bathroom?"),
]


def run_direction(t: SignNaturalTranslator, label: str, cases: list, direction: str,
                  sign_system, natural_lang) -> dict:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")
    parsed_ok = 0
    total_time = 0.0
    rows = []

    for i, case in enumerate(cases, 1):
        name = case[0]
        if direction == "sign_to_natural":
            gloss, gold = case[1], case[2]
            print(f"\n[{i:2}] {name}")
            print(f"     Input  : {gloss}")
            print(f"     Gold   : {gold}")
            t0 = time.time()
            r = t.sign_to_natural(gloss, sign_system, natural_lang)
            dt = time.time() - t0
        else:
            gold_gloss, text = case[1], case[2]
            print(f"\n[{i:2}] {name}")
            print(f"     Input  : {text}")
            print(f"     Gold   : {gold_gloss}")
            t0 = time.time()
            r = t.natural_to_sign(text, sign_system, natural_lang)
            dt = time.time() - t0

        total_time += dt
        parsed = r["parsed"]
        if parsed:
            parsed_ok += 1
            if direction == "sign_to_natural":
                print(f"     Output : {parsed.get('sentence', '???')}")
            else:
                print(f"     Output : {parsed.get('gloss', '???')}")
            notes = parsed.get("notes", "")
            if notes:
                print(f"     Notes  : {notes}")
        else:
            print(f"     [JSON parse FAILED] raw: {r['raw'][:160]}")
        print(f"     ({dt:.1f}s)")
        rows.append((name, "PARSED" if parsed else "FAILED", dt))

    return {
        "label": label,
        "rows": rows,
        "parsed_rate": parsed_ok / len(cases),
        "avg_latency": total_time / len(cases),
    }


def main() -> None:
    print("Loading Gemma 4...")
    t0 = time.time()
    t = SignNaturalTranslator()
    print(f"Loaded in {time.time() - t0:.1f}s")

    reports = [
        run_direction(t, "1. CSL gloss -> Chinese",
                      CSL_CHINESE_PAIRS, "sign_to_natural", "CSL", "Chinese"),
        run_direction(t, "2. Chinese -> CSL gloss",
                      CSL_CHINESE_PAIRS, "natural_to_sign", "CSL", "Chinese"),
        run_direction(t, "3. ASL gloss -> English",
                      ASL_ENGLISH_PAIRS, "sign_to_natural", "ASL", "English"),
        run_direction(t, "4. English -> ASL gloss",
                      ASL_ENGLISH_PAIRS, "natural_to_sign", "ASL", "English"),
    ]

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
    for r in reports:
        print(f"  {r['label']:40s}  parse_rate={r['parsed_rate']:.0%}  avg={r['avg_latency']:.1f}s")
    print("\nHuman evaluation needed: are the outputs semantically correct and fluent?")
    print("Score each direction 0-10. If all 4 score >=7, the primitive is validated.")


if __name__ == "__main__":
    main()
