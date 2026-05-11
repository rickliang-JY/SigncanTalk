"""Pure-text cross-sign-language translator: CSL gloss -> Chinese -> English -> ASL gloss.

This module tests whether Gemma 4 can do the language-pivot translation
WITHOUT needing visual input. If this works, the architecture is validated.
"""
from __future__ import annotations

import json
import re
from typing import Any

import torch
from transformers import AutoProcessor

from .config import DEVICE_MAP, DTYPE, MAX_NEW_TOKENS, MODEL_ID
from .translator import _MODEL_CLS


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


CROSS_SIGN_SYSTEM_PROMPT = """You are an expert sign language translator working across CSL (Chinese Sign Language) and ASL (American Sign Language).

Your task: translate a CSL gloss sequence through a 3-stage pipeline:
  Stage 1: CSL gloss -> natural Chinese
  Stage 2: Chinese -> natural English
  Stage 3: English -> ASL gloss

Key grammar rules:

CSL grammar:
- Often topic-comment structure: TIME + TOPIC + COMMENT
- Time markers usually appear first
- Verbs often appear at the end
- Glosses are given as Chinese characters

ASL grammar:
- Topic-comment structure: TIME + TOPIC + COMMENT
- Time markers first: TOMORROW, YESTERDAY, NOW, LAST-WEEK
- Proper nouns are fingerspelled, written as "FS:NAME" (e.g., "FS:BEIJING")
- Pronouns: IX-1 (I/me), IX-2 (you), IX-3 (he/she/it)
- Negation NOT comes after the verb: "I GO NOT" not "I NOT GO"
- ASL glosses use ALL CAPS English words, hyphens for compounds (e.g., "WANT-TO", "DON'T-KNOW")
- Yes/no questions get raised eyebrow marker: append "(Q)" to indicate

Examples:

Example 1:
Input CSL: ["明天", "北京", "我", "去"]
Chinese: "我明天去北京。"
English: "I'm going to Beijing tomorrow."
ASL gloss: ["TOMORROW", "FS:BEIJING", "IX-1", "GO"]

Example 2:
Input CSL: ["昨天", "妈妈", "家", "我", "吃饭"]
Chinese: "昨天我在妈妈家吃饭。"
English: "Yesterday I ate at my mother's house."
ASL gloss: ["YESTERDAY", "MOTHER", "POSS-3", "HOUSE", "IX-1", "EAT"]

Example 3:
Input CSL: ["谢谢", "你", "帮助", "我"]
Chinese: "谢谢你帮助我。"
English: "Thank you for helping me."
ASL gloss: ["THANK-YOU", "IX-2", "HELP", "IX-1"]

Output STRICTLY this JSON schema, no prose, no markdown fences:
{
  "chinese": "<natural Chinese sentence>",
  "english": "<natural English sentence>",
  "asl_gloss": ["GLOSS1", "GLOSS2", ...],
  "asl_grammar_notes": "<brief note on grammar transformation>"
}"""


def build_user_prompt(csl_gloss: list[str]) -> str:
    return f"""Translate the following CSL gloss sequence:

Input CSL: {json.dumps(csl_gloss, ensure_ascii=False)}

Apply the 3-stage pipeline and output JSON."""


class TextCrossSignTranslator:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = _MODEL_CLS.from_pretrained(
            model_id,
            device_map=DEVICE_MAP,
            dtype=_DTYPE_MAP[DTYPE],
        )
        self.model.eval()

    @torch.inference_mode()
    def translate(self, csl_gloss: list[str]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CROSS_SIGN_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": build_user_prompt(csl_gloss)}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
        )

        trimmed = generated[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(trimmed, skip_special_tokens=True)

        return {
            "input_csl": csl_gloss,
            "raw": text,
            "parsed": _try_parse_json(text),
        }


def _try_parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else _extract_first_object(text)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _extract_first_object(text: str) -> str | None:
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None
