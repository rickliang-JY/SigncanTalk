"""Validate the foundational primitive: sign language <-> natural language.

For ONE sign system (CSL or ASL), test both directions:
  - sign_to_natural: gloss sequence -> natural language sentence
  - natural_to_sign: natural language sentence -> gloss sequence

Cross-sign translation (CSL -> Chinese -> English -> ASL) is only valid if
all 4 primitives below work reliably:
  CSL gloss  <-> Chinese
  ASL gloss  <-> English

This module tests the primitives. The cross-sign chain is built on top later.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

import torch
from transformers import AutoProcessor

from .config import DEVICE_MAP, DTYPE, MAX_NEW_TOKENS, MODEL_ID
from .translator import _MODEL_CLS


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


SignSystem = Literal["CSL", "ASL"]


SIGN_GRAMMAR_NOTES = {
    "CSL": """CSL (Chinese Sign Language) grammar:
- Often topic-comment structure: TIME + TOPIC + COMMENT
- Time markers (今天/明天/昨天) usually appear at the start
- Verbs often appear at the end of clauses
- Glosses are written as Chinese characters/compounds
- Negation 不 typically precedes the verb""",
    "ASL": """ASL (American Sign Language) grammar:
- Topic-comment structure: TIME + TOPIC + COMMENT
- Time markers first: TOMORROW, YESTERDAY, NOW, LAST-WEEK
- Proper nouns fingerspelled as "FS:NAME" (e.g., "FS:BEIJING")
- Pronouns: IX-1 (I/me), IX-2 (you), IX-3 (he/she/it)
- Negation NOT often follows the verb: "I GO NOT"
- Glosses are ALL-CAPS English words, hyphens for compounds (WANT-TO, DON'T-KNOW)
- Yes/no questions marked with "(Q)" at end""",
}


SIGN_TO_NATURAL_PROMPT = """You are an expert {sign_system} interpreter.
Task: convert a {sign_system} gloss sequence into a natural, fluent {natural_lang} sentence.

{grammar_notes}

The gloss sequence reflects {sign_system} grammar. Your output must reorder/add
words/particles as needed so the {natural_lang} sentence sounds natural.
Preserve meaning exactly; do not invent content not implied by the glosses.

Examples for CSL -> Chinese:
  Input:  ["明天", "北京", "我", "去"]
  Output: {{"sentence": "我明天去北京。", "notes": "时间词前置，CSL 动词在末位调整到自然位置"}}

Examples for ASL -> English:
  Input:  ["TOMORROW", "FS:BEIJING", "IX-1", "GO"]
  Output: {{"sentence": "I'm going to Beijing tomorrow.", "notes": "ASL topic-comment reordered to SVO"}}

Output STRICT JSON only, no markdown:
{{
  "sentence": "<natural {natural_lang} sentence>",
  "notes": "<one-line note on grammar/word adjustments>"
}}"""


NATURAL_TO_SIGN_PROMPT = """You are an expert {sign_system} interpreter.
Task: convert a natural {natural_lang} sentence into a {sign_system} gloss sequence.

{grammar_notes}

Reorder/drop function words (articles, copulas, prepositions) as appropriate.
Use the gloss conventions for {sign_system}.

Examples for Chinese -> CSL:
  Input:  "我明天去北京。"
  Output: {{"gloss": ["明天", "北京", "我", "去"], "notes": "时间词前置，topic-comment"}}

Examples for English -> ASL:
  Input:  "I'm going to Beijing tomorrow."
  Output: {{"gloss": ["TOMORROW", "FS:BEIJING", "IX-1", "GO"], "notes": "time-first, proper noun fingerspelled"}}

Output STRICT JSON only, no markdown:
{{
  "gloss": ["GLOSS1", "GLOSS2", ...],
  "notes": "<one-line note on grammar/reordering>"
}}"""


class SignNaturalTranslator:
    """Bidirectional translator between ONE sign system and its paired natural language."""

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
    def sign_to_natural(
        self, gloss: list[str], sign_system: SignSystem, natural_lang: str
    ) -> dict[str, Any]:
        system_prompt = SIGN_TO_NATURAL_PROMPT.format(
            sign_system=sign_system,
            natural_lang=natural_lang,
            grammar_notes=SIGN_GRAMMAR_NOTES[sign_system],
        )
        user_prompt = (
            f"Convert this {sign_system} gloss sequence to natural {natural_lang}:\n"
            f"{json.dumps(gloss, ensure_ascii=False)}"
        )
        return self._generate(system_prompt, user_prompt, input_repr=gloss)

    @torch.inference_mode()
    def natural_to_sign(
        self, text: str, sign_system: SignSystem, natural_lang: str
    ) -> dict[str, Any]:
        system_prompt = NATURAL_TO_SIGN_PROMPT.format(
            sign_system=sign_system,
            natural_lang=natural_lang,
            grammar_notes=SIGN_GRAMMAR_NOTES[sign_system],
        )
        user_prompt = (
            f"Convert this {natural_lang} sentence to {sign_system} gloss:\n{text}"
        )
        return self._generate(system_prompt, user_prompt, input_repr=text)

    def _generate(
        self, system_prompt: str, user_prompt: str, input_repr: Any
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
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
        return {"input": input_repr, "raw": text, "parsed": _try_parse_json(text)}


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
