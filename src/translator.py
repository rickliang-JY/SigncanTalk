from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def _load_multimodal_model_class():
    import transformers as _tf
    for name in (
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        cls = getattr(_tf, name, None)
        if cls is not None:
            return cls, name
    raise ImportError(
        "No suitable multimodal model class found in transformers. "
        "Try: pip install -U transformers"
    )


_MODEL_CLS, _MODEL_CLS_NAME = _load_multimodal_model_class()

from .config import (
    DEVICE_MAP,
    DTYPE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    TOP_P,
)
from .frame_extractor import extract_frames
from .prompts import SYSTEM_PROMPT, build_user_text


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class SignTranslator:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = _MODEL_CLS.from_pretrained(
            model_id,
            device_map=DEVICE_MAP,
            dtype=_DTYPE_MAP[DTYPE],
        )
        self.model.eval()
        print(f"[SignTranslator] loaded model with class: {_MODEL_CLS_NAME}")

    @torch.inference_mode()
    def translate_video(
        self,
        video_path: str | Path,
        target_languages: list[str] | None = None,
        hint: str | None = None,
    ) -> dict[str, Any]:
        frames = extract_frames(video_path)
        return self.translate_frames(frames, target_languages, hint)

    @torch.inference_mode()
    def translate_frames(
        self,
        frames: list[Image.Image],
        target_languages: list[str] | None = None,
        hint: str | None = None,
    ) -> dict[str, Any]:
        if not frames:
            raise ValueError("No frames provided")

        target_languages = target_languages or ["English", "Chinese"]
        user_text = build_user_text(target_languages, hint)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": f} for f in frames],
                    {"type": "text", "text": user_text},
                ],
            },
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
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=TEMPERATURE > 0,
        )

        trimmed = generated[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(trimmed, skip_special_tokens=True)

        return {
            "raw": text,
            "parsed": _try_parse_json(text),
            "frame_count": len(frames),
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
