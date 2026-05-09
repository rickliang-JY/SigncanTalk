from __future__ import annotations

import json
import os
import time

import gradio as gr

from src.translator import SignTranslator


_translator: SignTranslator | None = None


def get_translator() -> SignTranslator:
    global _translator
    if _translator is None:
        _translator = SignTranslator()
    return _translator


def translate(video_path: str, target_langs: list[str], hint: str) -> tuple[str, str, str, str, str]:
    if not video_path:
        return "", "", "", "", "Please upload a video first."

    t0 = time.time()
    result = get_translator().translate_video(
        video_path,
        target_languages=target_langs or ["English", "Chinese"],
        hint=hint or None,
    )
    elapsed = time.time() - t0

    parsed = result.get("parsed") or {}
    sign_system = parsed.get("sign_system", "—")
    glosses = " | ".join(parsed.get("glosses") or []) or "—"
    translations = parsed.get("translation") or {}
    translation_md = "\n".join(
        f"**{lang}**: {text}" for lang, text in translations.items()
    ) or "—"
    confidence = parsed.get("confidence")
    notes = parsed.get("notes", "")

    meta = (
        f"frames: {result['frame_count']} | latency: {elapsed:.1f}s"
        + (f" | confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else "")
        + (f"\nnotes: {notes}" if notes else "")
    )

    return sign_system, glosses, translation_md, result["raw"], meta


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SigncanTalk MVP") as demo:
        gr.Markdown("# SigncanTalk — Sign Language Translator (MVP)")
        gr.Markdown(
            "Upload a short signing video (≤60s recommended). "
            "Frames are sampled at 1 fps and fed to Gemma 4."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Sign language video", sources=["upload"])
                target_langs = gr.CheckboxGroup(
                    choices=["English", "Chinese", "Japanese", "Spanish", "French"],
                    value=["English", "Chinese"],
                    label="Target translation languages",
                )
                hint = gr.Textbox(
                    label="Optional hint",
                    placeholder="e.g. 'ASL fingerspelling', 'CSL daily greeting'",
                    lines=1,
                )
                run_btn = gr.Button("Translate", variant="primary")

            with gr.Column(scale=1):
                sign_system_out = gr.Textbox(label="Detected sign system", interactive=False)
                glosses_out = gr.Textbox(label="Gloss sequence", interactive=False)
                translation_out = gr.Markdown(label="Translations")
                meta_out = gr.Textbox(label="Meta", interactive=False, lines=2)

        with gr.Accordion("Raw model output", open=False):
            raw_out = gr.Textbox(label="Raw", interactive=False, lines=10)

        run_btn.click(
            fn=translate,
            inputs=[video, target_langs, hint],
            outputs=[sign_system_out, glosses_out, translation_out, raw_out, meta_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "6006")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
