"""
Null Pointers - Gradio Arayüzü (Adım 11)
--------------------------------------------
Kullanıcıdan fotoğraf alır → Caption üretir → Hikaye/Şiir üretir.

ÇALIŞTIRMA (Windows'ta):
    python app/app.py

    # .env içinde HUGGINGFACE_TOKEN ve model repo ayarları olmalı!
    # Varsayılan: HuggingFace Inference API kullanır (local model gerektirmez)

Mimari:
    [Gradio UI] → [CLIP+LSTM API / Local] → [Caption]
                                                ↓
               [Gradio UI] → [GPT-2 API / Local] → [Hikaye/Şiir]
"""

import os
import sys
import io
import base64
import json
import time
import requests
from pathlib import Path
from typing import Optional, Tuple

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import gradio as gr
from PIL import Image
from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

# ─── Config ───────────────────────────────────────────────────────────────────

HF_TOKEN          = os.getenv("HUGGINGFACE_TOKEN", "")
CAPTIONING_REPO   = os.getenv("CAPTIONING_MODEL_REPO", "")
STORY_REPO        = os.getenv("STORY_MODEL_REPO", "")
USE_LOCAL_MODEL   = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
CAPTION_MODEL_DIR = ROOT_DIR / "checkpoints" / "captioning" / "best_model"
STORY_MODEL_DIR   = ROOT_DIR / "checkpoints" / "story" / "best_model"


# ─── Model Loading ────────────────────────────────────────────────────────────

class ModelManager:
    """
    Modelleri lazy loading ile yönetir.
    İlk istekte yükler, sonra cache'de tutar.
    Windows'ta CPU inference, Mac'te MPS kullanılır.
    """

    def __init__(self):
        self._captioning_model = None
        self._story_generator  = None
        self._device = self._get_device()
        print(f"🖥️  Device: {self._device}")

    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_captioning_model(self):
        """Caption modelini yükle (lazy)."""
        if self._captioning_model is not None:
            return self._captioning_model

        if USE_LOCAL_MODEL and CAPTION_MODEL_DIR.exists():
            print("📂 Caption model: Local'den yükleniyor...")
            from models.captioning.model import ImageCaptioningModel
            self._captioning_model = ImageCaptioningModel.from_pretrained(
                str(CAPTION_MODEL_DIR), device=self._device
            )
        else:
            print(f"☁️  Caption model: HuggingFace API kullanılacak ({CAPTIONING_REPO})")
            self._captioning_model = "api"  # API modunu işaret eder

        return self._captioning_model

    def get_story_generator(self):
        """Story modelini yükle (lazy)."""
        if self._story_generator is not None:
            return self._story_generator

        if USE_LOCAL_MODEL and STORY_MODEL_DIR.exists():
            print("📂 Story model: Local'den yükleniyor...")
            from models.story.generate import StoryGenerator
            self._story_generator = StoryGenerator(str(STORY_MODEL_DIR), self._device)
        else:
            print(f"☁️  Story model: HuggingFace API kullanılacak ({STORY_REPO})")
            self._story_generator = "api"

        return self._story_generator


model_manager = ModelManager()


# ─── Inference Functions ──────────────────────────────────────────────────────

def generate_caption(image: Image.Image) -> str:
    """
    Görüntüden caption üretir.
    Local model veya HF Inference API kullanır.
    """
    if image is None:
        return ""

    model = model_manager.get_captioning_model()

    # HF Inference API modu
    if model == "api":
        return _caption_via_hf_api(image)

    # Local model modu
    try:
        import clip
        _, preprocess = clip.load("ViT-B/32", device=model_manager._device)
        img_tensor = preprocess(image).unsqueeze(0).to(model_manager._device)
        captions = model.generate(img_tensor, method="beam", beam_size=5)
        return captions[0] if captions else "could not generate caption"
    except Exception as e:
        return f"Caption hatası: {str(e)}"


def _caption_via_hf_api(image: Image.Image) -> str:
    """HuggingFace Inference API ile caption üretir."""
    if not CAPTIONING_REPO or not HF_TOKEN:
        # HF API yoksa, blip-image-captioning ile fallback
        return _caption_via_blip_api(image)

    # Görüntüyü bytes'a çevir
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    api_url = f"https://api-inference.huggingface.co/models/{CAPTIONING_REPO}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)
    except Exception as e:
        return _caption_via_blip_api(image)


def _caption_via_blip_api(image: Image.Image) -> str:
    """
    Fallback: Salesforce BLIP modelini HF API üzerinden kullan.
    Bu model herkese açık, token gerektirmez (rate limit var).
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        response = requests.post(api_url, headers=headers, data=img_bytes, timeout=30)
        result = response.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "a photo")
        return "a photo"
    except Exception:
        return "a beautiful scene"


def generate_story(
    caption: str,
    mode: str,
    max_length: int,
    temperature: float,
) -> str:
    """
    Caption'dan hikaye/şiir üretir.
    """
    if not caption.strip():
        return "Önce bir fotoğraf yükleyin!"

    generator = model_manager.get_story_generator()
    mode_key = mode.lower().replace(" ", "")

    # Mode mapping
    if "şiir" in mode.lower() or "poem" in mode.lower():
        mode_key = "poem"
    elif "continuation" in mode.lower():
        mode_key = "continuation"
    else:
        mode_key = "story"

    # HF API modu
    if generator == "api":
        return _story_via_hf_api(caption, mode_key, max_length, temperature)

    # Local model
    try:
        story = generator.generate(
            caption=caption,
            mode=mode_key,
            max_new_tokens=max_length,
            temperature=temperature,
        )
        return story
    except Exception as e:
        return f"Hikaye üretme hatası: {str(e)}"


def _story_via_hf_api(
    caption: str,
    mode: str,
    max_length: int,
    temperature: float,
) -> str:
    """HuggingFace Inference API ile hikaye üretir."""
    if not STORY_REPO or not HF_TOKEN:
        return _story_via_gpt2_api(caption, mode, max_length, temperature)

    api_url = f"https://api-inference.huggingface.co/models/{STORY_REPO}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    if mode == "poem":
        prompt = f"<|prompt|> Write a poem inspired by: {caption} <|story|>"
    else:
        prompt = f"<|prompt|> {caption} <|story|>"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": 0.92,
            "do_sample": True,
            "return_full_text": False,
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        result = response.json()
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "")
            # Özel tokenları temizle
            for token in ["<|prompt|>", "<|story|>", "<|endoftext|>"]:
                text = text.replace(token, "")
            return text.strip()
        return str(result)
    except Exception as e:
        return _story_via_gpt2_api(caption, mode, max_length, temperature)


def _story_via_gpt2_api(caption, mode, max_length, temperature):
    """Fallback: Public GPT-2 API."""
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    if mode == "poem":
        prompt = f"A poem about {caption}:\n\n"
    else:
        prompt = f"Once upon a time, {caption}. "

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": min(max_length, 200),
            "temperature": temperature,
            "do_sample": True,
            "return_full_text": False,
        }
    }
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=30)
        result = r.json()
        if isinstance(result, list):
            return prompt + result[0].get("generated_text", "")
        return "API geçici olarak kullanılamıyor. Lütfen tekrar deneyin."
    except Exception:
        return "Bağlantı hatası. İnternet bağlantınızı kontrol edin."


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def full_pipeline(
    image,
    output_mode: str,
    story_length: int,
    creativity: float,
) -> Tuple[str, str]:
    """
    Tam pipeline: Fotoğraf → Caption → Hikaye/Şiir

    Returns:
        (caption, story)
    """
    if image is None:
        return "⚠️ Lütfen bir fotoğraf yükleyin.", ""

    # Convert numpy array to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Step 1: Caption
    with gr.Progress() as progress:
        progress(0.3, desc="🔍 Fotoğraf analiz ediliyor...")
        caption = generate_caption(image)

        progress(0.7, desc="✍️ Hikaye yazılıyor...")
        story = generate_story(caption, output_mode, story_length, creativity)

    return caption, story


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    """Modern Gradio arayüzü."""

    # CSS
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .title-area { text-align: center; padding: 20px; }
    .output-box { border-radius: 12px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Null Pointers 📸✍️",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        with gr.Row(elem_classes="title-area"):
            gr.HTML("""
            <div style="text-align:center; padding: 10px 0 20px 0;">
                <h1 style="font-size:2.5em; margin:0; background: linear-gradient(135deg, #7c3aed, #a78bfa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    📸 Null Pointers ✍️
                </h1>
                <p style="color:#94a3b8; margin-top:8px; font-size:1.1em;">
                    Fotoğrafını yükle · Dünyayı keşfet · Hikayeni yaz
                </p>
                <p style="color:#64748b; font-size:0.9em;">
                    Powered by CLIP + LSTM + GPT-2
                </p>
            </div>
            """)

        # ── Main Layout ────────────────────────────────────────────────────────
        with gr.Row():
            # Sol: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Fotoğraf Yükle")
                image_input = gr.Image(
                    type="pil",
                    label="Fotoğraf",
                    elem_id="image_upload",
                    height=300,
                )

                gr.Markdown("### ⚙️ Ayarlar")
                output_mode = gr.Radio(
                    choices=["Hikaye", "Şiir", "Continuation"],
                    value="Hikaye",
                    label="Çıktı Türü",
                    elem_id="output_mode",
                )
                story_length = gr.Slider(
                    minimum=50,
                    maximum=400,
                    value=200,
                    step=50,
                    label="Metin Uzunluğu (token)",
                    elem_id="story_length",
                )
                creativity = gr.Slider(
                    minimum=0.5,
                    maximum=1.2,
                    value=0.85,
                    step=0.05,
                    label="🎨 Yaratıcılık (temperature)",
                    elem_id="creativity",
                )

                generate_btn = gr.Button(
                    "✨ Hikaye Oluştur",
                    variant="primary",
                    size="lg",
                    elem_id="generate_btn",
                )

            # Sağ: Output
            with gr.Column(scale=1):
                gr.Markdown("### 🏷️ Fotoğraf Açıklaması (Caption)")
                caption_output = gr.Textbox(
                    label="AI Caption",
                    lines=3,
                    placeholder="Fotoğrafı yüklediğinizde burada görünecek...",
                    elem_id="caption_output",
                    elem_classes="output-box",
                    interactive=True,  # Kullanıcı düzenleyebilir
                    show_copy_button=True,
                )

                gr.Markdown("### 📖 Oluşturulan Metin")
                story_output = gr.Textbox(
                    label="Hikaye / Şiir",
                    lines=12,
                    placeholder="Hikayeniz burada görünecek...",
                    elem_id="story_output",
                    elem_classes="output-box",
                    show_copy_button=True,
                )

                # Yeniden üret (sadece story, caption aynı)
                regenerate_btn = gr.Button(
                    "🔄 Farklı Hikaye Üret",
                    variant="secondary",
                    elem_id="regenerate_btn",
                )

        # ── Examples ───────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🎯 Örnek Captionlar ile Dene")

        example_captions = [
            "a golden retriever playing on a sunny beach",
            "an old lighthouse standing alone in a stormy sea",
            "two children sharing an umbrella in the rain",
            "a cat sitting on a windowsill watching snow fall",
            "a mountain climber reaching the summit at sunrise",
        ]

        with gr.Row():
            for cap in example_captions[:3]:
                gr.Button(f'"{cap[:40]}..."', size="sm", elem_id=f"ex_{cap[:10]}").click(
                    fn=lambda c=cap: c,
                    outputs=caption_output,
                )

        # ── How it Works ───────────────────────────────────────────────────────
        with gr.Accordion("🔬 Nasıl Çalışır? (Akademik Detaylar)", open=False):
            gr.Markdown("""
            ## Mimari

            ```
            [Fotoğraf] ──→ CLIP ViT-B/32 (Encoder) ──→ 512-dim feature
                                                              ↓
                                                       LSTM Decoder ──→ [Caption]
                                                              ↓
                                                       GPT-2 (Fine-tuned) ──→ [Hikaye/Şiir]
            ```

            ### 1. Image Captioning (CLIP + LSTM)
            - **Encoder**: OpenAI CLIP ViT-B/32 — 400M görüntü-metin çiftiyle önceden eğitilmiş
            - **Decoder**: 2-katmanlı LSTM — MS-COCO üzerinde sıfırdan eğitildi
            - **Dataset**: MS-COCO 30K subset
            - **Metrik**: BLEU-4 ≥ 0.25 hedefi

            ### 2. Story Generation (GPT-2)
            - **Model**: GPT-2 Small (124M parametre)
            - **Fine-tuning**: WritingPrompts 50K hikaye üzerinde 3 epoch
            - **Sampling**: Nucleus (top-p=0.92) + Temperature (0.85)
            - **Metrik**: ROUGE-L, Perplexity ≤ 30 hedefi

            ### Platform
            - Eğitim: Apple M4 Mac Mini (MPS backend)
            - Deployment: HuggingFace Spaces (CPU/GPU)
            """)

        # ── Footer ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; color:#475569; font-size:0.85em; margin-top:20px; padding:10px;">
            Null Pointers · Akademik Deep Learning + NLP Projesi<br>
            CLIP ViT-B/32 + LSTM Decoder + GPT-2 Fine-tune
        </div>
        """)

        # ── Event Handlers ─────────────────────────────────────────────────────
        generate_btn.click(
            fn=full_pipeline,
            inputs=[image_input, output_mode, story_length, creativity],
            outputs=[caption_output, story_output],
            show_progress=True,
        )

        regenerate_btn.click(
            fn=generate_story,
            inputs=[caption_output, output_mode, story_length, creativity],
            outputs=story_output,
            show_progress=True,
        )

        # Caption değiştirilince otomatik güncelleme (opsiyonel)
        image_input.change(
            fn=generate_caption,
            inputs=[image_input],
            outputs=[caption_output],
        )

    return demo


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Public URL oluştur")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("Null Pointers — Gradio Arayüzü")
    print("="*60)
    print(f"Local URL: http://localhost:{args.port}")
    if args.share:
        print("Public URL: HuggingFace gradio.live linki oluşturulacak...")
    print()

    demo = create_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
