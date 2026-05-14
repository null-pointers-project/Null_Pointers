"""
NullPointers — AI Story Teller
Flask Backend

Çalıştırma:
    source venv/bin/activate
    python app/server.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from PIL import Image

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Device & Model ────────────────────────────────────────────────────────────

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_caption_model = None

# Her zaman İngilizce üretilir — TR seçilince /api/translate ile çevrilir
SYSTEM_EN = (
    "You are a creative short story writer. "
    "Given an image description, write a vivid, engaging story of 3-4 short paragraphs. "
    "Rules: relate directly to the image, use sensory language, "
    "no violence or profanity, end with a meaningful conclusion, write like a published author."
)

STYLE_HINTS = {
    "default":  "Write a touching, realistic story.",
    "dramatic": "Write a dramatic, emotional story with tension and conflict.",
    "humorous": "Write a light-hearted, witty and humorous story.",
    "mystery":  "Write a mysterious and suspenseful story with an eerie atmosphere.",
    "children": "Write a gentle, imaginative children's story.",
}


def get_caption_model():
    global _caption_model
    if _caption_model is None:
        from models.captioning.model import ImageCaptioningModel
        ckpt = ROOT / "checkpoints" / "captioning" / "best_model_v2"
        if not ckpt.exists():
            ckpt = ROOT / "checkpoints" / "captioning" / "best_model"
        print(f"📦 Caption modeli yükleniyor: {ckpt.name}")
        _caption_model = ImageCaptioningModel.from_pretrained(str(ckpt), device=DEVICE)
        _caption_model = _caption_model.to(DEVICE).eval()
        print("✅ Caption modeli hazır")
    return _caption_model


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/caption", methods=["POST"])
def caption_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Görsel gönderilmedi"}), 400

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
        model = get_caption_model()
        preprocess = model.encoder.get_preprocess()
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            captions = model.generate(
                tensor, method="beam", beam_size=5, repetition_penalty=1.3
            )

        return jsonify({"caption": captions[0], "device": str(DEVICE)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/story", methods=["POST"])
def story_endpoint():
    data = request.json or {}
    caption = data.get("caption", "").strip()
    style   = data.get("style", "default")
    # lang artık burada kullanılmıyor — her zaman İngilizce üretilir

    if not caption:
        return jsonify({"error": "Caption boş"}), 400

    hint = STYLE_HINTS.get(style, STYLE_HINTS["default"])
    user_msg = (
        f'Image description: "{caption}"\n\n'
        f"Style: {hint}\n\n"
        "Write a short story based on this image:"
    )

    def generate():
        import ollama
        try:
            for chunk in ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": SYSTEM_EN},
                    {"role": "user",   "content": user_msg},
                ],
                stream=True,
                options={"temperature": 0.8, "top_p": 0.9, "num_predict": 500},
            ):
                token = chunk.message.content
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/translate", methods=["POST"])
def translate_endpoint():
    """İngilizce hikayeyi Türkçeye çevirir (Google Translate, ücretsiz)."""
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Metin boş"}), 400
    try:
        from deep_translator import GoogleTranslator
        # Uzun metin için paragraflara böl, her birini çevir
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        translated = []
        translator = GoogleTranslator(source="en", target="tr")
        for para in paragraphs:
            translated.append(translator.translate(para))
        return jsonify({"translated": "\n\n".join(translated)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NullPointers — AI Story Teller")
    print(f"  Device: {DEVICE.type.upper()}")
    print("=" * 55)
    print("🔄 Model ön yükleniyor...")
    get_caption_model()
    print("\n🌐 Uygulama: http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
