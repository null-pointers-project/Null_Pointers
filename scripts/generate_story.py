"""
Smart Story Teller — Tam Pipeline
----------------------------------
Resim → CLIP+LSTM Caption → Llama 3.2 (Ollama) Hikaye

Kullanım:
    python scripts/generate_story.py resim.jpg
    python scripts/generate_story.py resim.jpg --lang tr   # Türkçe hikaye
    python scripts/generate_story.py resim.jpg --style dramatic
    python scripts/generate_story.py  # interaktif mod
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from models.captioning.model import ImageCaptioningModel

# Ollama kontrolü
try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ─── Sabit Promptlar ──────────────────────────────────────────────────────────

SYSTEM_PROMPT_EN = """You are a creative short story writer. 
Given an image description (caption), write a vivid, engaging story of 3-4 short paragraphs.
Rules:
- The story must directly relate to the image description
- Use descriptive, sensory language  
- No violence, profanity, or disturbing content
- End with a meaningful or surprising conclusion
- Write naturally, like a published author"""

SYSTEM_PROMPT_TR = """Sen yaratıcı bir kısa hikaye yazarısın.
Sana bir fotoğrafın açıklaması (caption) verilecek. Buna dayanarak 3-4 kısa paragraftan oluşan canlı, etkileyici bir hikaye yaz.
Kurallar:
- Hikaye doğrudan fotoğraf açıklamasıyla ilgili olmalı
- Betimleyici, duyusal bir dil kullan
- Şiddet, küfür veya rahatsız edici içerik olmasın
- Anlamlı veya sürpriz bir sonuçla bitir
- Yayımlanmış bir yazar gibi doğal yaz"""

STYLES = {
    "default":   "Write a touching, realistic story.",
    "dramatic":  "Write a dramatic, emotional story with tension.",
    "humorous":  "Write a light-hearted, humorous story.",
    "mystery":   "Write a mysterious, suspenseful story.",
    "children":  "Write a gentle, imaginative children's story.",
}


# ─── Model Yükleme ────────────────────────────────────────────────────────────

def load_caption_model(device):
    ckpt = ROOT / "checkpoints" / "captioning" / "best_model"
    if not ckpt.exists():
        print("❌ Caption modeli bulunamadı: checkpoints/captioning/best_model/")
        sys.exit(1)
    model = ImageCaptioningModel.from_pretrained(str(ckpt), device=device)
    return model.to(device).eval()


def check_ollama(model_name: str) -> bool:
    """Ollama servisinin çalışıp çalışmadığını ve modelin mevcut olduğunu kontrol et."""
    try:
        models = ollama_client.list()
        names = [m.model for m in models.models]
        # llama3.2 veya llama3.2:latest her ikisini de kabul et
        return any(model_name.split(":")[0] in n for n in names)
    except Exception:
        return False


# ─── Caption ──────────────────────────────────────────────────────────────────

def get_caption(caption_model, image_path, device, beam_size=5, penalty=1.3):
    img = Image.open(image_path).convert("RGB")
    preprocess = caption_model.encoder.get_preprocess()
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        captions = caption_model.generate(
            tensor, method="beam",
            beam_size=beam_size,
            repetition_penalty=penalty,
        )
    return captions[0]


# ─── Hikaye Üretimi (Ollama) ──────────────────────────────────────────────────

def generate_story_ollama(caption: str, model_name: str, lang: str, style: str) -> str:
    system = SYSTEM_PROMPT_TR if lang == "tr" else SYSTEM_PROMPT_EN
    style_hint = STYLES.get(style, STYLES["default"])

    if lang == "tr":
        user_msg = (
            f"Fotoğraf açıklaması: \"{caption}\"\n\n"
            f"Tarz: {style_hint}\n\n"
            "Şimdi bu fotoğraf için Türkçe kısa bir hikaye yaz:"
        )
    else:
        user_msg = (
            f"Image description: \"{caption}\"\n\n"
            f"Style: {style_hint}\n\n"
            "Write a short story based on this image:"
        )

    response = ollama_client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        options={
            "temperature": 0.8,
            "top_p": 0.9,
            "num_predict": 400,
        },
    )
    return response.message.content.strip()


# ─── Ana İşlem ────────────────────────────────────────────────────────────────

def process_image(image_path, caption_model, device, args):
    path = Path(image_path)
    if not path.exists():
        print(f"❌ Resim bulunamadı: {image_path}")
        return

    img = Image.open(path)
    print(f"\n{'='*60}")
    print(f"📸  {path.name}  ({img.width}x{img.height}px)")
    print(f"{'='*60}")

    # 1. Caption
    print("🔍 Caption üretiliyor...")
    caption = get_caption(caption_model, path, device,
                          beam_size=args.beam_size, penalty=args.penalty)
    print(f"   Caption: \"{caption}\"")

    # 2. Hikaye (Ollama)
    print(f"✍️  Hikaye yazılıyor (Llama 3.2, {args.lang.upper()}, stil: {args.style})...")
    story = generate_story_ollama(caption, args.model, args.lang, args.style)

    print(f"\n📖 Hikaye:\n")
    # Her paragrafı girintili yaz
    for para in story.split("\n"):
        if para.strip():
            print(f"   {para}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Smart Story Teller (Ollama)")
    parser.add_argument("images", nargs="*", help="Resim dosyaları")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--penalty",  type=float, default=1.3, help="Caption repetition penalty")
    parser.add_argument("--model",    default="llama3.2", help="Ollama model adı")
    parser.add_argument("--lang",     default="en", choices=["en", "tr"], help="Hikaye dili")
    parser.add_argument("--style",    default="default",
                        choices=list(STYLES.keys()), help="Hikaye tarzı")
    parser.add_argument("--device",   default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"✅ Caption device: {str(device).upper()}")

    # Ollama kontrolü
    if not OLLAMA_AVAILABLE:
        print("❌ 'ollama' paketi yok: pip install ollama")
        sys.exit(1)
    if not check_ollama(args.model):
        print(f"❌ '{args.model}' modeli Ollama'da bulunamadı.")
        print(f"   Çalıştır: ollama pull {args.model}")
        sys.exit(1)
    print(f"✅ Ollama hazır: {args.model}")

    # Caption modeli yükle
    print("📦 Caption modeli yükleniyor...")
    caption_model = load_caption_model(device)
    print("✅ Hazır!\n")

    if args.images:
        for img in args.images:
            process_image(img, caption_model, device, args)
    else:
        print("💡 İnteraktif mod — çıkmak için 'q'\n")
        while True:
            try:
                img_path = input("📂 Resim yolu: ").strip()
                if img_path.lower() in ("q", "quit", "exit", ""):
                    break
                process_image(img_path, caption_model, device, args)
            except (KeyboardInterrupt, EOFError):
                break
        print("Çıkılıyor...")


if __name__ == "__main__":
    main()
