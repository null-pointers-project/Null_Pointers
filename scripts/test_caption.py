"""
Görsel Caption Test Scripti
----------------------------
Eğitilmiş CLIP+LSTM modeliyle herhangi bir resim dosyasına caption üretir.

Kullanım:
    python scripts/test_caption.py resim.jpg
    python scripts/test_caption.py resim.jpg --beam-size 5
    python scripts/test_caption.py  # interaktif mod

Eğitim devam ederken AYRI bir terminalde çalıştırabilirsin.
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from models.captioning.model import ImageCaptioningModel


def load_model(device: torch.device, ckpt_path: str = None) -> ImageCaptioningModel:
    if ckpt_path:
        ckpt = Path(ckpt_path)
    else:
        ckpt = ROOT / "checkpoints" / "captioning" / "best_model"
    if not ckpt.exists():
        print(f"❌ Model bulunamadı: {ckpt}")
        sys.exit(1)

    print(f"✅ Model yükleniyor: {ckpt.name}")
    model = ImageCaptioningModel.from_pretrained(str(ckpt), device=device)
    return model.to(device).eval()


def caption_image(model, image_path: str, beam_size: int, penalty: float, device):
    path = Path(image_path)
    if not path.exists():
        print(f"❌ Resim bulunamadı: {image_path}")
        return

    print(f"\n📸 Resim: {path.name}")
    print("-" * 50)

    img = Image.open(path).convert("RGB")
    w, h = img.size
    print(f"   Boyut: {w}x{h} px")

    preprocess = model.encoder.get_preprocess()
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        beam_captions   = model.generate(img_tensor, method="beam", beam_size=beam_size,
                                         repetition_penalty=penalty)
        greedy_captions = model.generate(img_tensor, method="greedy")

    print(f"   🔍 Beam (penalty={penalty}): {beam_captions[0]}")
    print(f"   ⚡ Greedy:               {greedy_captions[0]}")


def main():
    parser = argparse.ArgumentParser(description="Image Caption Test")
    parser.add_argument("images", nargs="*", help="Resim dosya yolları")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--penalty", type=float, default=1.3,
                        help="Repetition penalty (1.0=yok, 1.3=orta, 2.0=sert)")
    parser.add_argument("--checkpoint", default=None,
                        help="Model klasörü (varsayılan: best_model, v2 için: best_model_v2)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    # Device seç
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ Device: Apple MPS")
        else:
            device = torch.device("cpu")
            print("✅ Device: CPU")
    else:
        device = torch.device(args.device)
        print(f"✅ Device: {args.device.upper()}")

    model = load_model(device, args.checkpoint)
    print(f"   Vocabulary: {model.decoder.vocab_size} kelime")

    if args.images:
        for img_path in args.images:
            caption_image(model, img_path, args.beam_size, args.penalty, device)
    else:
        print("\n💡 İnteraktif mod — çıkmak için 'q' yaz\n")
        while True:
            try:
                img_path = input("📂 Resim yolu (veya 'q'): ").strip()
                if img_path.lower() in ("q", "quit", "exit"):
                    break
                if img_path:
                    caption_image(model, img_path, args.beam_size, args.penalty, device)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nÇıkılıyor...")


if __name__ == "__main__":
    main()
