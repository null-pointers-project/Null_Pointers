"""
Null Pointers - HuggingFace Hub Upload (Adım 12)
-----------------------------------------------------
Eğitilmiş modelleri HuggingFace Hub'a yükler.

ÇALIŞTIRMA (Mac'te, eğitim bittikten sonra):
    python scripts/upload_to_hub.py --captioning --story

Gereksinim:
    .env dosyasında HUGGINGFACE_TOKEN ve HUGGINGFACE_USERNAME ayarlanmış olmalı.
    HF hesabında token'ın write izni olmalı.
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

HF_TOKEN    = os.getenv("HUGGINGFACE_TOKEN", "")
HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME", "")


def upload_captioning_model(repo_name: str = None):
    """Caption modelini (decoder + vocab) HF Hub'a yükler."""
    from huggingface_hub import HfApi, create_repo

    if not HF_TOKEN:
        print("❌ HUGGINGFACE_TOKEN .env'de tanımlı değil!")
        return

    api = HfApi(token=HF_TOKEN)
    repo_id = repo_name or f"{HF_USERNAME}/null-pointers-captioning"

    print(f"📤 Caption model yükleniyor: {repo_id}")

    # Repo oluştur (zaten varsa hata vermez)
    create_repo(repo_id, token=HF_TOKEN, exist_ok=True, private=False)

    model_dir = ROOT_DIR / "checkpoints" / "captioning" / "best_model"
    if not model_dir.exists():
        print(f"❌ Model bulunamadı: {model_dir}")
        print("   Önce eğitimi tamamla: python training/train_captioning.py")
        return

    # Tüm dosyaları yükle
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Null Pointers captioning model",
    )
    print(f"✅ Caption model yüklendi: https://huggingface.co/{repo_id}")
    return repo_id


def upload_story_model(repo_name: str = None):
    """GPT-2 story modelini HF Hub'a yükler."""
    from huggingface_hub import HfApi, create_repo

    if not HF_TOKEN:
        print("❌ HUGGINGFACE_TOKEN .env'de tanımlı değil!")
        return

    api = HfApi(token=HF_TOKEN)
    repo_id = repo_name or f"{HF_USERNAME}/null-pointers-gpt2"

    print(f"📤 Story model yükleniyor: {repo_id}")
    create_repo(repo_id, token=HF_TOKEN, exist_ok=True, private=False)

    model_dir = ROOT_DIR / "checkpoints" / "story" / "best_model"
    if not model_dir.exists():
        print(f"❌ Model bulunamadı: {model_dir}")
        print("   Önce eğitimi tamamla: python training/train_story.py")
        return

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Null Pointers GPT-2 story model",
    )
    print(f"✅ Story model yüklendi: https://huggingface.co/{repo_id}")
    return repo_id


def create_model_cards(captioning_repo: str, story_repo: str):
    """Her model için HuggingFace model card (README) oluşturur."""

    # Caption model card
    caption_card = f"""---
language: en
tags:
- image-captioning
- clip
- lstm
- coco
- deep-learning
license: mit
---

# Null Pointers - Image Captioning Model

## Description
CLIP ViT-B/32 encoder + LSTM decoder ile MS-COCO üzerinde eğitilmiş image captioning modeli.
Akademik Deep Learning + NLP projesi için geliştirilmiştir.

## Architecture
- **Encoder**: OpenAI CLIP ViT-B/32 (frozen, pretrained)
- **Decoder**: 2-layer LSTM with attention
- **Training Data**: MS-COCO 30K subset
- **Evaluation**: BLEU-4 score

## Usage
```python
from models.captioning.model import ImageCaptioningModel
model = ImageCaptioningModel.from_pretrained("{captioning_repo}")
captions = model.generate(images, method="beam")
```

## Performance
| Metric | Score |
|--------|-------|
| BLEU-1 | TBD   |
| BLEU-2 | TBD   |
| BLEU-4 | TBD   |
"""

    # Story model card
    story_card = f"""---
language: en
tags:
- text-generation
- gpt2
- story-generation
- creative-writing
- fine-tuned
license: mit
---

# Null Pointers - GPT-2 Story Generator

## Description
GPT-2 Small (124M) ince ayarlı (fine-tuned) hikaye üretme modeli.
WritingPrompts 50K örnekle eğitildi.

## Architecture
- **Base Model**: GPT-2 Small (124M parameters)
- **Training Data**: WritingPrompts 50K subset
- **Input Format**: `<|prompt|> {{caption}} <|story|>`

## Usage
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("{story_repo}")
model = GPT2LMHeadModel.from_pretrained("{story_repo}")

prompt = "<|prompt|> a dog playing on the beach <|story|>"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                          temperature=0.85, top_p=0.92)
story = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance
| Metric   | Score |
|----------|-------|
| ROUGE-1  | TBD   |
| ROUGE-L  | TBD   |
| PPL      | TBD   |
"""

    # Kaydet
    caption_card_path = ROOT_DIR / "checkpoints" / "captioning" / "best_model" / "README.md"
    story_card_path   = ROOT_DIR / "checkpoints" / "story" / "best_model" / "README.md"

    for path, card in [(caption_card_path, caption_card), (story_card_path, story_card)]:
        if path.parent.exists():
            path.write_text(card)
            print(f"✅ Model card oluşturuldu: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Hub Upload")
    parser.add_argument("--captioning", action="store_true", help="Caption modelini yükle")
    parser.add_argument("--story", action="store_true", help="Story modelini yükle")
    parser.add_argument("--all", action="store_true", help="İkisini de yükle")
    parser.add_argument("--captioning-repo", default=None)
    parser.add_argument("--story-repo", default=None)
    args = parser.parse_args()

    if not HF_TOKEN:
        print("❌ .env dosyasında HUGGINGFACE_TOKEN eksik!")
        print("   .env.example dosyasını .env olarak kopyala ve token'ı doldur.")
        sys.exit(1)

    print("="*60)
    print("HuggingFace Hub Upload")
    print(f"Kullanıcı: {HF_USERNAME}")
    print("="*60)

    captioning_repo = None
    story_repo = None

    if args.captioning or args.all:
        captioning_repo = upload_captioning_model(args.captioning_repo)

    if args.story or args.all:
        story_repo = upload_story_model(args.story_repo)

    if captioning_repo or story_repo:
        create_model_cards(
            captioning_repo or f"{HF_USERNAME}/null-pointers-captioning",
            story_repo or f"{HF_USERNAME}/null-pointers-gpt2",
        )

    print("\n✅ Adım 12 tamamlandı!")
    print("   .env dosyasında model repo adlarını güncelle:")
    if captioning_repo:
        print(f"   CAPTIONING_MODEL_REPO={captioning_repo}")
    if story_repo:
        print(f"   STORY_MODEL_REPO={story_repo}")
    print("\n   Ardından Windows'ta Gradio'yu başlat:")
    print("   python app/app.py")
