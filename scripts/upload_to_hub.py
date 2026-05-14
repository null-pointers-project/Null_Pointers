"""
Null Pointers — Caption Modelini HuggingFace Hub'a Yükle
"""

from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
api = HfApi()
repo_id = "nullpointersproject/null-pointers-caption"

# Repo oluştur
try:
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"✅ Repo: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Repo zaten var: {e}")

# Dosyaları yükle
ckpt = ROOT / "checkpoints" / "captioning" / "best_model"
files = ["config.json", "decoder_weights.pt", "vocabulary.json"]

for f in files:
    path = ckpt / f
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"📤 Yükleniyor: {f} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"   ✅ {f} yüklendi")
    else:
        print(f"   ❌ Bulunamadı: {path}")

print(f"\n🎉 Tamamlandı!")
print(f"   https://huggingface.co/{repo_id}")
