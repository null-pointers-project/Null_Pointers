"""
Null Pointers - Data Download Script
-----------------------------------------
Bu script iki veri setini indirir:
1. MS-COCO 2017 (validation set, ~5K görüntü + caption) - image captioning için
   NOT: Full train set 18GB olduğu için validation set (1GB) + annotations kullanıyoruz
   Akademik çalışmada bu yaklaşım kabul görmektedir.

2. WritingPrompts (HuggingFace datasets üzerinden) - story generation için

ÇALIŞTIRMA (Mac'te):
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset coco
    python scripts/download_data.py --dataset writing_prompts
"""

import os
import json
import argparse
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# ─── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent
COCO_DIR = ROOT_DIR / "data" / "coco_subset"
WP_DIR   = ROOT_DIR / "data" / "writing_prompts"

# MS-COCO 2017 URLs (resmi kaynak)
COCO_URLS = {
    "val_images":    "http://images.cocodataset.org/zips/val2017.zip",          # ~1 GB
    "train_images":  "http://images.cocodataset.org/zips/train2017.zip",        # ~18 GB (opsiyonel)
    "annotations":   "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~241 MB
}

COCO_SUBSET_SIZE = 30_000   # Train'den kaç örnek alacağız
WP_SUBSET_SIZE   = 50_000   # WritingPrompts'tan kaç örnek


# ─── Utility ───────────────────────────────────────────────────────────────────

class DownloadProgressBar(tqdm):
    """urllib download için tqdm progress bar."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """URL'den dosya indirir, progress bar gösterir."""
    print(f"\n📥 İndiriliyor: {url.split('/')[-1]}")
    print(f"   → {output_path}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"   ✅ Tamamlandı!")


def extract_zip(zip_path: Path, extract_to: Path):
    """ZIP dosyasını çıkartır."""
    print(f"\n📦 Çıkartılıyor: {zip_path.name} → {extract_to}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"   ✅ Çıkartıldı!")


# ─── MS-COCO Download ──────────────────────────────────────────────────────────

def download_coco(use_train: bool = False):
    """
    MS-COCO veri setini indirir.

    Strateji:
    - Validation set (5K görüntü): Her zaman indirilir, test için kullanılır.
    - Train set (118K görüntü): Opsiyonel, 18GB gerektirir.
    - Annotations: Her iki split için gerekli.

    Akademik not: Val set ile eğitim yaparken caption annotations kullanılır,
    bu yaklaşım transfer learning + small dataset fine-tuning senaryolarında geçerlidir.
    """
    COCO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Annotations indir (her zaman gerekli)
    ann_zip = COCO_DIR / "annotations_trainval2017.zip"
    if not (COCO_DIR / "annotations").exists():
        download_url(COCO_URLS["annotations"], ann_zip)
        extract_zip(ann_zip, COCO_DIR)
        ann_zip.unlink()  # ZIP'i sil, yer açalım
    else:
        print("✅ Annotations zaten mevcut, atlanıyor.")

    # 2. Validation images indir (~1GB, her zaman alınır)
    val_zip = COCO_DIR / "val2017.zip"
    if not (COCO_DIR / "val2017").exists():
        download_url(COCO_URLS["val_images"], val_zip)
        extract_zip(val_zip, COCO_DIR)
        val_zip.unlink()
    else:
        print("✅ val2017 görüntüleri zaten mevcut, atlanıyor.")

    # 3. Train images (opsiyonel, 18GB)
    if use_train:
        train_zip = COCO_DIR / "train2017.zip"
        if not (COCO_DIR / "train2017").exists():
            print("\n⚠️  Train set 18GB! İndirme başlıyor...")
            download_url(COCO_URLS["train_images"], train_zip)
            extract_zip(train_zip, COCO_DIR)
            train_zip.unlink()
        else:
            print("✅ train2017 görüntüleri zaten mevcut, atlanıyor.")

    # 4. Subset oluştur (30K caption-image çifti)
    create_coco_subset()


def create_coco_subset():
    """
    COCO annotations'dan 30K subset oluşturur ve kaydeder.
    Çıktı: data/coco_subset/coco_subset.json

    Format:
    [
        {
            "image_id": 12345,
            "image_path": "val2017/000000012345.jpg",
            "caption": "a dog sitting on a bench"
        },
        ...
    ]
    """
    subset_path = COCO_DIR / "coco_subset.json"
    if subset_path.exists():
        print(f"✅ COCO subset zaten mevcut ({subset_path}), atlanıyor.")
        return

    print("\n🔧 COCO subset oluşturuluyor...")

    # Annotations yükle
    ann_file_val   = COCO_DIR / "annotations" / "captions_val2017.json"
    ann_file_train = COCO_DIR / "annotations" / "captions_train2017.json"

    subset = []

    # Val2017 captions (5K görüntü, ~25K caption - her görüntünün 5 caption'ı var)
    if ann_file_val.exists():
        with open(ann_file_val, 'r') as f:
            val_data = json.load(f)

        # image_id → filename mapping
        id_to_filename = {img['id']: img['file_name'] for img in val_data['images']}

        for ann in val_data['annotations']:
            img_id = ann['image_id']
            if img_id in id_to_filename:
                subset.append({
                    "image_id": img_id,
                    "image_path": f"val2017/{id_to_filename[img_id]}",
                    "caption": ann['caption'].strip(),
                    "split": "val"
                })

    # Train2017 captions (eğer indirilmişse, 30K'ya tamamla)
    if ann_file_train.exists() and len(subset) < COCO_SUBSET_SIZE:
        with open(ann_file_train, 'r') as f:
            train_data = json.load(f)

        id_to_filename_train = {img['id']: img['file_name'] for img in train_data['images']}
        remaining = COCO_SUBSET_SIZE - len(subset)

        for ann in train_data['annotations'][:remaining * 2]:  # Biraz fazla al, filtrele
            img_id = ann['image_id']
            img_file = f"train2017/{id_to_filename_train.get(img_id, '')}"
            img_full_path = COCO_DIR / img_file

            # Sadece gerçekten var olan görüntüleri ekle
            if img_full_path.exists():
                subset.append({
                    "image_id": img_id,
                    "image_path": img_file,
                    "caption": ann['caption'].strip(),
                    "split": "train"
                })

            if len(subset) >= COCO_SUBSET_SIZE:
                break

    # Shuffle ve kaydet
    import random
    random.seed(42)
    random.shuffle(subset)
    subset = subset[:COCO_SUBSET_SIZE]

    with open(subset_path, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"✅ COCO subset oluşturuldu: {len(subset)} örnek → {subset_path}")

    # Split stats
    splits = {}
    for item in subset:
        splits[item['split']] = splits.get(item['split'], 0) + 1
    for split, count in splits.items():
        print(f"   {split}: {count} örnek")


# ─── WritingPrompts Download ───────────────────────────────────────────────────

def download_writing_prompts():
    """
    WritingPrompts veri setini HuggingFace datasets üzerinden indirir.

    Dataset: euclaise/writingprompts (HF Hub'da mevcut)
    50K subset alınır, prompt + story çifti olarak kaydedilir.

    Akademik not: Bu dataset Reddit r/WritingPrompts'tan derlenmiştir.
    GPT-2 fine-tuning için format: "<|prompt|> {prompt} <|story|> {story} <|endoftext|>"
    """
    WP_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = WP_DIR / "writing_prompts_subset.json"

    if subset_path.exists():
        print(f"✅ WritingPrompts subset zaten mevcut, atlanıyor.")
        return

    print("\n📥 WritingPrompts indiriliyor (HuggingFace datasets)...")
    print("   Dataset: euclaise/writingprompts")

    try:
        from datasets import load_dataset

        # HuggingFace'den yükle
        dataset = load_dataset("euclaise/writingprompts", split="train")
        print(f"   Toplam örnek: {len(dataset)}")

        # 50K subset al
        subset_size = min(WP_SUBSET_SIZE, len(dataset))
        subset_data = dataset.select(range(subset_size))

        # GPT-2 fine-tune formatına dönüştür
        processed = []
        for item in tqdm(subset_data, desc="İşleniyor"):
            prompt = item.get('prompt', item.get('title', '')).strip()
            story  = item.get('story', item.get('text', '')).strip()

            # Çok kısa olanları filtrele
            if len(prompt) < 20 or len(story) < 100:
                continue

            # Çok uzun olanları kırp (GPT-2 max 1024 token)
            story_words = story.split()
            if len(story_words) > 400:
                story = ' '.join(story_words[:400])

            processed.append({
                "prompt": prompt,
                "story": story,
                # GPT-2 input formatı
                "text": f"<|prompt|> {prompt} <|story|> {story} <|endoftext|>"
            })

        import random
        random.seed(42)
        random.shuffle(processed)

        with open(subset_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        print(f"✅ WritingPrompts subset kaydedildi: {len(processed)} örnek → {subset_path}")

    except ImportError:
        print("❌ 'datasets' paketi bulunamadı. Yüklemek için: pip install datasets")
    except Exception as e:
        print(f"❌ WritingPrompts indirilirken hata: {e}")
        _download_writing_prompts_fallback()


def _download_writing_prompts_fallback():
    """
    HF datasets çalışmazsa alternatif kaynak.
    Manuel indirme talimatları gösterir.
    """
    print("\n⚠️  Alternatif indirme yöntemi:")
    print("   1. https://huggingface.co/datasets/euclaise/writingprompts adresine git")
    print("   2. 'Files' sekmesinden 'data/train-*.parquet' dosyalarını indir")
    print(f"   3. data/writing_prompts/ klasörüne koy")
    print("   4. python scripts/process_writing_prompts.py --from-parquet")


# ─── Verification ──────────────────────────────────────────────────────────────

def verify_downloads():
    """İndirilen dosyaların varlığını ve boyutunu kontrol eder."""
    print("\n" + "="*50)
    print("VERİFİKASYON")
    print("="*50)

    checks = {
        "COCO annotations": COCO_DIR / "annotations",
        "COCO val images":  COCO_DIR / "val2017",
        "COCO subset JSON": COCO_DIR / "coco_subset.json",
        "WritingPrompts":   WP_DIR / "writing_prompts_subset.json",
    }

    all_ok = True
    for name, path in checks.items():
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"  ✅ {name}: {count} dosya")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {name}: BULUNAMADI ({path})")
            all_ok = False

    # COCO subset istatistikleri
    coco_subset = COCO_DIR / "coco_subset.json"
    if coco_subset.exists():
        with open(coco_subset) as f:
            data = json.load(f)
        print(f"\n  📊 COCO subset: {len(data)} örnek")
        print(f"     Örnek caption: '{data[0]['caption']}'")

    # WP istatistikleri
    wp_subset = WP_DIR / "writing_prompts_subset.json"
    if wp_subset.exists():
        with open(wp_subset, encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n  📊 WritingPrompts subset: {len(data)} örnek")
        print(f"     Örnek prompt: '{data[0]['prompt'][:80]}...'")

    print("\n" + "="*50)
    return all_ok


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Null Pointers - Veri İndirme Scripti")
    parser.add_argument("--dataset", choices=["all", "coco", "writing_prompts"],
                        default="all", help="Hangi veri setini indireceksin?")
    parser.add_argument("--coco-train", action="store_true",
                        help="COCO train set indir (18GB! Dikkatli ol)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Sadece mevcut dosyaları kontrol et")
    args = parser.parse_args()

    print("="*50)
    print("Null Pointers - Veri İndirme")
    print("="*50)
    print(f"Hedef klasörler:")
    print(f"  COCO: {COCO_DIR}")
    print(f"  WritingPrompts: {WP_DIR}")

    if args.verify_only:
        verify_downloads()
        return

    if args.dataset in ("all", "coco"):
        print("\n[1/2] MS-COCO indiriliyor...")
        download_coco(use_train=args.coco_train)

    if args.dataset in ("all", "writing_prompts"):
        print("\n[2/2] WritingPrompts indiriliyor...")
        download_writing_prompts()

    verify_downloads()
    print("\n✅ Adım 2 tamamlandı!")
    print("   Bir sonraki adım: encoder.py (CLIP wrapper)")


if __name__ == "__main__":
    main()
