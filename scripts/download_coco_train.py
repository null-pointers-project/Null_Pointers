"""
COCO Train2017 Subset İndirici + Fine-tuning Verisi
---------------------------------------------------
18GB zip YOK — sadece seçilen ~10K görseli tek tek indirir (~1.5GB).
Mevcut val2017 + yeni train subset = 50K caption için hazır.

Kullanım:
    python scripts/download_coco_train.py              # 50K caption (varsayılan)
    python scripts/download_coco_train.py --size 100000 # 100K
    python scripts/download_coco_train.py --dry-run    # İndir, sadece annotation hazırla
"""

import json
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
COCO_DIR = ROOT_DIR / "data" / "coco_subset"
TRAIN_DIR = COCO_DIR / "train2017"

ANN_TRAIN = COCO_DIR / "annotations" / "captions_train2017.json"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{filename}"


def check_annotations():
    """Annotation dosyası var mı?"""
    if not ANN_TRAIN.exists():
        print("❌ captions_train2017.json bulunamadı!")
        print("   Önce çalıştır: python scripts/download_data.py --dataset coco")
        print("   (Val imajlarını zaten indirmişsen annotations da mevcuttur.)")
        return False
    return True


def select_train_captions(target_size: int):
    """
    Train annotations'dan caption seç.
    Her görsel için sadece 1 caption al → daha çeşitli görsel seti.
    """
    print(f"\n📋 Train annotations yükleniyor: {ANN_TRAIN}")
    with open(ANN_TRAIN) as f:
        data = json.load(f)

    # image_id → filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    print(f"   Toplam train görsel: {len(id_to_file):,}")
    print(f"   Toplam train caption: {len(data['annotations']):,}")

    # Her görsel için ilk caption'ı al (çeşitlilik için)
    seen_images = set()
    selected = []

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in seen_images and img_id in id_to_file:
            seen_images.add(img_id)
            selected.append({
                "image_id": img_id,
                "filename": id_to_file[img_id],
                "caption": ann["caption"].strip(),
                "split": "train",
            })
        if len(selected) >= target_size:
            break

    print(f"   Seçilen: {len(selected):,} caption ({len(seen_images):,} benzersiz görsel)")
    return selected


def download_image(item: dict, dry_run: bool = False) -> bool:
    """Tek bir COCO train görselini indirir."""
    filename = item["filename"]
    out_path = TRAIN_DIR / filename

    if out_path.exists():
        return True  # Zaten var

    if dry_run:
        return True

    url = TRAIN_IMAGE_URL.format(filename=filename)
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception:
        return False


def download_images_parallel(items: list, dry_run: bool, workers: int = 8):
    """Görselleri paralel indirir."""
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    already_exists = sum(1 for item in items if (TRAIN_DIR / item["filename"]).exists())
    to_download = len(items) - already_exists

    if to_download == 0:
        print(f"✅ Tüm görseller zaten mevcut ({len(items):,} dosya).")
        return

    print(f"\n📥 {to_download:,} görsel indirilecek (~{to_download * 0.15:.0f}MB tahmini)")
    if dry_run:
        print("   [DRY-RUN] İndirme atlandı.")
        return

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(download_image, item, dry_run): item for item in items}
        with tqdm(total=len(futures), desc="⬇️  Görseller", unit="img") as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    success += 1
                else:
                    fail += 1
                pbar.update(1)

    print(f"   ✅ {success:,} başarılı  |  ❌ {fail:,} başarısız")


def merge_and_save_subset(train_items: list, subset_size: int, output_name: str):
    """
    Mevcut val2017 subset'i ile yeni train verilerini birleştir.
    Çıktı: data/coco_subset/coco_train50k.json (veya 100k)
    """
    output_path = COCO_DIR / output_name

    # Mevcut val verisi
    val_subset = COCO_DIR / "coco_subset.json"
    combined = []
    if val_subset.exists():
        with open(val_subset) as f:
            val_data = json.load(f)
        combined.extend(val_data)
        print(f"   Val verisi: {len(val_data):,} caption eklendi")

    # Yeni train verisi — sadece gerçekten indirilen görselleri ekle
    valid_train = []
    for item in train_items:
        img_path = TRAIN_DIR / item["filename"]
        if img_path.exists():
            valid_train.append({
                "image_id":   item["image_id"],
                "image_path": f"train2017/{item['filename']}",
                "caption":    item["caption"],
                "split":      "train",
            })

    combined.extend(valid_train)
    print(f"   Train verisi: {len(valid_train):,} caption eklendi")
    print(f"   Toplam: {len(combined):,} caption")

    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n✅ Kaydedildi: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=50_000,
                        help="Kaç train caption (varsayılan: 50000)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Paralel indirme worker sayısı")
    parser.add_argument("--dry-run", action="store_true",
                        help="Görselleri indirme, sadece JSON hazırla")
    args = parser.parse_args()

    print("=" * 60)
    print(f"COCO Train Subset İndirici  (hedef: {args.size:,} caption)")
    print("=" * 60)

    if not check_annotations():
        return

    # 1. Caption seç
    train_items = select_train_captions(args.size)

    # 2. Görselleri indir
    download_images_parallel(train_items, dry_run=args.dry_run, workers=args.workers)

    # 3. Birleşik JSON oluştur
    size_label = f"{args.size // 1000}k"
    output_name = f"coco_train{size_label}.json"
    print(f"\n📦 Subset birleştiriliyor → {output_name}")
    output_path = merge_and_save_subset(train_items, args.size, output_name)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅ HAZIR!                                           ║
║                                                      ║
║  Fine-tuning başlatmak için:                         ║
║  python training/train_captioning.py \\               ║
║    --finetune \\                                      ║
║    --data {output_name:<35} ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
