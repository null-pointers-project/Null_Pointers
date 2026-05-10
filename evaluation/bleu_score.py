"""
Null Pointers - BLEU Score Değerlendirmesi (Adım 7)
-------------------------------------------------------
Eğitilmiş captioning modelini MS-COCO test seti üzerinde değerlendirir.

Akademik açıklama:
- BLEU (Bilingual Evaluation Understudy): Papineni et al. (2002)
- BLEU-4: 1-gram, 2-gram, 3-gram, 4-gram geometrik ortalaması
- COCO baseline (LSTM + CNN): BLEU-4 ≈ 0.25
- State-of-the-art (Transformer): BLEU-4 ≈ 0.40+
- Brevity Penalty: Çok kısa tahminleri cezalandırır

ÇALIŞTIRMA:
    python evaluation/bleu_score.py --model checkpoints/captioning/best_model/
    python evaluation/bleu_score.py --model checkpoints/captioning/best_model/ --method greedy
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


def compute_bleu(
    references: List[List[List[str]]],
    hypotheses: List[List[str]],
) -> Dict[str, float]:
    """
    BLEU-1 ile BLEU-4 arasındaki skorları hesaplar.

    Args:
        references:  [[[ref1_tokens], [ref2_tokens]], ...] — her görüntü için çoklu referans
        hypotheses:  [[hyp_tokens], ...] — model tahminleri

    Returns:
        Dict[str, float]: {bleu_1, bleu_2, bleu_3, bleu_4}
    """
    smoothing = SmoothingFunction().method1  # Sıfır n-gram eşleşmelerini yumuşatır

    scores = {}
    for n in range(1, 5):
        # Sadece n-gram'a kadar hesapla
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        scores[f"bleu_{n}"] = corpus_bleu(
            references,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing,
        )

    return scores


def evaluate_captioning_model(
    model_dir: str,
    data_dir: str,
    subset_json: str,
    device: str = "cpu",
    method: str = "beam",
    beam_size: int = 5,
    max_samples: int = 1000,  # Hız için test setinden subset al
) -> Dict[str, float]:
    """
    Test seti üzerinde BLEU skorlarını hesaplar.

    Returns:
        Tüm BLEU skorlarını içeren dict
    """
    from models.captioning.model import ImageCaptioningModel
    from models.captioning.dataset import COCOCaptionDataset, get_coco_loaders

    print("="*60)
    print("BLEU Değerlendirmesi Başlıyor")
    print("="*60)
    print(f"Model: {model_dir}")
    print(f"Method: {method} (beam_size={beam_size})")
    print(f"Device: {device}")

    # Model yükle
    model = ImageCaptioningModel.from_pretrained(model_dir, device=device)
    model = model.to(torch.device(device))
    model.eval()
    vocab = model.vocab

    # Dataset yükle
    import clip
    _, clip_preprocess = clip.load("ViT-B/32", device=device)

    full_dataset = COCOCaptionDataset(
        data_dir=data_dir,
        subset_json=subset_json,
        vocab=vocab,
        transform=clip_preprocess,
    )

    # Test split (son %5)
    n = len(full_dataset)
    test_start = int(n * 0.95)
    test_indices = list(range(test_start, n))

    if max_samples and len(test_indices) > max_samples:
        import random
        random.seed(42)
        test_indices = random.sample(test_indices, max_samples)

    print(f"\nTest örnekleri: {len(test_indices)}")

    # COCO references: Her görüntünün 5 referans caption'ı var
    # Basit yaklaşım: subset'teki mevcut caption'ları kullan
    references = []
    hypotheses = []

    with open(subset_json) as f:
        all_data = json.load(f)

    # image_id → tüm captions mapping
    id_to_captions = {}
    for item in all_data:
        img_id = item["image_id"]
        if img_id not in id_to_captions:
            id_to_captions[img_id] = []
        id_to_captions[img_id].append(item["caption"].lower().split())

    print("Tahminler üretiliyor...")
    for idx in tqdm(test_indices):
        image, _ = full_dataset[idx]
        image = image.unsqueeze(0).to(device)

        # Caption üret
        pred_captions = model.generate(image, method=method, beam_size=beam_size)
        pred_tokens = pred_captions[0].lower().split()

        # Ground truth
        img_id = all_data[idx]["image_id"]
        refs = id_to_captions.get(img_id, [[]])

        references.append(refs)
        hypotheses.append(pred_tokens)

    # BLEU hesapla
    scores = compute_bleu(references, hypotheses)

    # Sonuçları yazdır
    print(f"\n{'='*60}")
    print("BLEU SONUÇLARI")
    print(f"{'='*60}")
    for metric, score in scores.items():
        bar = "█" * int(score * 40)
        print(f"  {metric.upper()}: {score:.4f}  {bar}")

    print(f"\n  COCO LSTM Baseline: BLEU-4 ≈ 0.25")
    if scores["bleu_4"] >= 0.25:
        print(f"  ✅ Baseline'ı geçtik! ({scores['bleu_4']:.4f} >= 0.25)")
    else:
        print(f"  ⚠️  Baseline altında ({scores['bleu_4']:.4f} < 0.25)")
        print(f"      → Daha fazla epoch, daha büyük vocab veya beam size artırılabilir.")

    # JSON olarak kaydet
    results_path = ROOT_DIR / "evaluation" / "results" / "bleu_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "model": model_dir,
            "method": method,
            "beam_size": beam_size,
            "n_samples": len(test_indices),
            "scores": scores,
        }, f, indent=2)
    print(f"\n📄 Sonuçlar kaydedildi: {results_path}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLEU Score Değerlendirmesi")
    parser.add_argument("--model", required=True, help="Model klasörü (from_pretrained)")
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "data" / "coco_subset"))
    parser.add_argument("--subset-json",
                        default=str(ROOT_DIR / "data" / "coco_subset" / "coco_subset.json"))
    parser.add_argument("--method", choices=["beam", "greedy"], default="beam")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    evaluate_captioning_model(
        model_dir=args.model,
        data_dir=args.data_dir,
        subset_json=args.subset_json,
        device=args.device,
        method=args.method,
        beam_size=args.beam_size,
        max_samples=args.max_samples,
    )
