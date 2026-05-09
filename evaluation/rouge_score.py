"""
Smart Story Teller - ROUGE Score Değerlendirmesi (Adım 10)
----------------------------------------------------------
Fine-tune edilmiş GPT-2'yi ROUGE ve perplexity ile değerlendirir.

Akademik açıklama:
- ROUGE (Lin, 2004): Recall-oriented metrik, özetleme/generation değerlendirmesi
- ROUGE-1: Unigram (tek kelime) örtüşmesi
- ROUGE-2: Bigram örtüşmesi
- ROUGE-L: En uzun ortak alt dizi
- Perplexity: exp(cross_entropy_loss) — dil modelinin "şaşkınlığı"
  Düşük perplexity = daha iyi model (GPT-2 baseline ~30)
"""

import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from models.story.generate import StoryGenerator


def compute_rouge(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L skorlarını hesaplar.

    Args:
        references:  Ground truth hikaye metinleri
        hypotheses:  Model tarafından üretilen metinler

    Returns:
        {
            "rouge1": {"precision": .., "recall": .., "fmeasure": ..},
            "rouge2": {...},
            "rougeL": {...},
        }
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,  # Kelime kökü eşleştirme (better accuracy)
    )

    totals = {
        "rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
        "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
        "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0},
    }

    n = len(references)
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for metric in totals:
            totals[metric]["precision"] += scores[metric].precision
            totals[metric]["recall"]    += scores[metric].recall
            totals[metric]["fmeasure"]  += scores[metric].fmeasure

    # Average
    for metric in totals:
        for stat in totals[metric]:
            totals[metric][stat] /= n

    return totals


def compute_perplexity(
    model_dir: str,
    test_texts: List[str],
    device: str = "cpu",
    max_length: int = 512,
) -> float:
    """
    GPT-2 modelinin perplexity'sini hesaplar.

    Perplexity = exp(average cross-entropy loss over test set)
    Düşük = iyi. GPT-2 baseline Wikipedia üzerinde ~29.

    Returns:
        perplexity: float
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(test_texts, desc="Perplexity hesaplanıyor"):
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            input_ids = encodings["input_ids"]
            n_tokens = input_ids.shape[1]

            outputs = model(input_ids=input_ids, labels=input_ids)
            total_loss   += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_story_model(
    story_model_dir: str,
    wp_subset_json: str,
    device: str = "cpu",
    max_samples: int = 200,
) -> Dict:
    """
    ROUGE + Perplexity ile story modelini değerlendirir.
    """
    print("="*60)
    print("ROUGE + Perplexity Değerlendirmesi")
    print("="*60)

    # Test verisi yükle
    with open(wp_subset_json, encoding='utf-8') as f:
        data = json.load(f)

    # Son %5 test olarak kullan
    n = len(data)
    test_data = data[int(n * 0.95):]

    if max_samples and len(test_data) > max_samples:
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)

    print(f"Test örnekleri: {len(test_data)}")

    # Story Generator yükle
    generator = StoryGenerator(story_model_dir, device=device)

    # Tahminler üret
    print("\n📝 Hikayeler üretiliyor...")
    references = []
    hypotheses = []

    for item in tqdm(test_data):
        prompt = item["prompt"]
        ref_story = item["story"]

        # Model tahminini üret
        hyp_story = generator.generate(
            caption=prompt,
            mode="story",
            max_new_tokens=200,
            temperature=0.85,
        )

        references.append(ref_story)
        hypotheses.append(hyp_story)

    # ROUGE hesapla
    rouge_scores = compute_rouge(references, hypotheses)

    # Perplexity hesapla (test metinleri üzerinde)
    test_texts = [item["text"] for item in test_data]
    perplexity = compute_perplexity(story_model_dir, test_texts, device=device)

    # Sonuçları yazdır
    print(f"\n{'='*60}")
    print("ROUGE + PERPLEXITY SONUÇLARI")
    print(f"{'='*60}")

    for metric, scores in rouge_scores.items():
        f1 = scores["fmeasure"]
        bar = "█" * int(f1 * 40)
        print(f"  {metric.upper()}-F1: {f1:.4f}  {bar}")

    print(f"\n  Perplexity: {perplexity:.2f}")
    print(f"  (GPT-2 baseline ~29, düşük = iyi)")

    if perplexity < 30:
        print(f"  ✅ Perplexity hedefin altında!")
    else:
        print(f"  ⚠️  Perplexity yüksek — daha fazla epoch veya daha fazla veri dene")

    # Kaydet
    results = {
        "model": story_model_dir,
        "n_samples": len(test_data),
        "rouge": rouge_scores,
        "perplexity": perplexity,
    }
    out_path = ROOT_DIR / "evaluation" / "results" / "rouge_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Sonuçlar kaydedildi: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Story model klasörü")
    parser.add_argument("--wp-json",
                        default=str(ROOT_DIR / "data" / "writing_prompts" /
                                    "writing_prompts_subset.json"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    evaluate_story_model(
        story_model_dir=args.model,
        wp_subset_json=args.wp_json,
        device=args.device,
        max_samples=args.max_samples,
    )
    print("\n✅ Adım 10 tamamlandı!")
    print("   Bir sonraki adım: app.py (Gradio arayüzü)")
