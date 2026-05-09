"""
Null Pointers - GPT-2 Fine-tuning (Adım 8)
-----------------------------------------------
GPT-2 Small'u WritingPrompts üzerinde fine-tune eder.

Akademik açıklama:
- GPT-2 (Radford et al., 2019): 124M parametreli causal dil modeli
- Fine-tuning stratejisi: Full fine-tune (tüm katmanlar), küçük LR (5e-5)
- Alternative: LoRA (PEFT) ile sadece adaptör katmanları eğitilir → daha az RAM
- Gradient checkpointing: RAM'i 3-4x azaltır, biraz daha yavaş
- Input format: "<|prompt|> {caption} <|story|> {story} <|endoftext|>"

ÇALIŞTIRMA (Mac M4'te):
    python training/train_story.py
    python training/train_story.py --use-lora  # RAM tasarrufu için
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import yaml
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader, random_split

from models.captioning.dataset import WritingPromptsDataset


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✅ Device: Apple MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    else:
        print("⚠️  Device: CPU")
        return torch.device("cpu")


# ─── Tokenizer Setup ──────────────────────────────────────────────────────────

def setup_tokenizer() -> GPT2Tokenizer:
    """
    GPT-2 tokenizer'ı özel tokenlarla hazırla.

    Özel tokenlar:
    - <|prompt|>: Caption/prompt başlangıcı
    - <|story|>:  Hikaye başlangıcı (model bu tokendan sonra üretir)
    - pad_token:  GPT-2'nin yoktur, eos_token kullanılır

    Bu tokenlar vocabulary'ye eklendikten sonra model.resize_token_embeddings() gerekir!
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Özel tokenlar ekle
    special_tokens = {
        "additional_special_tokens": ["<|prompt|>", "<|story|>"],
        "pad_token": tokenizer.eos_token,  # GPT-2'nin pad token'ı yok
    }
    tokenizer.add_special_tokens(special_tokens)

    print(f"✅ GPT-2 Tokenizer hazır")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Özel tokenlar: <|prompt|>, <|story|>")
    print(f"   Pad token: {tokenizer.pad_token}")

    return tokenizer


# ─── Model Setup ──────────────────────────────────────────────────────────────

def setup_model(
    tokenizer: GPT2Tokenizer,
    use_lora: bool = False,
    gradient_checkpointing: bool = True,
) -> GPT2LMHeadModel:
    """
    GPT-2 modelini fine-tune için hazırla.

    Args:
        use_lora: True ise LoRA adaptörleri ekle (RAM tasarrufu)
        gradient_checkpointing: Aktif ederse RAM azalır, speed düşer
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Vocabulary genişlet (özel tokenlar için)
    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ GPT-2 Small yüklendi")
    print(f"   Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")

    # Gradient checkpointing (RAM tasarrufu)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("   Gradient checkpointing: AÇIK (RAM ~3x azalır)")

    # LoRA (opsiyonel)
    if use_lora:
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,                    # Rank — düşük = az parametre
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],  # GPT-2 attention
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"   LoRA aktif: {trainable:,}/{total:,} parametre eğitilecek "
                  f"({100*trainable/total:.1f}%)")
        except ImportError:
            print("   ⚠️  PEFT yüklü değil, full fine-tune yapılacak")

    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_story_model(args):
    print("="*60)
    print("Null Pointers — GPT-2 Story Fine-tuning")
    print("="*60)

    # Config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    log_cfg   = config["logging"]

    device = get_device()

    # Tokenizer & Model
    tokenizer = setup_tokenizer()
    model = setup_model(
        tokenizer,
        use_lora=args.use_lora,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )
    model = model.to(device)

    # Dataset
    wp_path = ROOT_DIR / "data" / "writing_prompts" / "writing_prompts_subset.json"
    if not wp_path.exists():
        print(f"❌ WritingPrompts bulunamadı: {wp_path}")
        print("   Önce çalıştır: python scripts/download_data.py --dataset writing_prompts")
        return

    full_dataset = WritingPromptsDataset(
        subset_json=str(wp_path),
        tokenizer=tokenizer,
        max_length=512,
    )

    # Train/Val split
    n = len(full_dataset)
    n_val = int(n * 0.1)
    n_train = n - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # DataCollator: labels = input_ids (causal LM, MLM=False)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["story_batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["story_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=False,
    )

    print(f"\n📊 Dataset:")
    print(f"   Train: {len(train_set)} örnek ({len(train_loader)} batch)")
    print(f"   Val:   {len(val_set)} örnek ({len(val_loader)} batch)")

    # Optimizer & Scheduler
    total_steps = len(train_loader) * train_cfg["story_epochs"] // \
                  train_cfg["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * train_cfg["story_warmup_ratio"])

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["story_lr"],
        weight_decay=train_cfg["story_weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n🚀 Eğitim başlıyor:")
    print(f"   Epochs: {train_cfg['story_epochs']}")
    print(f"   LR: {train_cfg['story_lr']}")
    print(f"   Gradient accumulation: {train_cfg['gradient_accumulation_steps']} steps")
    print(f"   Total steps: {total_steps}, Warmup: {warmup_steps}\n")

    ckpt_dir = ROOT_DIR / log_cfg["checkpoint_dir"].lstrip("./") / "story"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    grad_accum = train_cfg["gradient_accumulation_steps"]

    for epoch in range(1, train_cfg["story_epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for step, batch in enumerate(pbar):
            # Tensörleri device'a taşı
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum  # Gradient accumulation için normalize
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += outputs.loss.item()
            avg_loss = total_train_loss / (step + 1)
            ppl = math.exp(min(avg_loss, 10))
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "ppl": f"{ppl:.1f}"})

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss   = total_val_loss / len(val_loader)
        train_ppl = math.exp(min(avg_train_loss, 10))
        val_ppl   = math.exp(min(avg_val_loss, 10))

        print(f"\nEpoch {epoch}/{train_cfg['story_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.1f}) | "
              f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.1f})")

        # Best model kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = ckpt_dir / "best_model"
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            print(f"   🏆 Best model kaydedildi! ({save_path})")

        # Epoch checkpoint
        model.save_pretrained(str(ckpt_dir / f"epoch_{epoch}"))
        tokenizer.save_pretrained(str(ckpt_dir / f"epoch_{epoch}"))

    print(f"\n{'='*60}")
    print(f"✅ GPT-2 Fine-tuning tamamlandı!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Model: {ckpt_dir}/best_model/")
    print(f"{'='*60}")
    print(f"\n✅ Adım 8 tamamlandı!")
    print(f"   Bir sonraki adım: generate.py (Story inference)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=str(ROOT_DIR / "configs" / "config.yaml"))
    parser.add_argument("--use-lora", action="store_true",
                        help="LoRA fine-tuning kullan (daha az RAM)")
    args = parser.parse_args()
    train_story_model(args)
