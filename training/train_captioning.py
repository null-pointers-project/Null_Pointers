"""
Null Pointers - Image Captioning Eğitim Döngüsü (Adım 6)
------------------------------------------------------------
CLIP + LSTM modelini MS-COCO üzerinde eğitir.

ÇALIŞTIRMA (Mac M4'te):
    python training/train_captioning.py
    python training/train_captioning.py --config configs/config.yaml
    python training/train_captioning.py --resume checkpoints/captioning/last.pt

Akademik notlar:
- Learning rate scheduler: Cosine annealing → training sonuna doğru LR'yi düşürür
- Early stopping: Val BLEU artmayı bırakırsa eğitimi durdurur (overfitting önleme)
- Gradient clipping: Exploding gradient'ı önler (LSTM'lerde sıklıkla gerekli)
- Checkpoint: Her epoch sonunda en iyi val loss modelini kaydeder
"""

import sys
import os
import json
import time
import math
import argparse
from pathlib import Path

# Project root'u path'e ekle
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from tqdm import tqdm

from models.captioning.model import ImageCaptioningModel, CaptioningLoss
from models.captioning.dataset import Vocabulary, get_coco_loaders


# ─── Config Loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Device Setup ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Platform-aware device seçimi.
    Öncelik sırası: MPS (Apple Silicon) > CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Device: Apple MPS (Metal Performance Shaders)")
        print("   M4 Mac'te GPU hızlı eğitim aktif!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚠️  Device: CPU (yavaş olabilir)")
        print("   MPS aktif değil — macOS 12.3+ ve PyTorch 2.x gerekli")
    return device


# ─── Vocabulary Builder ────────────────────────────────────────────────────────

def build_or_load_vocabulary(
    subset_json: str,
    vocab_save_path: str,
    freq_threshold: int = 5,
) -> Vocabulary:
    """
    Vocabulary'yi COCO subset'ten oluştur veya cache'den yükle.
    """
    if os.path.exists(vocab_save_path):
        print(f"📚 Vocabulary cache'den yükleniyor: {vocab_save_path}")
        return Vocabulary.load(vocab_save_path)

    print("📚 Vocabulary oluşturuluyor (ilk çalıştırma)...")
    with open(subset_json) as f:
        data = json.load(f)

    captions = [item["caption"] for item in data]
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_from_captions(captions)
    vocab.save(vocab_save_path)
    return vocab


# ─── Training Metrics ─────────────────────────────────────────────────────────

class MetricTracker:
    """
    Eğitim metriklerini takip eder ve TensorBoard'a yazar.
    """
    def __init__(self, use_tensorboard: bool = True, log_dir: str = "./runs"):
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_ppl": [], "val_ppl": [],
            "lr": [],
        }
        self.use_tb = use_tensorboard
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"📊 TensorBoard aktif: tensorboard --logdir {log_dir}")
            except ImportError:
                print("⚠️  TensorBoard yüklü değil, sadece konsol loglaması yapılacak.")

    def log(self, epoch: int, **kwargs):
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
            if self.writer:
                self.writer.add_scalar(key, value, epoch)

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()


# ─── Train Epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ImageCaptioningModel,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: CaptioningLoss,
    device: torch.device,
    epoch: int,
    grad_clip: float = 5.0,
    log_every: int = 50,
) -> float:
    """
    Bir epoch boyunca modeli eğitir.

    Returns:
        avg_loss: Epoch boyunca ortalama loss
    """
    model.train()
    # Encoder'ı eval modunda tut (BatchNorm, Dropout için)
    model.encoder.clip_model.eval()

    total_loss = 0.0
    total_batches = len(loader)
    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (images, captions) in enumerate(pbar):
        images   = images.to(device)
        captions = captions.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(images, captions)
        loss = criterion(logits, captions)

        # Backward
        loss.backward()

        # Gradient clipping — LSTM'lerde exploding gradient sık görülür
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        # Perplexity: exp(loss) — dil modeli kalitesi metriği
        ppl = math.exp(min(avg_loss, 10))  # Overflow önleme

        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "ppl": f"{ppl:.1f}",
        })

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            print(f"   Batch {batch_idx+1}/{total_batches} | "
                  f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | "
                  f"Time: {elapsed:.0f}s")

    return total_loss / total_batches


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: ImageCaptioningModel,
    loader,
    criterion: CaptioningLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Validation seti üzerinde loss hesaplar.

    Returns:
        avg_val_loss
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, captions in pbar:
        images   = images.to(device)
        captions = captions.to(device)

        logits = model(images, captions)
        loss = criterion(logits, captions)
        total_loss += loss.item()

        pbar.set_postfix({"val_loss": f"{total_loss/len(pbar):.4f}"})

    return total_loss / len(loader)


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(
    model: ImageCaptioningModel,
    optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: str,
    is_best: bool = False,
):
    """Model checkpoint'ini kaydet."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "decoder_state_dict": model.decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.config,
    }

    # Son checkpoint (her zaman üzerine yaz)
    torch.save(state, Path(checkpoint_dir) / "last.pt")

    # En iyi checkpoint
    if is_best:
        torch.save(state, Path(checkpoint_dir) / "best.pt")
        model.save_pretrained(str(Path(checkpoint_dir) / "best_model"))
        print(f"   🏆 Yeni best model kaydedildi! Val Loss: {val_loss:.4f}")


def load_checkpoint(
    model: ImageCaptioningModel,
    optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """Checkpoint'ten eğitime devam et. Returns: başlangıç epoch'u"""
    print(f"📂 Checkpoint yükleniyor: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.decoder.load_state_dict(state["decoder_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    start_epoch = state["epoch"] + 1

    print(f"   ✅ Epoch {state['epoch']}'den devam ediliyor (Val Loss: {state['val_loss']:.4f})")
    return start_epoch


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Val loss iyileşmezse eğitimi durdurur.

    Args:
        patience: Kaç epoch bekleyeceğiz
        min_delta: Anlamlı iyileşme için minimum fark
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"   ⚠️  EarlyStopping: {self.counter}/{self.patience} "
                  f"(best: {self.best_loss:.4f})")
            return self.counter >= self.patience


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(args):
    print("="*60)
    print("Null Pointers — Image Captioning Eğitimi")
    print("="*60)

    # Config
    config = load_config(args.config)
    model_cfg   = config["model"]
    train_cfg   = config["training"]
    data_cfg    = config["data"]
    log_cfg     = config["logging"]

    # Device
    device = get_device()

    # Paths
    data_dir     = ROOT_DIR / data_cfg["coco_data_dir"].lstrip("./")
    subset_json  = data_dir / "coco_subset.json"
    vocab_path   = data_dir / "vocabulary.json"
    ckpt_dir     = ROOT_DIR / log_cfg["checkpoint_dir"].lstrip("./") / "captioning"

    # Vocabulary
    vocab = build_or_load_vocabulary(
        str(subset_json),
        str(vocab_path),
        freq_threshold=5,
    )
    actual_vocab_size = len(vocab)
    print(f"\n📚 Vocabulary boyutu: {actual_vocab_size}")

    # Model
    model = ImageCaptioningModel(
        vocab_size=actual_vocab_size,
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["lstm_hidden_dim"],
        num_layers=model_cfg["lstm_num_layers"],
        dropout=model_cfg["lstm_dropout"],
        device=str(device),
    ).to(device)
    model.set_vocabulary(vocab)

    # Loss & Optimizer
    criterion = CaptioningLoss(pad_idx=0, label_smoothing=0.1)
    optimizer = Adam(
        model.get_trainable_params(),
        lr=train_cfg["caption_lr"],
        weight_decay=train_cfg["caption_weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["caption_epochs"],
        eta_min=1e-6,
    )

    # DataLoaders
    # CLIP preprocess'i encoder'dan al
    clip_preprocess = model.encoder.get_preprocess()
    train_loader, val_loader, test_loader = get_coco_loaders(
        data_dir=str(data_dir),
        subset_json=str(subset_json),
        vocab=vocab,
        clip_preprocess=clip_preprocess,
        batch_size=train_cfg["caption_batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    # Resume from checkpoint?
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    # Metrics & Early Stopping
    tracker = MetricTracker(
        use_tensorboard=log_cfg["use_tensorboard"],
        log_dir=str(ROOT_DIR / "runs" / "captioning"),
    )
    early_stop = EarlyStopping(patience=5, min_delta=1e-4)
    best_val_loss = float('inf')

    print(f"\n🚀 Eğitim başlıyor: {train_cfg['caption_epochs']} epoch")
    print(f"   Batch size: {train_cfg['caption_batch_size']}")
    print(f"   LR: {train_cfg['caption_lr']}")
    print(f"   Checkpoint: {ckpt_dir}\n")

    # ── Epoch Loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, train_cfg["caption_epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_clip=train_cfg["caption_clip_grad_norm"],
            log_every=log_cfg["log_every_n_steps"],
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # LR schedule
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Perplexity
        train_ppl = math.exp(min(train_loss, 10))
        val_ppl   = math.exp(min(val_loss, 10))

        # Epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch:03d}/{train_cfg['caption_epochs']} "
              f"({epoch_time:.0f}s) | "
              f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.1f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.1f}) | "
              f"LR: {current_lr:.2e}")

        # Metrics log
        tracker.log(epoch,
                    train_loss=train_loss, val_loss=val_loss,
                    train_ppl=train_ppl, val_ppl=val_ppl,
                    lr=current_lr)

        # Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % log_cfg["save_every_n_epochs"] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            str(ckpt_dir), is_best=is_best)

        # Early stopping
        if early_stop(val_loss):
            print(f"\n⛔ Early stopping tetiklendi (epoch {epoch})")
            break

    # ── Training Complete ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Eğitim tamamlandı!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Model: {ckpt_dir}/best_model/")
    print(f"{'='*60}")

    tracker.save_history(str(ckpt_dir / "training_history.json"))
    tracker.close()

    return model


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Null Pointers — Caption Eğitimi")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "config.yaml"),
        help="Config dosyası yolu",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint'ten devam et (örn: checkpoints/captioning/last.pt)",
    )
    args = parser.parse_args()

    train(args)
