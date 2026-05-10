"""
Null Pointers - Birleşik Encoder-Decoder Modeli (Adım 5)
------------------------------------------------------------
CLIP Encoder + LSTM Decoder'ı tek bir PyTorch modülünde birleştirir.

Bu sınıf:
1. Training ve inference için unified API sağlar
2. Model kaydetme/yükleme işlemlerini yönetir
3. HuggingFace Hub'a upload için hazırlar
4. Parametre sayısı ve model istatistiklerini raporlar
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn

from .encoder import CLIPEncoder
from .decoder import LSTMDecoder
from .dataset import Vocabulary


class ImageCaptioningModel(nn.Module):
    """
    CLIP + LSTM Image Captioning Model.

    Kullanım (training):
        model = ImageCaptioningModel(vocab_size=10000)
        logits = model(images, captions)       # teacher forcing
        loss = criterion(logits, targets)

    Kullanım (inference):
        captions = model.generate(images, method='beam')

    Kullanım (HF Hub):
        model.save_pretrained("./checkpoints/best_model")
        model = ImageCaptioningModel.from_pretrained("./checkpoints/best_model")
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        fine_tune_clip: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        # Konfigürasyonu kaydet (save/load için)
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "fine_tune_clip": fine_tune_clip,
        }
        self.device_name = device

        # ── Alt modüller ────────────────────────────────────────────────────────
        self.encoder = CLIPEncoder(
            embed_dim=embed_dim,
            fine_tune_clip=fine_tune_clip,
            device=device,
        )

        self.decoder = LSTMDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.vocab: Optional[Vocabulary] = None

        print(f"\n{'='*50}")
        print("ImageCaptioningModel oluşturuldu")
        print(f"{'='*50}")
        self._print_model_stats()

    def _print_model_stats(self):
        """Model parametre istatistiklerini yazdırır."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters()
                                if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + decoder_params

        print(f"Encoder (CLIP):  {encoder_params:>12,} param "
              f"({encoder_trainable:,} trainable)")
        print(f"Decoder (LSTM):  {decoder_params:>12,} param (tümü trainable)")
        print(f"TOPLAM:          {total:>12,} param")
        print(f"{'='*50}\n")

    def set_vocabulary(self, vocab: Vocabulary):
        """Vocabulary'yi modelle ilişkilendir (inference için gerekli)."""
        self.vocab = vocab

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            images:   (B, 3, 224, 224) — CLIP preprocess uygulanmış
            captions: (B, max_len) — token IDs [<SOS>...words...<EOS>]

        Returns:
            logits: (B, max_len-1, vocab_size) — CrossEntropyLoss için
        """
        # 1. Görüntüden feature çıkar
        image_features = self.encoder(images)          # (B, embed_dim)

        # 2. LSTM ile caption üret (teacher forcing)
        logits = self.decoder(image_features, captions)  # (B, seq-1, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        method: str = "beam",
        beam_size: int = 5,
        max_len: int = 50,
        repetition_penalty: float = 1.3,
    ) -> List[str]:
        """
        Inference: görüntüden caption üretir.

        Args:
            images:    (B, 3, 224, 224) veya (1, 3, 224, 224)
            method:    "beam" | "greedy"
            beam_size: Beam search için beam sayısı
            max_len:   Maksimum caption uzunluğu

        Returns:
            captions: String listesi
        """
        if self.vocab is None:
            raise ValueError("Vocabulary set edilmedi! model.set_vocabulary(vocab) çağır.")

        self.eval()
        image_features = self.encoder(images)  # (B, embed_dim)

        sos_idx = self.vocab.word2idx[Vocabulary.SOS_TOKEN]
        eos_idx = self.vocab.word2idx[Vocabulary.EOS_TOKEN]

        captions = []
        for i in range(image_features.size(0)):
            feat = image_features[i].unsqueeze(0)  # (1, embed_dim)

            if method == "beam":
                token_ids = self.decoder.generate_beam_search(
                    feat, beam_size=beam_size,
                    sos_idx=sos_idx, eos_idx=eos_idx,
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
            else:
                token_ids = self.decoder.generate_greedy(
                    feat, sos_idx=sos_idx, eos_idx=eos_idx, max_len=max_len,
                )

            caption = self.vocab.decode(token_ids, skip_special=True)
            captions.append(caption)

        return captions

    def save_pretrained(self, save_dir: str):
        """
        Modeli kaydet. Şunları kaydeder:
        - config.json: Model konfigürasyonu
        - decoder_weights.pt: Sadece LSTM ağırlıkları (CLIP kaydedilmez, zaten pretrained)
        - vocabulary.json: Kelime hazinesi
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Config kaydet
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        # Sadece decoder ağırlıklarını kaydet (CLIP frozen, tekrar yüklenecek)
        torch.save(self.decoder.state_dict(), save_path / "decoder_weights.pt")

        # Vocabulary kaydet
        if self.vocab is not None:
            self.vocab.save(str(save_path / "vocabulary.json"))

        print(f"✅ Model kaydedildi: {save_dir}")
        print(f"   - config.json")
        print(f"   - decoder_weights.pt")
        if self.vocab:
            print(f"   - vocabulary.json")

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        device: str = "cpu",
    ) -> "ImageCaptioningModel":
        """
        Kaydedilmiş modeli yükle.

        Args:
            load_dir: save_pretrained() ile kaydedilen klasör
            device:   Hangi device'a yüklenecek

        Returns:
            Yüklenmiş ImageCaptioningModel
        """
        load_path = Path(load_dir)

        # Config yükle
        with open(load_path / "config.json") as f:
            config = json.load(f)

        # Modeli oluştur
        model = cls(device=device, **config)

        # Decoder ağırlıklarını yükle
        decoder_weights = torch.load(
            load_path / "decoder_weights.pt",
            map_location=device,
            weights_only=True,
        )
        model.decoder.load_state_dict(decoder_weights)
        print(f"✅ Decoder ağırlıkları yüklendi.")

        # Vocabulary yükle
        vocab_path = load_path / "vocabulary.json"
        if vocab_path.exists():
            vocab = Vocabulary.load(str(vocab_path))
            model.set_vocabulary(vocab)
            print(f"✅ Vocabulary yüklendi: {len(vocab)} kelime")

        model.eval()
        return model

    def get_trainable_params(self) -> List[nn.Parameter]:
        """
        Sadece eğitilecek parametreleri döndürür.
        CLIP frozen ise sadece decoder parametreleri döner.
        Optimizer'a bu liste verilir.
        """
        return [p for p in self.parameters() if p.requires_grad]


# ─── Loss Function ─────────────────────────────────────────────────────────────

class CaptioningLoss(nn.Module):
    """
    Image captioning için Cross-Entropy loss.

    Önemli: PAD token'ları (idx=0) loss hesabına dahil edilmez.
    Bu, farklı uzunluktaki caption'ların eşit ağırlıklandırılmasını sağlar.

    ignore_index=0: PAD token loss'u yoksay
    label_smoothing: Overfitting önleme (0.1 iyi bir değer)
    """

    def __init__(self, pad_idx: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits:   (B, seq_len-1, vocab_size) — model çıktısı
            captions: (B, seq_len) — ground truth token IDs

        Returns:
            loss: Skaler loss değeri
        """
        B, seq_len_minus_1, vocab_size = logits.shape

        # Target: caption'ın <SOS>'tan sonraki kısmı
        # [<SOS>, w1, w2, ..., wn, <EOS>] → [w1, w2, ..., wn, <EOS>]
        targets = captions[:, 1:]  # (B, seq_len-1)

        # CrossEntropyLoss için reshape
        # logits:  (B * (seq-1), vocab_size)
        # targets: (B * (seq-1),)
        loss = self.criterion(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        return loss


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("ImageCaptioningModel (Encoder-Decoder) test ediliyor...")
    print("="*50)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}\n")

    VOCAB_SIZE = 5000
    BATCH_SIZE = 2
    SEQ_LEN    = 15

    # Model oluştur
    model = ImageCaptioningModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=512,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5,
        device=device,
    ).to(device)

    # Loss
    criterion = CaptioningLoss(pad_idx=0, label_smoothing=0.1)

    # Dummy data
    images   = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
    captions = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    captions[:, 0] = 1   # <SOS>
    captions[:, -1] = 2  # <EOS>

    # Forward pass
    logits = model(images, captions)
    print(f"Forward pass OK: logits shape = {logits.shape}")

    # Loss hesapla
    loss = criterion(logits, captions)
    print(f"Loss: {loss.item():.4f}")

    # Backward
    loss.backward()
    print("Backward pass OK ✅")

    # Save/Load test
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = ImageCaptioningModel.from_pretrained(tmpdir, device=device)
        print("Save/Load test ✅")

    print(f"\n✅ Adım 5 tamamlandı!")
    print("   Bir sonraki adım: train_captioning.py (Eğitim döngüsü)")
