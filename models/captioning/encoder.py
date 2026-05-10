"""
Null Pointers - CLIP Encoder Wrapper (Adım 3)
--------------------------------------------------
CLIP ViT-B/32 modelini image feature extractor olarak kullanır.

Akademik açıklama:
- CLIP (Contrastive Language-Image Pre-Training) OpenAI tarafından 2021'de yayınlandı.
- 400 milyon (image, text) çiftiyle eğitildi → güçlü görsel temsiller üretir.
- ViT-B/32: Vision Transformer, patch size 32, Base variant (86M parametre)
- Çıktı: 512 boyutlu embedding vektörü (her görüntü için)
- FROZEN tutulur: Gradyanlar hesaplanmaz → training çok daha hızlı ve stabil.

Neden CLIP?
- Alternatif: ResNet, EfficientNet, ViT (scratch'ten)
- CLIP avantajı: Dil-görüntü hizalaması önceden yapılmış → caption üretimine
  daha uygun özellikler çıkarıyor. COCO ile fine-tune etmeye gerek yok.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CLIPEncoder(nn.Module):
    """
    CLIP ViT-B/32 tabanlı görüntü encoder.

    Pipeline:
    1. CLIP modeli yüklenir ve DONDURULUR (requires_grad=False)
    2. Görüntü → CLIP.encode_image() → (B, 512) float32 tensor
    3. Opsiyonel: Linear projection → LSTM embedding boyutuna map'le

    Args:
        embed_dim: LSTM decoder'ın beklediği embedding boyutu.
                   CLIP ViT-B/32 512 çıkarır. embed_dim != 512 ise
                   bir projection layer eklenir.
        fine_tune_clip: Eğer True ise CLIP ağırlıkları da güncellenir
                        (önerilmez — overfitting riski yüksek, RAM çok artar).
    """

    CLIP_EMBED_DIM = 512  # ViT-B/32'nin sabit çıktı boyutu

    def __init__(
        self,
        embed_dim: int = 512,
        fine_tune_clip: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

        # ── CLIP model yükleme ──────────────────────────────────────────────────
        try:
            import clip
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
            print(f"✅ CLIP ViT-B/32 yüklendi → device: {device}")
        except ImportError:
            raise ImportError(
                "CLIP yüklü değil. Yüklemek için:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )

        # ── Freeze / Unfreeze ───────────────────────────────────────────────────
        if not fine_tune_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("   CLIP ağırlıkları DONDURULDU (frozen) ✅")
        else:
            print("   ⚠️  CLIP ağırlıkları eğitiliyor (fine-tune mod)")

        # ── Projection Layer (gerekirse) ────────────────────────────────────────
        # CLIP 512-dim çıkarır. Farklı embed_dim istenirse linear projection ekle.
        if embed_dim != self.CLIP_EMBED_DIM:
            self.projection = nn.Sequential(
                nn.Linear(self.CLIP_EMBED_DIM, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            print(f"   Projection: {self.CLIP_EMBED_DIM} → {embed_dim}")
        else:
            self.projection = None
            print(f"   Projection: yok (embed_dim == CLIP dim = {embed_dim})")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224) — CLIP preprocess uygulanmış görüntüler

        Returns:
            features: (B, embed_dim) — görüntü feature vektörleri
        """
        # CLIP encode — FP16 olarak döner (CLIP default), FP32'ye çevir
        with torch.no_grad() if not self._is_fine_tuning() else torch.enable_grad():
            features = self.clip_model.encode_image(images)
            features = features.float()  # FP16 → FP32 (MPS uyumluluğu)

        # Opsiyonel projection
        if self.projection is not None:
            features = self.projection(features)

        return features

    def _is_fine_tuning(self) -> bool:
        """CLIP parametrelerinin eğitilip eğitilmediğini kontrol eder."""
        return next(self.clip_model.parameters()).requires_grad

    def get_preprocess(self):
        """CLIP preprocess transform fonksiyonunu döndürür (DataLoader için)."""
        return self.preprocess

    @property
    def output_dim(self) -> int:
        """Encoder'ın çıktı boyutunu döndürür."""
        return self.embed_dim

    def encode_single(self, image_path: str) -> torch.Tensor:
        """
        Tek bir görüntü dosyasından feature çıkarır (inference için).

        Args:
            image_path: Görüntü dosyası yolu

        Returns:
            features: (1, embed_dim)
        """
        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.forward(image_tensor)

    def __repr__(self) -> str:
        frozen = not self._is_fine_tuning()
        proj_info = f"Projection({self.CLIP_EMBED_DIM}→{self.embed_dim})" \
                    if self.projection else "No projection"
        return (f"CLIPEncoder(\n"
                f"  backbone=ViT-B/32, frozen={frozen},\n"
                f"  {proj_info},\n"
                f"  output_dim={self.embed_dim}\n"
                f")")


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("CLIPEncoder test ediliyor...")
    print("="*50)

    # Device seç
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Encoder oluştur
    encoder = CLIPEncoder(embed_dim=512, device=device)
    print(f"\n{encoder}\n")

    # Dummy input ile test (3 görüntü, 3x224x224)
    dummy_images = torch.randn(3, 3, 224, 224).to(device)
    features = encoder(dummy_images)

    print(f"Input shape:  {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    assert features.shape == (3, 512), f"Beklenen (3, 512), alınan {features.shape}"
    print("✅ CLIPEncoder forward pass OK!")

    # Parametre sayısı
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nToplam parametre: {total_params:,}")
    print(f"Eğitilecek parametre: {trainable_params:,} (CLIP frozen)")

    print("\n✅ Adım 3 tamamlandı!")
    print("   Bir sonraki adım: decoder.py (LSTM Decoder)")
