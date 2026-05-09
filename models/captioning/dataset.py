"""
Null Pointers - COCO Dataset Preprocessing & DataLoader
------------------------------------------------------------
Bu modül MS-COCO veri setini PyTorch DataLoader formatına dönüştürür.

Akademik arka plan:
- Her COCO görüntüsünün 5 farklı insan yazısı caption'ı vardır.
- Eğitimde her (görüntü, caption) çifti ayrı bir örnek olarak kullanılır.
- CLIP encoder zaten pretrained olduğu için görüntüler normalize edilir
  (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
  Bu değerler CLIP'in eğitim veri setinden (400M görüntü) gelmektedir.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# ─── Vocabulary Builder ────────────────────────────────────────────────────────

class Vocabulary:
    """
    Caption metinlerini token ID'lerine dönüştüren kelime hazinesi.

    Özel tokenlar:
    - <PAD> (0): Sequence padding
    - <SOS> (1): Start of Sequence — decoder ilk adımda bunu alır
    - <EOS> (2): End of Sequence — üretim burada durur
    - <UNK> (3): Bilinmeyen kelime
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, freq_threshold: int = 5):
        """
        Args:
            freq_threshold: Kelime hazinesine girmek için minimum frekans.
                           Nadir kelimeleri <UNK> ile eşleştiririz.
                           5 iyi bir başlangıç değeridir (BLEU'yu iyileştirir).
        """
        self.freq_threshold = freq_threshold
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Dict[str, int] = {}

        # Özel tokenları yerleştir
        self._add_special_tokens()

    def _add_special_tokens(self):
        for idx, token in enumerate([self.PAD_TOKEN, self.SOS_TOKEN,
                                     self.EOS_TOKEN, self.UNK_TOKEN]):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_from_captions(self, captions: List[str]):
        """Caption listesinden kelime hazinesi oluşturur."""
        print("🔨 Vocabulary oluşturuluyor...")

        # Frekans sayımı
        for caption in captions:
            for word in self._tokenize(caption):
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Eşiği geçen kelimeleri ekle
        idx = len(self.word2idx)  # Özel tokenlardan sonra başla
        for word, freq in sorted(self.word_freq.items()):
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(f"   ✅ Vocabulary boyutu: {len(self.word2idx)} kelime")
        print(f"   (freq_threshold={self.freq_threshold}, toplam unique: {len(self.word_freq)})")

    def _tokenize(self, text: str) -> List[str]:
        """Basit whitespace tokenizer — NLTK gerektirir."""
        return text.lower().strip().split()

    def encode(self, caption: str) -> List[int]:
        """Caption → token ID listesi. <SOS> ve <EOS> eklenir."""
        tokens = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN])
                  for w in self._tokenize(caption)]
        return [self.word2idx[self.SOS_TOKEN]] + tokens + [self.word2idx[self.EOS_TOKEN]]

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Token ID listesi → caption metni."""
        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if skip_special and word in (self.PAD_TOKEN, self.SOS_TOKEN,
                                          self.EOS_TOKEN, self.UNK_TOKEN):
                continue
            words.append(word)
        return ' '.join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        """Vocabulary'yi JSON olarak kaydet."""
        data = {
            "word2idx": self.word2idx,
            "freq_threshold": self.freq_threshold
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"✅ Vocabulary kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Kaydedilmiş vocabulary'yi yükle."""
        with open(path) as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in
                          {v: k for k, v in data["word2idx"].items()}.items()}
        return vocab


# ─── Collate Function ──────────────────────────────────────────────────────────

def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DataLoader batch'lerini düzenler.
    Caption'lar farklı uzunlukta olabilir → padding gerekli.

    Args:
        batch: [(image_tensor, caption_tensor), ...] listesi

    Returns:
        images: (batch_size, 3, 224, 224)
        captions: (batch_size, max_caption_len) — PAD=0 ile doldurulmuş
    """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # En uzun caption'a göre pad et
    max_len = max(cap.size(0) for cap in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :cap.size(0)] = cap

    return images, padded_captions


# ─── COCO Dataset ─────────────────────────────────────────────────────────────

class COCOCaptionDataset(Dataset):
    """
    MS-COCO Image Captioning Dataset.

    Her __getitem__ çağrısında:
    1. Görüntüyü diskten yükler
    2. CLIP normalize dönüşümü uygular
    3. Caption'ı token ID'lerine çevirir

    Args:
        data_dir: COCO root klasörü (data/coco_subset/)
        subset_json: coco_subset.json dosya yolu
        vocab: Vocabulary nesnesi
        transform: torchvision transform (CLIP preprocess)
        max_caption_len: Maksimum caption uzunluğu (keser)
    """

    def __init__(
        self,
        data_dir: str,
        subset_json: str,
        vocab: Vocabulary,
        transform=None,
        max_caption_len: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.transform = transform
        self.max_caption_len = max_caption_len

        with open(subset_json) as f:
            self.data = json.load(f)

        print(f"✅ COCOCaptionDataset yüklendi: {len(self.data)} örnek")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]

        # ── Image ──
        img_path = self.data_dir / item["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, Exception):
            # Bozuk/eksik görüntü: siyah görüntü döndür
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # ── Caption ──
        caption_ids = self.vocab.encode(item["caption"])
        # Max uzunluğu kır (EOS dahil)
        caption_ids = caption_ids[:self.max_caption_len]
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        return image, caption_tensor

    def get_raw_caption(self, idx: int) -> str:
        """BLEU hesaplaması için ham caption döndürür."""
        return self.data[idx]["caption"]


# ─── DataLoader Factory ────────────────────────────────────────────────────────

def get_coco_loaders(
    data_dir: str,
    subset_json: str,
    vocab: Vocabulary,
    clip_preprocess,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train / Val / Test DataLoader'larını oluşturur.

    Splits:
    - Train: %85 — model eğitimi
    - Val:   %10 — hyperparameter tuning, erken durdurma
    - Test:  %05 — final BLEU değerlendirmesi (dokunulmaz!)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_dataset = COCOCaptionDataset(
        data_dir=data_dir,
        subset_json=subset_json,
        vocab=vocab,
        transform=clip_preprocess,
    )

    n = len(full_dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    from torch.utils.data import random_split
    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),  # Reproducibility
    )

    # MPS için pin_memory=False (MPS pinned memory desteklemez)
    common_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # CUDA için True, MPS için False
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, **common_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, **common_kwargs)

    print(f"\n📊 DataLoader splits:")
    print(f"   Train: {len(train_set)} örnek ({len(train_loader)} batch)")
    print(f"   Val:   {len(val_set)} örnek ({len(val_loader)} batch)")
    print(f"   Test:  {len(test_set)} örnek ({len(test_loader)} batch)")

    return train_loader, val_loader, test_loader


# ─── WritingPrompts Dataset ───────────────────────────────────────────────────

class WritingPromptsDataset(Dataset):
    """
    WritingPrompts GPT-2 Fine-tune Dataset.

    Her örnek, GPT-2'nin beklediği formatta tokenize edilmiş bir metin dizisidir:
    "<|prompt|> {prompt} <|story|> {story} <|endoftext|>"

    GPT-2 causal LM olduğu için input = target (shifted by 1).
    Transformers'ın DataCollatorForLanguageModeling bunu otomatik halleder.
    """

    def __init__(
        self,
        subset_json: str,
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            subset_json: writing_prompts_subset.json yolu
            tokenizer: GPT-2 tokenizer (HuggingFace)
            max_length: Maksimum token uzunluğu (GPT-2 max 1024)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(subset_json, encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"✅ WritingPromptsDataset yüklendi: {len(self.data)} örnek")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]["text"]  # Önceden hazırlanmış format

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # Causal LM: labels = input shifted (handled by model)
        }


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Preprocessing modülünü test et."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("="*50)
    print("Preprocessing modülü test ediliyor...")
    print("="*50)

    # Dummy captions ile vocabulary test
    test_captions = [
        "a dog sitting on a couch",
        "a cat playing with a ball",
        "two people walking in a park",
        "a red car driving on the street",
    ] * 100  # Frekans eşiğini geçmek için

    vocab = Vocabulary(freq_threshold=2)
    vocab.build_from_captions(test_captions)
    print(f"\nVocabulary size: {len(vocab)}")

    # Encode/decode test
    sample = "a dog sitting on a couch"
    encoded = vocab.encode(sample)
    decoded = vocab.decode(encoded)
    print(f"Original: '{sample}'")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  '{decoded}'")
    assert sample == decoded, "Encode/decode mismatch!"
    print("✅ Vocabulary encode/decode OK")

    print("\n✅ Preprocessing modülü hazır!")
