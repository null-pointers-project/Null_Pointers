# Null Pointers 📸✍️

> **Fotoğraftan Hikayeye** — CLIP + LSTM ile görsel açıklama üretimi ve Llama 3.2 ile yaratıcı hikaye yazarlığı.

---

## Mimari

```
[Fotoğraf]
    │
    ▼
CLIP ViT-B/32 (Encoder, frozen)
    │  512-boyutlu görsel özellik vektörü
    ▼
LSTM Decoder (8.1M parametre, eğitildi)
    │
    ▼
[Caption]  →  "a herd of elephants standing next to each other"
    │
    ▼
Llama 3.2 — Ollama (lokal LLM)
    │  EN üretir, isteğe bağlı TR çevirisi (Google Translate)
    ▼
[Hikaye]
```

---

## Özellikler

- 🖼️ **Görsel Caption Üretimi** — CLIP + LSTM, MS-COCO üzerinde eğitildi (48K caption)
- 📖 **Hikaye Üretimi** — Llama 3.2 (lokal, Ollama aracılığıyla), 5 farklı tarz
- 🌍 **Türkçe Desteği** — Hikaye İngilizce üretilir, Google Translate ile çevrilir
- 🎨 **Web Arayüzü** — Karanlık temalı Flask uygulaması, sürükle-bırak görsel yükleme
- ⚡ **Gerçek Zamanlı Streaming** — Hikaye token token ekrana yazılır

---

## Kurulum

### Gereksinimler

- Python 3.11
- [Ollama](https://ollama.com) kurulu ve çalışıyor olmalı
- Apple Silicon (MPS) veya NVIDIA GPU (CUDA) — CPU ile de çalışır

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/null-pointers-project/Null_Pointers.git
cd Null_Pointers

# 2. Sanal ortam oluştur ve aktive et
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements-mac-training.txt
pip install git+https://github.com/openai/CLIP.git
pip install flask deep-translator

# 4. Llama 3.2 modelini indir (Ollama gerekli)
ollama pull llama3.2
```

---

## Kullanım

### Web Arayüzü (Önerilen)

```bash
source venv/bin/activate
python app/server.py
# → http://localhost:5001 adresini aç
```

Aynı ağdaki diğer cihazlardan da erişilebilir: `http://<Mac-IP>:5001`

### Komut Satırı — Sadece Caption

```bash
python scripts/test_caption.py resim.jpg --penalty 1.3
# v2 modeli ile:
python scripts/test_caption.py resim.jpg --checkpoint checkpoints/captioning/best_model_v2
```

### Komut Satırı — Caption + Hikaye

```bash
python scripts/generate_story.py resim.jpg
python scripts/generate_story.py resim.jpg --style dramatic
python scripts/generate_story.py resim.jpg --lang tr    # TR çevirisi
python scripts/generate_story.py resim.jpg --style mystery --lang tr
```

Hikaye tarzları: `default`, `dramatic`, `humorous`, `mystery`, `children`

---

## Model Eğitimi

### Caption Modeli — Sıfırdan Eğitim

```bash
# 1. Veri indir (MS-COCO val2017, ~1GB)
python scripts/download_data.py --dataset coco

# 2. Eğit
python training/train_captioning.py
```

### Caption Modeli — Fine-tuning (Daha Fazla Veri)

```bash
# 1. COCO train2017'den ek görsel indir
python scripts/download_coco_train.py --size 50000

# 2. Mevcut model üzerine fine-tune et
python training/train_captioning.py --finetune --data coco_train50k.json
```

---

## Model Sonuçları

| Model | Veri | Val Loss | PPL |
|-------|------|----------|-----|
| best_model (v1) | 25K caption (COCO val) | 3.51 | 33.4 |
| best_model_v2 | 48K caption (val+train) | **3.33** | **28.0** |

**Örnek Çıktılar:**

| Girdi | v1 | v2 |
|-------|----|----|
| 🚗 Araba | "a white and white photo of a" | "a white car parked in a parking lot" |
| 🐘 Filler | "a group of elephants" | "a **herd** of elephants" |

---

## Proje Yapısı

```
null-pointers/
├── app/
│   ├── server.py              # Flask backend (caption + story + translate)
│   └── templates/
│       └── index.html         # Karanlık temalı web arayüzü
├── models/
│   └── captioning/
│       ├── encoder.py         # CLIP ViT-B/32 wrapper
│       ├── decoder.py         # LSTM decoder (beam search, repetition penalty)
│       └── model.py           # Encoder-Decoder birleşik model
├── training/
│   └── train_captioning.py    # Eğitim döngüsü (--finetune desteği)
├── scripts/
│   ├── download_data.py       # MS-COCO veri indirme
│   ├── download_coco_train.py # COCO train subset indirme
│   ├── test_caption.py        # Caption test aracı
│   ├── generate_story.py      # Komut satırı pipeline
│   └── upload_to_hub.py       # HuggingFace Hub yükleme
├── checkpoints/
│   └── captioning/
│       ├── best_model/        # v1 model
│       └── best_model_v2/     # v2 model (fine-tuned)
├── configs/
│   └── config.yaml
└── data/
    └── coco_subset/           # İndirilen görseller + JSON subset
```

---

## Donanım

Proje **Apple M4 Mac Mini (16GB Unified Memory)** üzerinde geliştirildi.

| Bileşen | VRAM/RAM |
|---------|----------|
| CLIP + LSTM | ~632 MB |
| Llama 3.2 3B (Q4) | ~2.0 GB |
| **Toplam** | **~2.6 GB** |

Minimum: 3GB VRAM veya 8GB RAM (CPU modunda).

---

## HuggingFace

Caption modeli HuggingFace'te yayınlanmıştır:  
🤗 [nullpointersproject/null-pointers-caption](https://huggingface.co/nullpointersproject/null-pointers-caption)

---

## Takım

**Null Pointers** — Okul projesi kapsamında geliştirilmiştir.
