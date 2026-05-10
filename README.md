# Null Pointers 📸✍️

> Fotoğraftan Hikayeye: CLIP + LSTM + GPT-2 ile Uçtan Uca Yaratıcı Yazarlık Sistemi

## Mimari Özet

```
[Fotoğraf] → CLIP ViT-B/32 → LSTM Decoder → [Caption]
                                                  ↓
                                            GPT-2 Fine-tune → [Hikaye / Şiir]
```

---

## 🖥️ Platform Kurulumu

### 1. Windows PC (UI & Deployment)

```powershell
# Proje klasörüne git
cd C:\nullpointers\null-pointers

# Sanal ortam oluştur
python -m venv venv

# Aktive et
.\venv\Scripts\Activate.ps1

# Bağımlılıkları yükle
pip install -r requirements-windows-deploy.txt

# CLIP'i ayrıca yükle (git gerekli)
pip install git+https://github.com/openai/CLIP.git

# .env dosyasını oluştur
copy .env.example .env
# Not: .env içindeki değerleri doldurmayı unutma!

# Ortamı doğrula
python scripts/check_environment.py --mode deploy
```

---

### 2. Mac M4 Mini (Model Eğitimi)

```bash
# Projeyi Mac'e kopyala (veya git clone)
cd /Users/YourName/null-pointers

# Sanal ortam oluştur
python3.11 -m venv venv

# Aktive et
source venv/bin/activate

# PyTorch MPS destekli yükle
# NOT: pip ile yükle, conda ile değil (MPS uyumluluğu için)
pip install --upgrade pip

# Bağımlılıkları yükle
pip install -r requirements-mac-training.txt

# CLIP'i yükle
pip install git+https://github.com/openai/CLIP.git

# pycocotools (Mac'te sorun çıkarabilir, bu şekilde yükle)
pip install pycocotools

# .env dosyasını oluştur ve doldur
cp .env.example .env

# Ortamı doğrula
python scripts/check_environment.py --mode train
```

#### 🔍 MPS Çalışıyor mu? (Mac'te test et)

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('MPS Available:', torch.backends.mps.is_available())
print('MPS Built:', torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.ones(3, device=device)
    print('MPS Test Tensor:', x)
    print('✅ MPS aktif, eğitim GPU hızında çalışacak!')
else:
    print('⚠️ MPS bulunamadı, CPU kullanılacak.')
"
```

---

## 🔄 İş Akışı

### Faz 1: Mac'te Eğitim
```bash
# 1. Veri indir (Adım 2)
python scripts/download_data.py

# 2. Caption modelini eğit (Adım 6)
python training/train_captioning.py

# 3. GPT-2 fine-tune (Adım 8)
python training/train_story.py

# 4. Modeli HuggingFace'e yükle (Adım 12)
python scripts/upload_to_hub.py
```

### Faz 2: Windows'ta Çalıştır
```powershell
# Gradio uygulamasını başlat (Adım 11)
python app/app.py
# → http://localhost:7860 adresini aç
```

---

## 📊 Değerlendirme Metrikleri

| Modül | Metrik | Hedef |
|-------|--------|-------|
| Image Captioning | BLEU-4 | ≥ 0.25 (MS-COCO baseline) |
| Story Generation | ROUGE-L | ≥ 0.20 |
| Story Generation | Perplexity | ≤ 30 |

---

## 🗂️ Klasör Yapısı

```
null-pointers/
├── data/                          # Ham ve işlenmiş veriler
│   ├── coco_subset/               # MS-COCO 30K görüntü + caption
│   └── writing_prompts/           # WritingPrompts 50K hikaye
├── models/                        # Model mimarileri
│   ├── captioning/
│   │   ├── encoder.py             # CLIP ViT-B/32 wrapper
│   │   ├── decoder.py             # LSTM decoder
│   │   └── model.py               # Encoder-Decoder birleşik
│   └── story/
│       ├── train.py               # GPT-2 fine-tune
│       └── generate.py            # Story inference
├── training/
│   ├── train_captioning.py        # Caption eğitim döngüsü
│   └── train_story.py             # Story eğitim döngüsü
├── evaluation/
│   ├── bleu_score.py              # BLEU-4 hesaplama
│   └── rouge_score.py             # ROUGE hesaplama
├── app/
│   └── app.py                     # Gradio arayüzü
├── scripts/
│   ├── check_environment.py       # Ortam doğrulama
│   ├── download_data.py           # Veri indirme (Adım 2)
│   └── upload_to_hub.py           # HF Hub upload (Adım 12)
├── notebooks/
│   └── demo.ipynb                 # Demo notebook
├── configs/
│   └── config.yaml                # Hyperparameter ayarları
├── requirements-mac-training.txt
├── requirements-windows-deploy.txt
└── .env.example
```

---

## 🛠️ Geliştirme Adımları

- [x] Adım 1: requirements.txt + ortam kurulumu
- [ ] Adım 2: Veri indirme ve preprocessing
- [ ] Adım 3: CLIP Encoder wrapper
- [ ] Adım 4: LSTM Decoder mimarisi
- [ ] Adım 5: Birleşik Encoder-Decoder modeli
- [ ] Adım 6: Captioning eğitim döngüsü
- [ ] Adım 7: BLEU değerlendirme
- [ ] Adım 8: GPT-2 fine-tune
- [ ] Adım 9: Story generation inference
- [ ] Adım 10: ROUGE değerlendirme
- [ ] Adım 11: Gradio arayüzü
- [ ] Adım 12: HuggingFace Spaces deploy
