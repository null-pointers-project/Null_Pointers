# Null Pointers - Proje Bilgilendirmesi

## Klasör Yapısı
```
null-pointers/
├── data/
│   ├── coco_subset/          # MS-COCO 30K görüntü (Mac'e indirilecek)
│   └── writing_prompts/      # WritingPrompts 50K metin (Mac'e indirilecek)
├── models/
│   ├── captioning/           # CLIP Encoder + LSTM Decoder
│   └── story/                # GPT-2 fine-tune + inference
├── training/                 # Eğitim scriptleri (Mac'te çalışır)
├── evaluation/               # BLEU + ROUGE metrik hesaplamaları
├── app/                      # Gradio arayüzü (Windows'ta çalışır)
├── notebooks/                # Demo Jupyter notebook
├── scripts/                  # Yardımcı scriptler
├── configs/                  # Hyperparameter config dosyaları
├── requirements-mac-training.txt   # Mac M4 eğitim bağımlılıkları
├── requirements-windows-deploy.txt # Windows UI bağımlılıkları
└── .env.example              # Konfigürasyon şablonu
```

## İş Akışı
1. **Mac M4**: Eğitim → Model HuggingFace'e upload
2. **Windows**: Gradio UI çalıştır → HF API ile inference

## Kurulum Talimatları
→ Bkz. README.md'deki kurulum adımları
