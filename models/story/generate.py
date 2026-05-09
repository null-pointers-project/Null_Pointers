"""
Smart Story Teller - Story Generation Inference (Adım 9)
---------------------------------------------------------
Fine-tune edilmiş GPT-2 modeliyle hikaye ve şiir üretir.

Kullanım:
    from models.story.generate import StoryGenerator

    generator = StoryGenerator("checkpoints/story/best_model/")
    story = generator.generate(
        caption="a dog playing in the park",
        mode="story",
        max_length=300,
    )
"""

import sys
from pathlib import Path
from typing import Optional, Literal

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class StoryGenerator:
    """
    Fine-tune edilmiş GPT-2 ile hikaye/şiir üretici.

    Desteklenen modlar:
    - "story": Yaratıcı kısa hikaye
    - "poem":  Şiirsel form (farklı prompt prefix kullanır)
    - "continuation": Verilen caption'ı doğrudan devam ettirir

    Sampling parametreleri (akademik):
    - temperature: Düşük (0.7) = daha tutarlı, Yüksek (1.0) = daha yaratıcı
    - top_p (nucleus): Kümülatif olasılık eşiği — long-tail kelimeleri keser
    - top_k: Her adımda sadece en olası K kelimeyi değerlendir
    - repetition_penalty: Aynı kelimenin tekrarını cezalandır (>1.0)
    """

    MODE_PREFIXES = {
        "story": "<|prompt|> {caption} <|story|>",
        "poem":  "<|prompt|> Write a poem inspired by: {caption} <|story|>",
        "continuation": "{caption}",
    }

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_dir: save_pretrained() ile kaydedilmiş GPT-2 klasörü
            device: "mps" | "cuda" | "cpu" | None (otomatik seç)
        """
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model_dir = model_dir

        print(f"📖 StoryGenerator yükleniyor: {model_dir}")
        print(f"   Device: {self.device}")

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = GPT2LMHeadModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,  # MPS FP16 desteği sınırlı
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ GPT-2 StoryGenerator hazır!")

    def generate(
        self,
        caption: str,
        mode: Literal["story", "poem", "continuation"] = "story",
        max_new_tokens: int = 300,
        temperature: float = 0.85,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        num_return_sequences: int = 1,
        do_sample: bool = True,
    ) -> str:
        """
        Caption'dan yaratıcı metin üretir.

        Args:
            caption:           Görüntü caption'ı (CLIP+LSTM çıktısı)
            mode:              "story" | "poem" | "continuation"
            max_new_tokens:    Üretilecek maksimum yeni token sayısı
            temperature:       Yaratıcılık parametresi (0.7-1.0 arası iyi)
            top_p:             Nucleus sampling eşiği
            top_k:             Top-k token havuzu
            repetition_penalty: Tekrar cezası (1.0=yok, 1.2=orta)
            num_return_sequences: Kaç alternatif üret (1=tek çıktı)

        Returns:
            Üretilen hikaye/şiir metni (temizlenmiş)
        """
        # Prompt hazırla
        prompt_template = self.MODE_PREFIXES[mode]
        prompt = prompt_template.format(caption=caption.strip())

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # Prompt için max 256 token bırak
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode — sadece yeni üretilen tokenleri al
        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Temizle
        text = self._clean_output(text)
        return text

    def generate_batch(
        self,
        captions: list,
        mode: str = "story",
        **kwargs,
    ) -> list:
        """Birden fazla caption için toplu üretim."""
        return [self.generate(cap, mode=mode, **kwargs) for cap in captions]

    def _clean_output(self, text: str) -> str:
        """
        Üretilen metni temizler:
        - Özel tokenları kaldır
        - Başındaki/sonundaki boşlukları temizle
        - <|endoftext|> sonrasını kes
        """
        # Özel tokenları kaldır
        for token in ["<|prompt|>", "<|story|>", "<|endoftext|>"]:
            text = text.replace(token, "")

        # <|endoftext|> sonrasını kes
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>")[0]

        # Temizle
        text = text.strip()

        # Cümle ortasında kesilmişse tamamla (nokta ara)
        sentences = text.split('.')
        if len(sentences) > 1 and not text.endswith('.'):
            text = '.'.join(sentences[:-1]) + '.'

        return text

    def interactive_demo(self):
        """Terminal'de interaktif test modu."""
        print("\n" + "="*60)
        print("StoryGenerator İnteraktif Demo")
        print("="*60)
        print("Komutlar: 'quit' çıkış, 'mode:story/poem' mod değiştir")
        print()

        mode = "story"
        while True:
            try:
                user_input = input(f"[{mode}] Caption gir: ").strip()
                if user_input.lower() == 'quit':
                    break
                if user_input.startswith("mode:"):
                    mode = user_input.split(":")[1]
                    print(f"Mod değiştirildi: {mode}")
                    continue
                if not user_input:
                    continue

                print("\n⏳ Üretiliyor...")
                result = self.generate(user_input, mode=mode)
                print(f"\n{'─'*40}")
                print(f"📝 {mode.upper()}:")
                print(result)
                print(f"{'─'*40}\n")

            except KeyboardInterrupt:
                break

        print("\nDemo sonlandırıldı.")


# ─── Module-level inference function ─────────────────────────────────────────

_generator_cache: Optional[StoryGenerator] = None

def get_generator(model_dir: str, device: str = None) -> StoryGenerator:
    """Singleton pattern — modeli bir kez yükle, defalarca kullan."""
    global _generator_cache
    if _generator_cache is None:
        _generator_cache = StoryGenerator(model_dir, device=device)
    return _generator_cache


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT_DIR / "checkpoints" / "story" / "best_model"))
    parser.add_argument("--caption", default="a golden retriever playing on the beach at sunset")
    parser.add_argument("--mode", default="story", choices=["story", "poem", "continuation"])
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if not Path(args.model).exists():
        # Demo mode: HuggingFace'den base GPT-2 kullan (test için)
        print(f"⚠️  Model bulunamadı: {args.model}")
        print("   Base GPT-2 kullanılıyor (fine-tune edilmemiş, test amaçlı)...")
        args.model = "gpt2"

    gen = StoryGenerator(args.model)

    if args.interactive:
        gen.interactive_demo()
    else:
        print(f"\nCaption: '{args.caption}'")
        print(f"Mode: {args.mode}\n")
        result = gen.generate(args.caption, mode=args.mode)
        print("─"*60)
        print(f"📝 Üretilen {args.mode.upper()}:")
        print(result)
        print("─"*60)
        print("\n✅ Adım 9 tamamlandı!")
        print("   Bir sonraki adım: rouge_score.py (ROUGE değerlendirme)")
