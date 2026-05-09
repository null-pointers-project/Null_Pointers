"""
Null Pointers - LSTM Decoder (Adım 4)
------------------------------------------
CLIP encoder'dan gelen görüntü feature'larını alıp
kelime kelime caption üreten LSTM tabanlı decoder.

Akademik açıklama:
- Show and Tell (Vinyals et al., 2015) mimarisini temel alır.
- Encoder çıktısı (image feature) LSTM'in başlangıç hidden state'ini başlatır.
- Her adımda: [previous_word_embedding + context] → LSTM → next_word_distribution
- Beam search (inference) vs Greedy (hızlı test) desteklenir.

Neden LSTM?
- Alternatif: Transformer decoder (daha güçlü ama daha fazla veri gerektirir)
- LSTM avantajı: 30K gibi orta ölçekli veri setleri için daha iyi genelleme,
  daha az parametre, M4 Mac'te daha hızlı eğitim.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, List


class LSTMDecoder(nn.Module):
    """
    LSTM tabanlı caption decoder.

    Mimarisi:
    ┌─────────────────────────────────────────────────────────┐
    │  CLIP Feature (B, embed_dim)                            │
    │        ↓  init_hidden_state()                           │
    │  LSTM Hidden State (h0, c0)                             │
    │        ↓                                                │
    │  [SOS embedding] → LSTM Cell → softmax → word_1        │
    │  [word_1 embed]  → LSTM Cell → softmax → word_2        │
    │  ...                                                    │
    │  [word_n embed]  → LSTM Cell → softmax → [EOS]         │
    └─────────────────────────────────────────────────────────┘

    Args:
        embed_dim:   Kelime embedding boyutu (CLIP çıktısıyla aynı olmalı)
        hidden_dim:  LSTM hidden state boyutu
        vocab_size:  Kelime hazinesi boyutu
        num_layers:  LSTM katman sayısı (2 önerilir)
        dropout:     Dropout oranı (overfitting önleme)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 10_000,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # ── Word Embedding ──────────────────────────────────────────────────────
        # Kelime indekslerini dense vektörlere dönüştürür.
        # PAD (idx=0) için embedding sıfır olsun (padding_idx=0)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ── LSTM ────────────────────────────────────────────────────────────────
        # batch_first=True: (batch, seq, feature) formatı kullanır
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output Projection ───────────────────────────────────────────────────
        # LSTM hidden → vocabulary logits
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

        # ── Init Hidden State ───────────────────────────────────────────────────
        # Görüntü featureını LSTM başlangıç durumuna dönüştürür.
        # Her LSTM katmanı için ayrı h ve c üretilir.
        self.init_h = nn.Linear(embed_dim, num_layers * hidden_dim)
        self.init_c = nn.Linear(embed_dim, num_layers * hidden_dim)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform init — LSTM ağırlıkları için iyi başlangıç noktası.
        Embedding ve output linear de initialize edilir.
        """
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def init_hidden_state(
        self,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CLIP feature vektöründen LSTM başlangıç (h0, c0) üretir.

        Args:
            image_features: (B, embed_dim) — CLIP encoder çıktısı

        Returns:
            h0: (num_layers, B, hidden_dim)
            c0: (num_layers, B, hidden_dim)
        """
        B = image_features.size(0)

        h0 = torch.tanh(self.init_h(image_features))
        c0 = torch.tanh(self.init_c(image_features))

        # (B, num_layers * hidden_dim) → (num_layers, B, hidden_dim)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        return h0, c0

    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        TRAINING forward pass (teacher forcing).

        Teacher forcing: Her adımda ground truth tokeni input olarak kullanır.
        Bu yaklaşım eğitimi hızlandırır ama inference'ta scheduled sampling
        gerektirebilir. Akademik projelerde teacher forcing standarttır.

        Args:
            image_features: (B, embed_dim) — CLIP encoder çıktısı
            captions: (B, max_len) — token ID'leri [<SOS>, w1, w2, ..., <EOS>]

        Returns:
            logits: (B, max_len-1, vocab_size) — her adım için kelime skorları
                    NOT: <EOS> token'ı input olarak verilmez (son kelimeden sonra dur)
        """
        B = image_features.size(0)

        # LSTM başlangıç durumunu görüntüden hesapla
        h, c = self.init_hidden_state(image_features)

        # Caption embedding (son token hariç: <SOS>...last_word)
        # input:  [<SOS>, w1, w2, ..., w_{n-1}]
        # target: [w1,    w2, w3, ..., w_n, <EOS>]
        captions_input = captions[:, :-1]               # Son token'ı çıkar
        embeddings = self.dropout(self.embedding(captions_input))  # (B, seq-1, embed_dim)

        # LSTM forward
        outputs, _ = self.lstm(embeddings, (h, c))      # (B, seq-1, hidden_dim)

        # Output projection → logits
        logits = self.output_linear(self.dropout(outputs))  # (B, seq-1, vocab_size)

        return logits

    def generate_greedy(
        self,
        image_features: torch.Tensor,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_len: int = 50,
    ) -> List[int]:
        """
        Greedy decoding — her adımda en yüksek olasılıklı kelimeyi seçer.
        Hızlı ama optimal değil (beam search kadar iyi değil).

        Args:
            image_features: (1, embed_dim) — tek görüntü için
            sos_idx: <SOS> token indeksi
            eos_idx: <EOS> token indeksi
            max_len: Maksimum üretim uzunluğu

        Returns:
            token_ids: Üretilen token ID listesi (<SOS> ve <EOS> hariç)
        """
        self.eval()
        with torch.no_grad():
            h, c = self.init_hidden_state(image_features)

            # İlk input: <SOS>
            input_token = torch.tensor([[sos_idx]], device=image_features.device)
            generated = []

            for _ in range(max_len):
                embedding = self.embedding(input_token)         # (1, 1, embed_dim)
                output, (h, c) = self.lstm(embedding, (h, c))  # (1, 1, hidden_dim)
                logit = self.output_linear(output.squeeze(1))  # (1, vocab_size)

                # Greedy: argmax
                pred_token = logit.argmax(dim=-1).item()

                if pred_token == eos_idx:
                    break

                generated.append(pred_token)
                input_token = torch.tensor([[pred_token]], device=image_features.device)

        return generated

    def generate_beam_search(
        self,
        image_features: torch.Tensor,
        beam_size: int = 5,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_len: int = 50,
    ) -> List[int]:
        """
        Beam search decoding — daha kaliteli caption üretir.

        Algoritma:
        - Her adımda top-K olasılıklı kelimeyi koru
        - Tüm sekansların log-prob toplamını izle
        - <EOS>'e ulaşan sekansı tamamlanmış kabul et
        - En yüksek skorlu tamamlanmış sekansı döndür

        Args:
            image_features: (1, embed_dim)
            beam_size: Paralel tutulacak hipotez sayısı (5 = iyi denge)

        Returns:
            best_sequence: En iyi token ID listesi
        """
        self.eval()
        with torch.no_grad():
            device = image_features.device
            h, c = self.init_hidden_state(image_features)

            # Her beam için state kopyala
            # h: (num_layers, 1, hidden_dim) → (num_layers, beam_size, hidden_dim)
            h = h.repeat(1, beam_size, 1)
            c = c.repeat(1, beam_size, 1)

            # Başlangıç: beam_size adet <SOS>
            k_prev_words = torch.full((beam_size, 1), sos_idx,
                                       dtype=torch.long, device=device)
            seqs = k_prev_words  # (beam_size, 1)
            top_k_scores = torch.zeros(beam_size, 1, device=device)

            complete_seqs = []
            complete_seqs_scores = []

            step = 1
            while True:
                embeddings = self.embedding(k_prev_words)       # (beam, 1, embed_dim)

                # LSTM: batch olarak beam_size kadar çalıştır
                outputs, (h, c) = self.lstm(embeddings, (h, c))
                logits = self.output_linear(outputs.squeeze(1)) # (beam, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1)       # (beam, vocab_size)

                # Skor güncelle
                scores = top_k_scores + log_probs               # (beam, vocab_size)
                if step == 1:
                    # İlk adımda tüm beamler aynı, sadece ilkini kullan
                    top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)

                # Hangi beam ve hangi kelime?
                prev_word_inds = top_k_words // self.vocab_size
                next_word_inds = top_k_words % self.vocab_size

                # Sekansları güncelle
                seqs = torch.cat([seqs[prev_word_inds],
                                  next_word_inds.unsqueeze(1)], dim=1)

                # <EOS> üretenleri bitir
                incomplete = [i for i, w in enumerate(next_word_inds)
                              if w.item() != eos_idx]
                complete   = [i for i, w in enumerate(next_word_inds)
                              if w.item() == eos_idx]

                if complete:
                    complete_seqs.extend(seqs[complete].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete].tolist())

                beam_size -= len(complete)
                if beam_size == 0 or step >= max_len:
                    break

                # Incomplete beamleri devam ettir
                seqs = seqs[incomplete]
                h = h[:, prev_word_inds[incomplete], :]
                c = c[:, prev_word_inds[incomplete], :]
                top_k_scores = top_k_scores[incomplete].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete].unsqueeze(1)
                step += 1

            # En iyi sekansı seç
            if not complete_seqs:
                # Hiç tamamlanamadıysa greedy çıktısını al
                return seqs[0].tolist()[1:]  # <SOS> hariç

            i = complete_seqs_scores.index(max(complete_seqs_scores))
            best_seq = complete_seqs[i]
            # <SOS> ve <EOS> çıkar
            return [t for t in best_seq if t not in (sos_idx, eos_idx, 0)]

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"LSTMDecoder(\n"
                f"  embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim},\n"
                f"  vocab_size={self.vocab_size}, num_layers={self.num_layers},\n"
                f"  params={total:,} (trainable={trainable:,})\n"
                f")")


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("LSTMDecoder test ediliyor...")
    print("="*50)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    VOCAB_SIZE = 10_000
    EMBED_DIM  = 512
    HIDDEN_DIM = 512
    BATCH_SIZE = 4
    SEQ_LEN    = 20

    # Model oluştur
    decoder = LSTMDecoder(
        embed_dim=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        dropout=0.5,
    ).to(device)

    # Düzeltme: embed_dim'i doğru verelim
    decoder = LSTMDecoder(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=VOCAB_SIZE,
        num_layers=2,
        dropout=0.5,
    ).to(device)
    print(decoder)

    # Training forward pass test
    image_features = torch.randn(BATCH_SIZE, EMBED_DIM).to(device)
    captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)

    logits = decoder(image_features, captions)
    print(f"\nForward pass:")
    print(f"  Image features: {image_features.shape}")
    print(f"  Captions input: {captions.shape}")
    print(f"  Logits output:  {logits.shape}")
    assert logits.shape == (BATCH_SIZE, SEQ_LEN - 1, VOCAB_SIZE)
    print("  ✅ Training forward pass OK!")

    # Greedy generation test
    single_feat = torch.randn(1, EMBED_DIM).to(device)
    generated = decoder.generate_greedy(single_feat, max_len=10)
    print(f"\nGreedy generation: {generated}")
    print("  ✅ Greedy decode OK!")

    print(f"\n✅ Adım 4 tamamlandı!")
    print("   Bir sonraki adım: model.py (Encoder-Decoder birleşik model)")
