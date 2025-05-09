"""Train and serve a BPE tokenizer that plugs directly into nn.Embedding.

Usage (one‑off):
    $ python tokenizer.py --corpus ./text_corpus.txt --vocab_size 32768

At train time import `BPETokenizer` and call .encode / .decode.
"""
from pathlib import Path
import argparse
import sentencepiece as spm
import torch
from torch import nn

class BPETokenizer(nn.Module):
    """Thin wrapper exposing encode/decode + nn.Embedding weight for FSDP."""
    def __init__(self, model_file: str):
        super().__init__()
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.vocab_size = self.sp.vocab_size()
        self.padding_idx = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()
        self.embed = nn.Embedding(self.vocab_size, 768)  # dim will be resized by model

    # --- text ↔ ids helpers --------------------------------------------------
    def encode(self, text: str, add_special=True):
        return self.sp.encode(text, out_type=int, add_bos=add_special, add_eos=add_special)

    def decode(self, ids):
        # strip padding / eos
        ids = [i for i in ids if i not in (self.padding_idx, self.eos_id)]
        return self.sp.decode(ids)

    # -------------------------------------------------------------------------
    def forward(self, ids):
        return self.embed(ids)


# -----------------------------------------------------------------------------
# CLI: train new SentencePiece model
# -----------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="Path to a raw‑text corpus (one sentence per line)")
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--output", default="tokenizer.model")
    args = p.parse_args()

    corpus = Path(args.corpus)
    assert corpus.exists(), corpus

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        vocab_size=args.vocab_size,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=[],
        model_prefix=args.output.replace(".model", ""),
    )
    print(f"🥳  Saved tokenizer to {args.output}")

if __name__ == "__main__":
    _cli()