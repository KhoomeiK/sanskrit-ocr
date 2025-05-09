"""Export captions from a 🤗 dataset to plain‑text corpus.

Usage: python make_corpus.py --out text.txt

You can swap to any HF dataset/column via CLI flags; each caption
is written on its own line (UTF‑8).
"""
import argparse, sys
from pathlib import Path
from datasets import load_dataset

def main(args):
    ds = load_dataset(args.hf_id, split=args.split, streaming=args.streaming)
    text_col = args.text_col or infer_text_column(ds.column_names)
    out = Path(args.out)
    num = 0
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            cap = ex[text_col].replace("\n", " ").strip()
            if cap:
                f.write(cap + "\n")
                num += 1
                if args.limit and num >= args.limit:
                    break
    print(f"Wrote {num} lines → {out}")

def infer_text_column(cols):
    priors = ["text", "caption", "caption_0", "description"]
    for p in priors:
        if p in cols:
            return p
    print(f"❌   Can't infer text column from {cols}. Use --text_col.")
    sys.exit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hf_id", default="sizhkhy/open-images-captions-micro")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="text.txt")
    p.add_argument("--text_col", default=None, help="Name of the caption column")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests")
    p.add_argument("--streaming", action="store_true", help="Use HF streaming mode")
    main(p.parse_args())