#!/usr/bin/env python
# translate_bookcorpus_to_sanskrit.py
"""
Synthetic-translation data generator
────────────────────────────────────
• Input  : BookCorpus English passages (streaming)
• Model  : google/gemma-3-27b-it (bfloat16, TP = #GPUs)
• Output : translations.parquet – one row per passage, streamed in row-groups
           (so RAM stays ≪ GPU VRAM)

Example (8 × H100):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python translate_bookcorpus_to_sanskrit.py \
    --batch_size 32 --chunk_size 2048
"""
import os, argparse, datetime as dt
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset, logging as ds_logging
from vllm import LLM, SamplingParams

# ────────────────────────────── CLI ────────────────────────────── #
p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-3-27b-it", help="HF model repo")
p.add_argument("--max_passages", type=int, default=None,
               help="Optional hard stop (useful for smoke tests)")
p.add_argument("--batch_size", type=int, default=32, help="#prompts per vLLM call")
p.add_argument("--chunk_size", type=int, default=2_048,
               help="Rows per Parquet row-group write")
p.add_argument("--out", default="translations.parquet", help="Parquet destination")
p.add_argument("--tp", type=int, default=int(os.environ.get("TP", 8)),
               help="tensor_parallel_size (defaults to #GPUs)")
args = p.parse_args()

# ─────────────────────── vLLM initialisation ───────────────────── #
print(f"[{dt.datetime.now():%F %T}] → Loading Gemma …")
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tp,
    pipeline_parallel_size=1,
    dtype="bfloat16",
    max_model_len=128_000,
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

prompt_tmpl = (
    "Translate the following English text to Sanskrit. "
    "Return only Devanagari Sanskrit. Do NOT return any English.\n\n"
    "English:\n{passage}\n\nSanskrit:"
)

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, n=1)

# ─────────────────────── dataset streaming ─────────────────────── #
ds_logging.set_verbosity_error()  # silence HF progress bars
book = load_dataset(
    "bookcorpus/bookcorpus",
    "plain_text",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

def batched(iterable, n):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

# ───────────────────────── parquet writer ───────────────────────── #
writer      = None             # will hold ParquetWriter after first flush
chunk_rows  = []               # small in-RAM buffer
seen        = 0                # global counter

def flush_chunk():
    """Write current chunk_rows to disk and clear the buffer."""
    global writer, chunk_rows
    if not chunk_rows:
        return
    table = pa.Table.from_pylist(chunk_rows)
    if writer is None:         # first time → create file & schema
        writer = pq.ParquetWriter(args.out, table.schema, compression="zstd")
    writer.write_table(table)
    chunk_rows.clear()

# ───────────────────────── main loop ───────────────────────────── #
for batch in batched(book, args.batch_size):
    prompts = [prompt_tmpl.format(passage=ex["text"]) for ex in batch]
    outs    = llm.generate(prompts, sampling)

    for ex, out in zip(batch, outs):
        chunk_rows.append(
            dict(
                bookcorpus_id = seen,
                english       = ex["text"],
                sanskrit      = out.outputs[0].text.strip(),
            )
        )
        seen += 1
        if len(chunk_rows) >= args.chunk_size:
            flush_chunk()

    if seen % 10_000 == 0:
        print(f"[{dt.datetime.now():%T}] translated {seen:,} passages …")
    if args.max_passages and seen >= args.max_passages:
        break

flush_chunk()          # final partial chunk
if writer is not None:
    writer.close()
print(f"✓ Saved {seen:,} rows → {os.path.abspath(args.out)}")
