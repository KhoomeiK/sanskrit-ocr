#!/usr/bin/env python
# translate_bookcorpus_to_sanskrit.py
"""
Synthetic-translation data generator (resumable, variable-length HDF5)
─────────────────────────────────────────────────────────────────────
• Input  : BookCorpus English passages (streaming)
• Model  : google/gemma-3-27b-it (bfloat16, TP = #GPUs)
• Output : translations.h5 – three variable-length columns
           ↳ Resumes automatically if the file already contains rows
"""
import os, argparse, datetime as dt
import numpy as np
import tables as tb  # low-level PyTables API
from datasets import load_dataset, DownloadConfig, logging as ds_logging
from vllm import LLM, SamplingParams

# ────────────────────────────── CLI ────────────────────────────── #
p = argparse.ArgumentParser()
p.add_argument("--model", default="google/gemma-3-27b-it", help="HF model repo")
p.add_argument(
    "--max_passages", type=int, default=None, help="Optional hard stop (smoke tests)"
)
p.add_argument("--batch_size", type=int, default=1024, help="#prompts per vLLM call")
p.add_argument("--chunk_size", type=int, default=2_048, help="Rows per HDF5 append")
p.add_argument(
    "--out", default="/home/ubuntu/output/translations.h5", help="HDF5 destination"
)
p.add_argument(
    "--tp",
    type=int,
    default=int(os.environ.get("TP", 8)),
    help="tensor_parallel_size (defaults to #GPUs)",
)
args = p.parse_args()

# ─────────── Open / create HDF5 file & arrays ─────────── #
h5 = tb.open_file(args.out, mode="a")
if "/data" in h5:
    group = h5.root.data
    id_arr = group.id
    eng_arr = group.english
    san_arr = group.sanskrit
    existing_rows = id_arr.nrows
    print(f"[resume] found {existing_rows:,} rows – skipping those.")
else:
    group = h5.create_group("/", "data")
    id_arr = h5.create_earray(group, "id", atom=tb.Int64Atom(), shape=(0,))
    eng_arr = h5.create_vlarray(group, "english", atom=tb.VLUnicodeAtom())
    san_arr = h5.create_vlarray(group, "sanskrit", atom=tb.VLUnicodeAtom())
    existing_rows = 0

# ─────────────────────── dataset streaming ─────────────────────── #
ds_logging.set_verbosity_error()

# cfg = DownloadConfig(
#     resume_download=True,   # pick up where it left off
#     max_retries=20,         # be more patient
#     # timeout=100             # seconds per chunk
# )
# # one-time local download
# load_dataset(
#     "bookcorpus/bookcorpus",
#     "plain_text",
#     split="train",
#     streaming=False,
#     trust_remote_code=True,
#     download_config=cfg,
# )

offline_cfg = DownloadConfig(local_files_only=True)
dataset_iter = iter(
    load_dataset(
        "bookcorpus/bookcorpus",
        "plain_text",
        split="train",
        streaming=False,  # do not stream to force use local cache
        download_config=offline_cfg,
        trust_remote_code=True,
    ).skip(existing_rows)
)


def batched(it, n):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


# ───────────────────── buffers & helpers ───────────────────── #
ids_buf, eng_buf, san_buf = [], [], []


def flush_chunk():
    """Append buffers to VLArrays and clear them."""
    if not ids_buf:  # nothing to write
        return

    # 1.  numeric IDs — EArray happily accepts a list
    id_arr.append(ids_buf)

    # 2.  variable-length strings — one call per element
    for s in eng_buf:
        eng_arr.append(s)
    for s in san_buf:
        san_arr.append(s)

    h5.flush()
    ids_buf.clear()
    eng_buf.clear()
    san_buf.clear()

# ─────────────────────── vLLM initialisation ───────────────────── #
print(f"[{dt.datetime.now():%F %T}] → Loading Gemma …")
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tp,
    pipeline_parallel_size=1,
    dtype="bfloat16",
    max_model_len=4096,
    enable_chunked_prefill=True,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

prompt_tmpl = (
    "Translate the following English text to Sanskrit. Return only one Devanagari Sanskrit translation wrapped in triple backticks. Do NOT return any English.\n\n"
    "English:\n```\n{passage}\n```\n\nSanskrit:"
)

sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, n=1, truncate_prompt_tokens=2048)

# ───────────────────────── main loop ───────────────────────────── #
seen = existing_rows
for i, batch in enumerate(batched(dataset_iter, args.batch_size)):
    prompts = [prompt_tmpl.format(passage=ex["text"]) for ex in batch]
    try:
        outs = llm.generate(prompts, sampling)
    except ValueError as e:
        if "4096" in str(e):
            print(f"Error: {str(e)}")
            continue

    for ex, out in zip(batch, outs):
        ids_buf.append(seen)
        eng_buf.append(ex["text"])
        san_buf.append(out.outputs[0].text.strip())
        seen += 1

        if len(ids_buf) >= args.chunk_size:
            flush_chunk()

    print(f"{i}: [{dt.datetime.now():%T}] translated {seen:,} passages …")
    if args.max_passages and seen >= args.max_passages:
        break

flush_chunk()  # final partial chunk
h5.close()
print(f"✓ Total rows in file: {seen:,} → {os.path.abspath(args.out)}")
