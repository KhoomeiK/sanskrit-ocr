from vllm import LLM, SamplingParams
import pandas as pd
from datetime import datetime
import os

MODEL_NAME = "google/gemma-3-27b-it"  # or base: "google/gemma-3-27b"
TP = 8  # 1-GPU-per-TP shard
PIPELINE_PARALLEL_SIZE = 1  # keep 1 to avoid pipeline bubbles
DTYPE = "bfloat16"  # Gemma3 weights ship in bf16
MAX_MODEL_LEN = 128_000  # full context window
MAX_NUM_SEQS = 256  # raise because H100s have room
ENABLE_CHUNKED_PREFILL = True  # big win on long prompts
GPU_MEM_UTIL = 0.90  # expose almost all VRAM to KV-cache

PROMPT = (
    "Write a random story in Sanskrit. Please return only Devanagari in your response."
)
NUM_SAMPLES = 10
MAX_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.95
OUT_PATH = "samples.parquet"


def main():
    print(f"[{datetime.now():%F %T}] loading model …")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TP,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        gpu_memory_utilization=GPU_MEM_UTIL,  # new
        trust_remote_code=True,  # Gemma3 uses custom classes
    )

    params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=1,
    )

    rows = []
    for i in range(NUM_SAMPLES):
        out = llm.generate([PROMPT], params)[0]
        rows.append({"id": i, "generation": out.outputs[0].text.strip()})
        print(f"✓ {i+1}/{NUM_SAMPLES}")

    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    print("saved →", os.path.abspath(OUT_PATH))


if __name__ == "__main__":
    main()
