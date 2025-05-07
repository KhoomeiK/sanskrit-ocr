from vllm import LLM, SamplingParams
import pandas as pd
from datetime import datetime
import os

MODEL_NAME   = "cognitivecomputations/DeepSeek-V3-AWQ"
TP           = 8
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 8192  
MAX_NUM_SEQS = 128
ENABLE_CHUNKED_PREFILL = False
ENFORCE_EAGER = True
TRUST_REMOTE_CODE = True

PROMPT       = "Write a random story in Sanskrit. Please return only Devanagari in your response."
NUM_SAMPLES  = 10
MAX_TOKENS   = 256
TEMPERATURE  = 0.9
TOP_P        = 0.95
OUT_PATH     = "samples.parquet"


def main() -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] loading model …")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TP,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        enforce_eager=ENFORCE_EAGER,
        trust_remote_code=TRUST_REMOTE_CODE
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
        rows.append(
            dict(
                sample_id=i,
                prompt=PROMPT,
                generation=out.outputs[0].text.strip(),
            )
        )
        print(f"✓ {i+1}/{NUM_SAMPLES}")

    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    print(f"Saved → {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
