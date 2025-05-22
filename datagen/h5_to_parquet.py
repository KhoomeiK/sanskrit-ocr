import os
import re
import tables
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFolder

# ─── CONFIG ────────────────────────────────────────────────────────────────────
H5_PATH        = "/home/ubuntu/output/translations.h5"
TABLE_PATH     = "/data/sanskrit"      # path to your Table
ID_PATH        = "/data/id"            # path to your ID array
PARQUET_PATH   = "/home/ubuntu/output/translations.parquet"
HF_REPO_ID     = "khoomeik/samhitika-0.0.1"  # change to your namespace/repo
CHUNK_SIZE     = 100_000               # tune for your memory/I/O
# ────────────────────────────────────────────────────────────────────────────────

# Compile once: keep only Devanagari U+0900–U+097F and spaces
_non_dev_or_space = re.compile(r"[^\u0900-\u097F ]+")

def filter_devanagari(text: str) -> str:
    return _non_dev_or_space.sub("", text).strip()

def stream_h5_to_parquet(h5_path, table_path, id_path, parquet_path, chunk_size):
    writer = None
    buffer = []
    seen = set()
    total = 0

    with tables.open_file(h5_path, mode="r") as h5f:
        table = h5f.get_node(table_path)
        ids   = h5f.get_node(id_path)

        for idx, row in enumerate(table.iterrows()):
            # get the parallel ID
            bookcorpus_id = ids[idx]
            # pull & clean text
            raw = row['text'] if isinstance(row, dict) or hasattr(row, 'dtype') else row
            orig = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
            clean = filter_devanagari(orig)

            # filter out empty / whitespace-only
            if not clean:
                continue

            # skip duplicates
            if clean in seen:
                continue
            seen.add(clean)

            buffer.append({
                "bookcorpus_id": bookcorpus_id,
                # "original":      orig,
                "text":    clean
            })
            total += 1

            # flush chunk
            if total % chunk_size == 0:
                print(buffer[-1])

                import pandas as pd
                df_chunk = pd.DataFrame(buffer)
                table_batch = pa.Table.from_pandas(df_chunk)

                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table_batch.schema)
                writer.write_table(table_batch)
                buffer.clear()
                print(f"Flushed {total} unique rows…")

        # final flush
        if buffer:
            import pandas as pd
            df_chunk = pd.DataFrame(buffer)
            table_batch = pa.Table.from_pandas(df_chunk)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table_batch.schema)
            writer.write_table(table_batch)
            print(f"Flushed final {len(buffer)} rows (total {total})")

    if writer:
        writer.close()
    print(f"Done: wrote {total} deduplicated rows to {parquet_path}")

if __name__ == "__main__":
    # 1) Stream H5 → Parquet
    stream_h5_to_parquet(H5_PATH, TABLE_PATH, ID_PATH, PARQUET_PATH, CHUNK_SIZE)

    # # 2) Upload to Hugging Face Hub
    # token = HfFolder.get_token()
    # api = HfApi()
    # try:
    #     api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=False, token=token)
    #     print(f"Created dataset repo: {HF_REPO_ID}")
    # except Exception as e:
    #     print(f"Repo exists or creation failed: {e}")

    # api.upload_file(
    #     path_or_fileobj=PARQUET_PATH,
    #     path_in_repo=os.path.basename(PARQUET_PATH),
    #     repo_id=HF_REPO_ID,
    #     repo_type="dataset",
    #     token=token
    # )
    # print("Upload complete.")

    # # 3) Token counting
    # from transformers import AutoTokenizer
    # from datasets import load_dataset
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    # total_tokens = 0
    # batch_size = 10_000

    # print("Calculating total number of tokens in parquet's 'text' column...")
    # dataset = load_dataset("parquet", data_files=PARQUET_PATH, split="train")
    # for batch in dataset.iter(batch_size=batch_size):
    #     lengths = tokenizer(batch["text"], add_special_tokens=False, return_length=True)["length"]
    #     total_tokens += sum(lengths)
    #     print(f"Total tokens: {total_tokens}")

    # print("Calculating total number of tokens in indic_merged.txt...")
    # buffer = []
    # with open("/home/ubuntu/indic_merged.txt", "r", encoding="utf-8", errors="ignore") as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         buffer.append(line)
    #         if len(buffer) >= batch_size:
    #             # tokenize the batch
    #             out = tokenizer(buffer, 
    #                             add_special_tokens=False, 
    #                             return_length=True)
    #             total_tokens += sum(out["length"])
    #             print(f"Total tokens: {total_tokens}")
    #             buffer.clear()
    #     if buffer: # catch any remainder
    #         out = tokenizer(buffer, 
    #                         add_special_tokens=False, 
    #                         return_length=True)
    #         total_tokens += sum(out["length"])
    # print(f"Total tokens in file: {total_tokens}")
