"""Run OCR sampling evaluation for a vision–language model checkpoint.

For every example in the evaluation split we:
1. Build the same chat-style prompt used during fine‑tuning
   (system + user with image).
2. Sample / decode a completion with `model.generate()`.
3. Compute Character‑Error‑Rate (CER) and Word‑Error‑Rate (WER)
   against the ground‑truth transcription.

Example
-------
>>> python sampling_eval.py \
        --checkpoint sft_output/checkpoint-1200 \
        --eval_dataset_id rs545837/sanskrit-ocr-images \
        --batch_size 4

Results will be printed to stdout and optionally written to a JSON
file via ``--save_preds`` for later inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, NamedTuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from datasets import load_dataset, Dataset
from PIL import Image
import nltk
from nltk.metrics.distance import edit_distance

from load_data import collate_fn as train_collate_fn

# Download nltk resources if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def calculate_cer(gt, pred):
    """Calculate Character Error Rate"""
    if not gt:
        return 1.0 if pred else 0.0
    return edit_distance(gt, pred) / len(gt)


def calculate_wer(gt, pred):
    """Calculate Word Error Rate"""
    # Split texts into words
    gt_words = gt.split()
    pred_words = pred.split()

    if not gt_words:
        return 1.0 if pred_words else 0.0

    # Use dynamic programming to calculate Levenshtein distance for words
    d = [[0 for _ in range(len(pred_words) + 1)] for _ in range(len(gt_words) + 1)]

    # Initialize first row and column
    for i in range(len(gt_words) + 1):
        d[i][0] = i
    for j in range(len(pred_words) + 1):
        d[0][j] = j

    # Fill the matrix
    for i in range(1, len(gt_words) + 1):
        for j in range(1, len(pred_words) + 1):
            if gt_words[i - 1] == pred_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                # Substitution
                substitution = d[i - 1][j - 1] + 1
                # Insertion
                insertion = d[i][j - 1] + 1
                # Deletion
                deletion = d[i - 1][j] + 1

                d[i][j] = min(substitution, insertion, deletion)

    # The last value in the matrix is the Levenshtein distance
    distance = d[len(gt_words)][len(pred_words)]

    # Calculate WER
    return distance / len(gt_words)


class EvalConfig(NamedTuple):
    """Configuration for running evaluation."""

    checkpoint: str = ""
    eval_dataset_id: str = "rs545837/sanskrit-ocr-images"
    split: str = "train"
    batch_size: int = 4
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    save_preds: Optional[str] = None
    use_4bit: bool = False


# def build_prompt(image) -> Dict:
#     """Construct the chat template messages for a single (image) example."""
#     return [
#         {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": USER_PROMPT},
#                 {"type": "image", "image": image},
#             ],
#         },
#     ]


# def prepare_batch(batch: Dict, processor) -> Dict[str, torch.Tensor]:
#     """Tokenise & tensorise a batch of HF‑Dataset rows.

#     The processor expects *lists* of images for each example because the model
#     supports multi‑image chat messages. Hence we wrap each single image in a
#     list.
#     """
#     msgs: List[list] = [build_prompt(img) for img in batch["image"]]

#     text_inputs: List[str] = [
#         processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
#         for m in msgs
#     ]
#     image_inputs: List[List] = [process_vision_info(m) for m in msgs]

#     inputs = processor(
#         text=text_inputs,
#         images=image_inputs,
#         return_tensors="pt",
#         padding=True,
#     )
#     return inputs


def collate_fn(examples: list[dict], processor):
    batch = train_collate_fn(examples, processor, eval=True)
    del batch["labels"]
    batch["text"] = [ex["text"] for ex in examples]
    return batch


def run_sampling_eval(model=None, processor=None, dataset=None, **kwargs) -> Dict:
    """Run OCR evaluation with the given configuration.

    Returns
    -------
    Dict containing evaluation metrics and optionally saved predictions.
    """
    config = EvalConfig(**kwargs)

    # ---------------------------------------------------------------------
    # Model & processor
    if model is None:
        if config.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        else:
            bnb_cfg = None

        model = AutoModelForImageTextToText.from_pretrained(
            config.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_cfg,
            attn_implementation="eager",
        )
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            config.checkpoint, trust_remote_code=True
        )

    device = next(model.parameters()).device
    model.eval()
    processor.tokenizer.padding_side = (
        "left"  # TODO: do we need this for sampling eval?
    )
    processor.tokenizer.truncation_side = "left"

    # ---------------------------------------------------------------------
    # Dataset & loader
    if dataset is None:
        dataset: Dataset = load_dataset(config.eval_dataset_id, split=config.split)
        dataset = dataset.filter(lambda x: x["text"] and len(x["text"].strip()))

    dl = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, processor),
    )

    # ---------------------------------------------------------------------
    # Evaluation loop
    cer_scores: List[float] = []
    wer_scores: List[float] = []

    if config.save_preds:
        all_records = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="evaluating", unit="batch"):
            label_texts: List[str] = [t.strip() for t in batch["text"]]
            del batch["text"]

            # Move tensors to the same device as the model
            inputs = {k: v.to(device) for k, v in batch.items()}

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.temperature > 0.0,
                temperature=config.temperature if config.temperature > 0.0 else None,
                top_k=config.top_k,
                top_p=config.top_p,
            )

            # Extract only the newly generated tokens (skip prompt)
            # Find the position of the final non-padding token in each sequence
            # (assume padding is on the left)
            prompt_lens = (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(
                dim=1
            )
            gen_trimmed = [
                seq[prompt_len:] for seq, prompt_len in zip(gen_ids, prompt_lens)
            ]
            pred_texts: List[str] = processor.batch_decode(
                gen_trimmed, skip_special_tokens=True
            )

            # Metrics
            cer_scores.extend(
                calculate_cer(gt, pred) for gt, pred in zip(label_texts, pred_texts)
            )
            wer_scores.extend(
                calculate_wer(gt, pred) for gt, pred in zip(label_texts, pred_texts)
            )

            if config.save_preds:
                for gt, pred in zip(label_texts, pred_texts):
                    all_records.append({"ground_truth": gt, "prediction": pred})

    # ---------------------------------------------------------------------
    mean_cer = sum(cer_scores) / len(cer_scores)
    mean_wer = sum(wer_scores) / len(wer_scores)

    results = {
        "checkpoint": config.checkpoint,
        "eval_split": f"{config.eval_dataset_id}/{config.split}",
        "n_samples": len(dataset),
        "batch_size": config.batch_size,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "cer": mean_cer,
        "wer": mean_wer,
    }

    if config.save_preds:
        out_path = Path(config.save_preds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {**results, "records": all_records},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Predictions written to {out_path.resolve()}")

    return results


if __name__ == "__main__":
    """CLI entry point for running evaluation."""
    parser = argparse.ArgumentParser(description="OCR sampling evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a fine‑tuned checkpoint or hub ID",
    )
    parser.add_argument(
        "--eval_dataset_id",
        type=str,
        default="rs545837/sanskrit-ocr-images",
        help="HF dataset repo ID containing eval split with 'image' & 'text' columns",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to evaluate (default: train)"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0 = greedy)",
    )
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--save_preds",
        type=str,
        help="Path to JSON file where <pred, label> pairs will be saved",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load model with 4‑bit quantisation \n  (requires bitsandbytes & same config used in training)",
    )
    args = parser.parse_args()

    results = run_sampling_eval(**vars(args))

    print("\n=== Sampling Evaluation Results ===")
    print(f"Checkpoint    : {results['checkpoint']}")
    print(f"Eval split    : {results['eval_split']}  (n={results['n_samples']})")
    print(f"Batch size    : {results['batch_size']}")
    print(f"Max new tokens: {results['max_new_tokens']}")
    print(f"Temperature   : {results['temperature']}")
    print(f"CER          : {results['cer']:.4f}")
    print(f"WER          : {results['wer']:.4f}\n")
