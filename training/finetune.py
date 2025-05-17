import os
import time

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import wandb

from load_data import load_multimodal_dataset, collate_fn
from sampling_eval import run_sampling_eval

torch.autograd.set_detect_anomaly(True)

# -------- CONFIGURATION --------
MODEL_ID = "google/gemma-3-4b-it"  # vision-capable Gemma model
OUTPUT_DIR = "/home/ubuntu/sft_output"
EVAL_DATASET_ID = "rs545837/sanskrit-ocr-images"
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACC_STEPS = 4
EVAL_BATCH_SIZE = 8
EVAL_EVERY_STEPS = 32
LR = 2e-4

# QLoRA / BitsAndBytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

# PEFT LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

# SFT Training config
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    gradient_checkpointing=False,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    optim="adamw_torch_fused",
    learning_rate=LR,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=EVAL_EVERY_STEPS,
    eval_on_start=True,  # NOTE: set False later
    push_to_hub=False,
    report_to="wandb",
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)

# -------- DATA LOADING --------

train_ds = load_multimodal_dataset("/home/ubuntu/ocr_dataset/captions.jsonl")
eval_ds = load_dataset(EVAL_DATASET_ID, split="train")
eval_ds = eval_ds.filter(lambda x: x["text"] and len(x["text"].strip()))

# -------- PROCESSOR & MODEL --------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map=(
        None if int(os.getenv("LOCAL_RANK", "-1")) != -1 else "auto"
    ),  # turn off "auto" when using accelerate
)
processor.tokenizer.padding_side = "right"
processor.tokenizer.truncation_side = "right"

# TODO: try this
# torch._dynamo.config.cache_size_limit = 64_000
# model = torch.compile(model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=f"test-{time.strftime('%H%M%S')}",
        help="Name for this run (used in wandb)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debugpy debugging")
    args = parser.parse_args()

    run_name = f'{time.strftime("%m%d")}-{args.run_name}'

    rank = int(os.getenv("RANK", "-1"))
    if args.debug and rank == 0:
        import debugpy

        port = rank + 5678
        print(f"Initializing debugpy on {port}")
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()

    if rank == 0:
        wandb.init(
            name=run_name,
            project="sanskrit-ocr",
            # tags=[],
        )
    sampling_eval_init_results = run_sampling_eval(
        model=model,
        processor=processor,
        dataset=eval_ds,
        save_preds=f"/home/ubuntu/eval_output/{run_name}-eval-init.json",
    )
    if rank == 0:
        wandb.log(
            {
                f"sampling_eval_init/{key}": val
                for key, val in sampling_eval_init_results.items()
            }
        )

    processor.tokenizer.padding_side = "right"
    processor.tokenizer.truncation_side = "right"
    sft_args.run_name = run_name
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=lambda examples: collate_fn(examples, processor),
    )
    trainer.train()

    # save peft model and reload it for final evals
    trainer.save_model()
    del model
    del trainer
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map=None if int(os.getenv("LOCAL_RANK", "-1")) != -1 else "auto",
    )
    model.load_adapter(os.path.join(OUTPUT_DIR))

    sampling_eval_final_results = run_sampling_eval(
        model=model,
        processor=processor,
        dataset=eval_ds,
        save_preds=f"/home/ubuntu/eval_output/{run_name}-eval.json",
    )
    if rank == 0:
        wandb.log(
            {
                f"sampling_eval_final/{key}": val
                for key, val in sampling_eval_final_results.items()
            }
        )
        wandb.finish()
