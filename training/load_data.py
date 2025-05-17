import os
from datasets import load_dataset, Dataset
from PIL import Image
from io import BytesIO

SYSTEM_PROMPT = "You are an expert OCR model. Transcribe text (which may contain Sanskrit, English, Hindi, etc.) from the image the user provides. Output the entire extracted text and nothing else."
USER_PROMPT = "Extract text from this image."


def load_multimodal_dataset(captions_jsonl: str):
    """
    Returns a HuggingFace Dataset with columns: 'image', 'text'.
    'image' is a PIL.Image loaded on access.
    """
    # Load raw records
    ds = load_dataset("json", data_files=captions_jsonl, split="train")

    # Define a function to load images
    def load_image(example):
        try:
            image_path = "/home/ubuntu/" + example["image"]
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                return example
            example["image"] = Image.open(image_path).convert("RGB")
            return example
        except Exception as e:
            print(f"Error loading image {example['image']}: {str(e)}")
            return example

    # Map to load images lazily
    ds = ds.map(load_image, batched=False, num_proc=1)
    return ds


def process_vision_info(messages: list[dict]):
    images = []
    for msg in messages:
        for content in msg.get("content", []):
            if content.get("type") == "image":
                if isinstance(content["image"], dict) and "bytes" in content["image"]:
                    images.append(
                        Image.open(BytesIO(content["image"]["bytes"])).convert("RGB")
                    )
                elif isinstance(content["image"], Image.Image):
                    images.append(content["image"].convert("RGB"))
    return images


# Collate function to build `messages` structures and tokenize
def collate_fn(examples: list[dict], processor, eval=False):
    batch_messages = []
    for ex in examples:
        text = ex["text"]
        img = ex["image"]
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image", "image": img},
                ],
            },
        ]
        if not eval:
            msgs.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
        batch_messages.append({"messages": msgs})

    # Processor will flatten messages and tokenize images+text
    tokenized = processor(
        text=[
            processor.apply_chat_template(
                m["messages"], tokenize=False, add_generation_prompt=False
            )
            for m in batch_messages
        ],
        images=[process_vision_info(m["messages"]) for m in batch_messages],
        return_tensors="pt",
        padding=True,
    )

    labels = tokenized.input_ids.clone()
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100  # gemma image token id
    tokenized["labels"] = labels

    return tokenized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions", type=str, default="/home/ubuntu/ocr_dataset/captions.jsonl"
    )
    args = parser.parse_args()
    ds = load_multimodal_dataset(args.captions)
    print(ds)
