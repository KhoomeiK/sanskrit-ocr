import os
import json
from PIL import Image

from .rendering import render_image


def chunked_reader(f, chunk_size):
    """Yield lists of chunk_size lines, wrap around at EOF."""
    while True:
        lines = []
        for _ in range(chunk_size):
            line = f.readline()
            if not line:
                f.seek(0)
                line = f.readline()
                if not line:
                    raise RuntimeError("Input file is empty")
            lines.append(line.rstrip("\n"))
        yield lines


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with open(INPUT_TXT, "r", encoding="utf-8") as f_in, open(
        CAPTIONS_PATH, "w", encoding="utf-8"
    ) as f_out:
        reader = chunked_reader(f_in, CHUNK_SIZE)
        for idx in range(NUM_SAMPLES):
            lines = next(reader)
            text = " ।\n".join(lines) + "।।"
            images = render_image(text, renderer="random")
            if not images:
                print(f"[WARN] no images for sample {idx}")
                continue

            img = images[0].convert("RGB")
            img_filename = f"img_{idx:04d}.png"
            img_path = os.path.join(IMAGES_DIR, img_filename)
            img.save(img_path)

            record = {"image": img_path, "text": text}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"Generated {idx+1}/{NUM_SAMPLES} samples")

    print(f"Dataset generation complete. Captions at {CAPTIONS_PATH}")


if __name__ == "__main__":
    # Configuration
    INPUT_TXT = "./indic_merged.txt"
    OUTPUT_DIR = "ocr_dataset"
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    CAPTIONS_PATH = os.path.join(OUTPUT_DIR, "captions.jsonl")
    NUM_SAMPLES = 1000
    CHUNK_SIZE = 10

    main()

    import subprocess
    subprocess.run(["tar", "-czf", "ocr_dataset.tar.gz", "ocr_dataset"], check=True)
    print("Created tar.gz archive at ocr_dataset.tar.gz")
