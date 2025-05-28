import os
import json
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from augmentations.render_random import generate_dataset


def main(args):
    """Main function to generate OCR dataset."""
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    captions_path = output_dir / "captions.jsonl"
    
    with open(args.input_txt, "r", encoding="utf-8") as f:
        lines = (line.strip() for line in f if line.strip())
        
        print(f"Generating {'all' if args.num_samples is None else args.num_samples} samples with {args.images_per_sample} images per chunk...")
        samples = generate_dataset(lines, num_samples=args.num_samples, use_max=args.use_max, images_per_sample=args.images_per_sample)
    
    # Save images and captions
    with open(captions_path, "w", encoding="utf-8") as f_out:
        for idx, (img, caption) in enumerate(samples):
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img_filename = f"img_{idx:04d}.png"
            img_path = images_dir / img_filename
            img.save(img_path)
            
            record = {
                "image": str(img_path),
                "text": caption
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            if (idx + 1) % 100 == 0:
                print(f"Saved {idx + 1}/{len(samples)} images")
    
    print(f"Dataset generation complete!")
    print(f"Total images generated: {len(samples)}")
    print(f"Images saved to: {images_dir}")
    print(f"Captions saved to: {captions_path}")
    
    # Create tar.gz archive if requested
    if args.create_archive:
        import subprocess
        archive_name = f"{output_dir}.tar.gz"
        subprocess.run(["tar", "-czf", archive_name, str(output_dir)], check=True)
        print(f"Created archive: {archive_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OCR dataset with Sanskrit text")
    parser.add_argument("--input-txt", type=str, default="./sample_sa.txt",
                        help="Input text file with phrases")
    parser.add_argument("--output-dir", type=str, default="ocr_dataset",
                        help="Output directory for dataset")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to generate (default: all)")
    parser.add_argument("--use-max", action="store_true",
                        help="Use maximum sizing parameters for rendering")
    parser.add_argument("-n", "--images-per-sample", type=int, default=1,
                        help="Number of images to generate per text chunk (default: 1)")
    parser.add_argument("--create-archive", action="store_true",
                        help="Create tar.gz archive of output")
    
    args = parser.parse_args()
    main(args)