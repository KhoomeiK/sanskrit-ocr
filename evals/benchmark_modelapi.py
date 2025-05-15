import os
import argparse
import pandas as pd
from PIL import Image
import time
import io
import re
import base64
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.metrics.distance import edit_distance
from datasets import load_dataset
import sys
from pathlib import Path
from datetime import datetime

# Encoder function for base64
def encode_to_base64(data):
    """Encode bytes to base64 string"""
    return base64.b64encode(data).decode('utf-8')

# Check API availability and print status
def check_api_availability():
    apis = {}
    
    # Try Claude
    try:
        import anthropic
        try:
            encode_base64_fn = anthropic.util.encode_to_base64
        except AttributeError:
            encode_base64_fn = encode_to_base64
        
        apis["claude"] = {
            "client": anthropic,
            "encode_base64": encode_base64_fn
        }
        print("✓ Claude API client available")
    except ImportError:
        print("✗ Claude API client not available (run: pip install anthropic)")
    
    # Try OpenAI
    try:
        from openai import OpenAI
        apis["openai"] = {"client": OpenAI}
        print("✓ OpenAI API client available")
    except ImportError:
        print("✗ OpenAI API client not available (run: pip install openai)")
    
    # Try Gemini (using google-genai exactly as shown)
    try:
        from google import genai
        from google.genai import types
        apis["gemini"] = {
            "client": genai,
            "types": types
        }
        print("✓ Gemini API client available (google-genai)")
    except ImportError:
        print("✗ Gemini API client not available (run: pip install google-genai)")
    
    # Try Mistral OCR
    try:
        from mistralai import Mistral
        apis["mistral"] = {"client": Mistral}
        print("✓ Mistral API client available")
    except ImportError:
        print("✗ Mistral API client not available (run: pip install mistralai)")
    
    return apis

# Calculate error rates
def calculate_cer(gt, pred):
    """Calculate Character Error Rate"""
    if not gt: return 1.0 if pred else 0.0
    return edit_distance(gt, pred) / len(gt)

def calculate_wer(gt, pred):
    """Calculate Word Error Rate"""
    gt_words, pred_words = gt.split(), pred.split()
    if not gt_words: return 1.0 if pred_words else 0.0
    d = [[0 for _ in range(len(pred_words) + 1)] for _ in range(len(gt_words) + 1)]
    for i in range(len(gt_words) + 1): d[i][0] = i
    for j in range(len(pred_words) + 1): d[0][j] = j
    for i in range(1, len(gt_words) + 1):
        for j in range(1, len(pred_words) + 1):
            if gt_words[i - 1] == pred_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return d[len(gt_words)][len(pred_words)] / len(gt_words)

# Extract text with Claude
def extract_text_with_claude(image_path, api_key, model):
    import anthropic
    
    # Get encode_base64 function
    try:
        encode_base64_fn = anthropic.util.encode_to_base64
    except AttributeError:
        encode_base64_fn = encode_to_base64
    
    # Get image bytes
    if isinstance(image_path, str):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    else:
        buffer = io.BytesIO()
        image_path.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
    
    # Call Claude API
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image. Return only the extracted text."},
                {"type": "image", "source": {
                    "type": "base64", 
                    "media_type": "image/jpeg", 
                    "data": encode_base64_fn(image_bytes)
                }}
            ]
        }]
    )
    
    return message.content[0].text

# Extract text with OpenAI
def extract_text_with_openai(image_path, api_key, model):
    from openai import OpenAI
    
    # Get image bytes
    if isinstance(image_path, str):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    else:
        buffer = io.BytesIO()
        image_path.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
    
    # Call OpenAI API
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Extract text precisely as it appears."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract all text from this image. Return only the extracted text."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_to_base64(image_bytes)}"
                }}
            ]}
        ]
    )
    
    return response.choices[0].message.content.strip()

# Extract text with Gemini (exactly as you showed)
def extract_text_with_gemini(image_path, api_key, model, use_upload=True):
    from google import genai
    from google.genai import types
    
    # Initialize client with API key
    client = genai.Client(api_key=api_key)
    
    # Process image based on selected method
    if use_upload:
        # Method 1: Use file upload from your example
        uploaded_file = client.files.upload(file=image_path)
        response = client.models.generate_content(
            model=model,
            contents=[
                "Extract all text from this image. Return only the extracted text.",
                uploaded_file
            ]
        )
    else:
        # Method 2: Use direct bytes from your example
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        response = client.models.generate_content(
            model=model,
            contents=[
                "Extract all text from this image. Return only the extracted text.",
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg'
                )
            ]
        )
        
    return response.text.strip()

# Extract markdown content from Mistral OCR response
def extract_markdown_content(ocr_response):
    """Extract markdown content from OCR response."""
    try:
        # Try to find the markdown content using regex
        if isinstance(ocr_response, str):
            markdown_match = re.search(r"markdown='(.*?)',\s*images=", ocr_response, re.DOTALL)
            if markdown_match:
                # Extract the content and clean up escaped characters
                content = markdown_match.group(1)
                content = content.replace('\\n', '\n').replace("\\'", "'")
                return content
            
            # Alternative approach if the first method fails
            page_object_match = re.search(r"pages=\[(.*?)\]\s*model=", ocr_response, re.DOTALL)
            if page_object_match:
                page_object_str = page_object_match.group(1)
                markdown_match = re.search(r"markdown='(.*?)',", page_object_str, re.DOTALL)
                if markdown_match:
                    content = markdown_match.group(1)
                    content = content.replace('\\n', '\n').replace("\\'", "'")
                    return content
        
        # If response is an object with 'pages' attribute
        elif hasattr(ocr_response, 'pages') and len(ocr_response.pages) > 0:
            # Extract content from the 'markdown' field of the first page
            return ocr_response.pages[0].markdown
        
        # If response is a dict with 'pages' key
        elif isinstance(ocr_response, dict) and 'pages' in ocr_response:
            pages = ocr_response['pages']
            if len(pages) > 0 and 'markdown' in pages[0]:
                return pages[0]['markdown']
        
        # If we can't parse it, check if there's a text attribute
        if hasattr(ocr_response, 'text'):
            return ocr_response.text
            
        # Return the full response as a last resort
        return str(ocr_response)
    
    except Exception as e:
        print(f"Error extracting markdown content: {e}")
        return str(ocr_response)

# Extract text with Mistral OCR (robust implementation with retries)
def extract_text_with_mistral(image_path, api_key, model="mistral-ocr-latest", retry_count=3, retry_delay=5):
    from mistralai import Mistral
    
    # Get base64 encoded image
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            base64_image = encode_to_base64(image_file.read())
    else:
        buffer = io.BytesIO()
        image_path.save(buffer, format="JPEG")
        base64_image = encode_to_base64(buffer.getvalue())
    
    # Call Mistral OCR API with retry logic
    client = Mistral(api_key=api_key)
    
    for attempt in range(retry_count):
        try:
            ocr_response = client.ocr.process(
                model=model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            )
            
            # Extract markdown content
            return extract_markdown_content(ocr_response)
            
        except Exception as e:
            # Handle API errors with appropriate backoff
            if hasattr(e, 'status_code') and e.status_code == 429:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{retry_count}")
                time.sleep(wait_time)
            elif attempt < retry_count - 1:
                print(f"API error: {e}. Retrying {attempt+1}/{retry_count} in {retry_delay}s")
                time.sleep(retry_delay)
            else:
                print(f"Failed after {retry_count} attempts: {e}")
                return ""
    
    return ""

# Run OCR evaluation
def run_ocr_eval(dataset, api_name, api_key, model, output_dir="results", sample_size=10, use_upload=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output directory structure
    Path(f"{output_dir}/visualizations").mkdir(parents=True, exist_ok=True)
    
    # Subset dataset if needed
    if sample_size and sample_size < len(dataset):
        indices = list(range(min(sample_size, len(dataset))))
        dataset = dataset.select(indices)
    
    results = []
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    # Process images
    print(f"Running {api_name} OCR with model {model} on {len(dataset)} images...")
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Get image and ground truth
        image = item.get('image')
        if image is None and 'image_path' in item:
            image = Image.open(item['image_path'])
            
        ground_truth = item.get('text', '')
        book = item.get('book', f"book_{idx//100}")
        
        # Save temporary image
        image_path = os.path.join(image_dir, f"image_{idx}.jpg")
        if not isinstance(image, Image.Image):
            try:
                img = Image.fromarray(image) if hasattr(image, 'shape') else image
                img.save(image_path)
            except Exception as e:
                print(f"Error saving image {idx}: {e}")
                continue
        else:
            image.save(image_path)
        
        # Extract text with appropriate API
        try:
            if api_name == "claude":
                ocr_text = extract_text_with_claude(image_path, api_key, model)
            elif api_name == "openai":
                ocr_text = extract_text_with_openai(image_path, api_key, model)
            elif api_name == "gemini":
                ocr_text = extract_text_with_gemini(image_path, api_key, model, use_upload)
            elif api_name == "mistral":
                ocr_text = extract_text_with_mistral(image_path, api_key, model)
            else:
                raise ValueError(f"Unknown API: {api_name}")
                
            time.sleep(1)  # Rate limit protection
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            ocr_text = ""
        
        # Print first result for debugging
        if idx == 0:
            print("\nSample OCR result:")
            print(f"Ground truth: {ground_truth[:100]}..." if len(ground_truth) > 100 else f"Ground truth: {ground_truth}")
            print(f"OCR text: {ocr_text[:100]}..." if len(ocr_text) > 100 else f"OCR text: {ocr_text}")
        
        results.append({
            'ground_truth': ground_truth,
            'ocr_text': ocr_text,
            'book': book,
            'page_id': f"page-{idx:03d}"
        })
    
    # Save and evaluate results
    if results:
        # Calculate metrics
        metrics = []
        for idx, row in enumerate(results):
            gt = str(row["ground_truth"]).strip()
            pred = str(row["ocr_text"]).strip()
            metrics.append({
                "page_id": row["page_id"],
                "cer": calculate_cer(gt, pred),
                "wer": calculate_wer(gt, pred),
                "book": row["book"],
                "gt_length": len(gt) if gt else 0,
                "pred_length": len(pred) if pred else 0,
                "gt_words": len(gt.split()) if gt else 0,
                "pred_words": len(pred.split()) if pred else 0
            })
        
        metrics_df = pd.DataFrame(metrics)
        overall_cer = metrics_df["cer"].mean()
        overall_wer = metrics_df["wer"].mean()
        
        # Calculate metrics by book
        book_metrics = metrics_df.groupby("book").agg({
            "cer": "mean",
            "wer": "mean",
            "gt_length": "sum",
            "pred_length": "sum",
            "gt_words": "sum",
            "pred_words": "sum"
        }).reset_index()
        
        # Print summary
        print(f"\n{api_name.capitalize()} ({model}) OCR Results")
        print(f"Samples: {len(results)}")
        print(f"Overall Character Error Rate (CER): {overall_cer:.4f}")
        print(f"Overall Word Error Rate (WER): {overall_wer:.4f}")
        print("\nCER and WER by book:")
        print(book_metrics[['book', 'cer', 'wer']].to_string(index=False))
        
        # Save results
        results_df = pd.DataFrame(results)
        model_safe = model.replace('-', '_').replace('.', '_')
        csv_path = os.path.join(output_dir, f"{api_name}_{model_safe}_results.csv")
        results_df.to_csv(csv_path, index=False)
        metrics_df.to_csv(os.path.join(output_dir, f"{api_name}_metrics.csv"), index=False)
        book_metrics.to_csv(os.path.join(output_dir, f"{api_name}_book_metrics.csv"), index=False)
        
        # Save all OCR text to a single file for easy browsing
        with open(os.path.join(output_dir, f"{api_name}_all_ocr_text.txt"), 'w', encoding='utf-8') as f:
            for idx, row in enumerate(results):
                f.write(f"\n\n==== PAGE: {row['page_id']} (Book: {row['book']}) ====\n\n")
                f.write(row['ocr_text'])
        
        # Save summary to text file
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write(f"{api_name.capitalize()} OCR Benchmark Results\n")
            f.write(f"Model: {model}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Samples: {len(results_df)}\n\n")
            f.write(f"Overall Character Error Rate (CER): {overall_cer:.4f}\n")
            f.write(f"Overall Word Error Rate (WER): {overall_wer:.4f}\n\n")
            f.write("CER and WER by book:\n")
            f.write(book_metrics[['book', 'cer', 'wer']].to_string(index=False))
            f.write("\n\nDetailed Statistics:\n")
            f.write(f"Total characters in ground truth: {metrics_df['gt_length'].sum()}\n")
            f.write(f"Total characters in OCR output: {metrics_df['pred_length'].sum()}\n")
            f.write(f"Total words in ground truth: {metrics_df['gt_words'].sum()}\n")
            f.write(f"Total words in OCR output: {metrics_df['pred_words'].sum()}\n")
        
        # Generate visualizations
        generate_visualizations(dataset, results_df, metrics_df, output_dir, 5)
        
        print(f"\nResults saved to: {os.path.abspath(output_dir)}")
        
        return results_df, metrics_df
    
    return None, None

# Generate visualizations of OCR results
def generate_visualizations(dataset, results_df, metrics_df, output_dir, num_examples=5):
    """Generate and save visualizations of sample results."""
    # Select random examples
    if len(results_df) == 0:
        print("No results to visualize")
        return
    
    # Use the same indices for both dataframes
    metrics_df_indexed = metrics_df.set_index('page_id')
    
    sample_indices = np.random.choice(
        len(results_df), min(num_examples, len(results_df)), replace=False
    )
    
    for i, idx in enumerate(sample_indices):
        row = results_df.iloc[idx]
        page_id = row['page_id']
        
        try:
            example_image = dataset[idx]["image"]
            
            plt.figure(figsize=(12, 10))
            plt.imshow(example_image)
            plt.axis('off')
            plt.title(f"Page ID: {page_id} (Book: {row['book']})", fontsize=14)
            
            # Get metrics for this image
            metrics_row = metrics_df_indexed.loc[page_id] if page_id in metrics_df_indexed.index else None
            
            # Prepare text for display (limit to first 300 chars)
            gt_text = row['ground_truth'][:300] + "..." if len(row['ground_truth']) > 300 else row['ground_truth']
            ocr_text = row['ocr_text'][:300] + "..." if len(row['ocr_text']) > 300 else row['ocr_text']
            
            # Add text annotations below the image
            plt.figtext(0.1, 0.09, f"Ground Truth:", fontsize=10, fontweight='bold')
            plt.figtext(0.1, 0.07, gt_text, fontsize=8)
            plt.figtext(0.1, 0.05, f"OCR Result:", fontsize=10, fontweight='bold')
            plt.figtext(0.1, 0.03, ocr_text, fontsize=8)
            
            if metrics_row is not None:
                cer = metrics_row['cer'] if 'cer' in metrics_row else metrics_df.iloc[idx]['cer']
                wer = metrics_row['wer'] if 'wer' in metrics_row else metrics_df.iloc[idx]['wer']
                plt.figtext(0.1, 0.01, f"CER: {cer:.4f}  |  WER: {wer:.4f}", 
                          fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/visualizations/example_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save text comparison to file
            with open(f"{output_dir}/visualizations/example_{i+1}_text.txt", 'w', encoding='utf-8') as f:
                f.write(f"Page ID: {page_id}\n")
                f.write(f"Book: {row['book']}\n\n")
                f.write(f"Ground Truth:\n{row['ground_truth']}\n\n")
                f.write(f"OCR Result:\n{row['ocr_text']}\n\n")
                if metrics_row is not None:
                    f.write(f"CER: {cer:.4f}\n")
                    f.write(f"WER: {wer:.4f}\n")
        
        except Exception as e:
            print(f"Error visualizing example {i}: {e}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM OCR Evaluation')
    parser.add_argument('--api', choices=['claude', 'gemini', 'openai', 'mistral'], required=True,
                       help='Which API to use for OCR')
    parser.add_argument('--key', required=True, help='API key')
    parser.add_argument('--model', help='Model name (defaults to recommended model for each API)')
    parser.add_argument('--dataset', default="rs545837/sanskrit-ocr-images",
                       help='Hugging Face dataset ID')
    parser.add_argument('--output', default="results",
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=496,
                       help='Number of samples to process')
    parser.add_argument('--use_bytes', action='store_true',
                       help='For Gemini: use direct bytes instead of file upload')
    parser.add_argument('--visualize', type=int, default=5,
                       help='Number of sample visualizations to generate (0 to disable)')
    
    args = parser.parse_args()
    
    # Check if required API is available
    apis = check_api_availability()
    if args.api not in apis:
        pkg_name = "anthropic" if args.api == "claude" else "openai" if args.api == "openai" else "google-genai" if args.api == "gemini" else "mistralai"
        print(f"Error: {args.api} API client not installed. Please run: pip install {pkg_name}")
        sys.exit(1)
    
    # Set default model if not specified
    if not args.model:
        default_models = {
            "claude": "claude-3-5-sonnet-20240620",
            "gemini": "gemini-2.0-flash",
            "openai": "gpt-4o",
            "mistral": "mistral-ocr-latest"
        }
        args.model = default_models[args.api]
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.api}_{args.output}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    try:
        print(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        split = 'train' if 'train' in dataset else list(dataset.keys())[0]
        dataset = dataset[split]
        print(f"Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Run OCR evaluation
    results_df, metrics_df = run_ocr_eval(
        dataset, 
        args.api, 
        args.key, 
        args.model, 
        output_dir, 
        args.samples,
        not args.use_bytes if args.api == "gemini" else True
    )
    
    if results_df is not None:
        print("\nEvaluation complete!")
        print("\nTo analyze OCR results in a Jupyter notebook:")
        print(f"results = pd.read_csv('{output_dir}/{args.api}_{args.model.replace('-', '_')}_results.csv')")
        print(f"metrics = pd.read_csv('{output_dir}/{args.api}_metrics.csv')")
        print("\n# Plot error rate distributions")
        print("plt.figure(figsize=(12, 5))")
        print("plt.subplot(1, 2, 1)")
        print("plt.hist(metrics['cer'], bins=20, alpha=0.7)")
        print("plt.xlabel('Character Error Rate (CER)')")
        print("plt.title('CER Distribution')")
        print("plt.subplot(1, 2, 2)")
        print("plt.hist(metrics['wer'], bins=20, alpha=0.7)")
        print("plt.xlabel('Word Error Rate (WER)')")
        print("plt.title('WER Distribution')")
        print("plt.tight_layout()")
        print("plt.show()")
    else:
        print("No results were generated. Check for errors above.")