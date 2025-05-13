import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import nltk
from nltk.metrics.distance import edit_distance
import os
import datasets
from PIL import Image
import tempfile
from tqdm import tqdm
import pytesseract

# Download nltk resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

def download_hf_dataset(dataset_id="rs545837/sanskrit-ocr-images", cache_dir=None):
    """Download and prepare the dataset from Hugging Face"""
    print(f"Downloading dataset {dataset_id} from Hugging Face...")
    
    # Download dataset
    dataset = datasets.load_dataset(dataset_id, cache_dir=cache_dir)
    
    print(f"Dataset downloaded. Available splits: {dataset.keys()}")
    
    # Use the 'train' split by default (adjust as needed)
    split = 'train' if 'train' in dataset else list(dataset.keys())[0]
    ds = dataset[split]
    
    print(f"Dataset info: {len(ds)} samples")
    print(f"Features: {ds.features}")
    
    return ds

def run_tesseract_ocr(dataset, temp_dir=None, lang="san", config=""):
    """Run Tesseract OCR on the dataset images"""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="tesseract_ocr_")
    
    results = []
    
    print(f"Running Tesseract OCR on {len(dataset)} images...")
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Extract relevant data
        image = item.get('image')
        if image is None and 'image_path' in item:
            # If dataset contains paths instead of images
            image = Image.open(item['image_path'])
        
        # Extract ground truth and book info
        ground_truth = item.get('text', '')
        book = item.get('book', f"book_{idx//100}")
        
        # Save image to temp file if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, dict) and 'path' in image:
                image_path = image['path']
            else:
                # Save as temporary image
                image_path = os.path.join(temp_dir, f"image_{idx}.png")
                img = Image.fromarray(image) if hasattr(image, 'shape') else image
                img.save(image_path)
        else:
            # Save as temporary image
            image_path = os.path.join(temp_dir, f"image_{idx}.png")
            image.save(image_path)
        
        # Run OCR
        try:
            ocr_text = pytesseract.image_to_string(image_path, lang=lang, config=config)
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            ocr_text = ""
        
        # Store result
        results.append({
            'ground_truth': ground_truth,
            'ocr_text': ocr_text,
            'book': book,
            'image_path': image_path,
            'page_id': f"page-{idx:03d}"
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(temp_dir, "tesseract_sanskrit_benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"OCR completed. Results saved to {csv_path}")
    return csv_path

# Main function to process the results
def evaluate_tesseract_results(csv_path, output_dir="tesseract_eval_results"):
    """Process Tesseract results and calculate metrics"""
    print(f"Loading results from {csv_path}")
    
    # Load the CSV file
    results_df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_columns = ['ground_truth', 'ocr_text', 'book']
    if not all(col in results_df.columns for col in required_columns):
        print("Warning: CSV file missing required columns. Available columns:", results_df.columns.tolist())
        
        # Try to find alternative column names
        rename_map = {}
        for req_col in required_columns:
            if req_col not in results_df.columns:
                if req_col == 'ground_truth' and 'text' in results_df.columns:
                    rename_map['text'] = 'ground_truth'
                elif req_col == 'ocr_text' and any(col for col in results_df.columns if 'ocr' in col.lower()):
                    ocr_col = next(col for col in results_df.columns if 'ocr' in col.lower())
                    rename_map[ocr_col] = 'ocr_text'
                    
        if rename_map:
            print(f"Renaming columns: {rename_map}")
            results_df = results_df.rename(columns=rename_map)
        
        # Check again after renaming
        missing_cols = [col for col in required_columns if col not in results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure we have the page_id column (if not, create it)
    if 'page_id' not in results_df.columns:
        if 'image_path' in results_df.columns:
            results_df['page_id'] = results_df['image_path'].apply(lambda x: x.split('/')[-1])
        else:
            results_df['page_id'] = [f"page-{i:03d}" for i in range(len(results_df))]
    
    # Calculate metrics
    print("Calculating CER and WER...")
    metrics = []
    
    for idx, row in results_df.iterrows():
        gt = str(row["ground_truth"]).strip()
        pred = str(row["ocr_text"]).strip()
        
        cer = calculate_cer(gt, pred)
        wer = calculate_wer(gt, pred)
        
        metrics.append({
            "page_id": row.get("page_id", f"page-{idx}"),
            "cer": cer,
            "wer": wer,
            "gt_length": len(gt) if gt else 0,
            "pred_length": len(pred) if pred else 0,
            "gt_words": len(gt.split()) if gt else 0,
            "pred_words": len(pred.split()) if pred else 0,
            "book": row["book"]
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Calculate overall metrics
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
    print("\nTesseract OCR Benchmark Results")
    print(f"Dataset: rs545837/sanskrit-ocr-images")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Samples: {len(results_df)}")
    print()
    print(f"Overall Character Error Rate (CER): {overall_cer:.4f}")
    print(f"Overall Word Error Rate (WER): {overall_wer:.4f}")
    print()
    print(f"CER and WER by book:")
    print(book_metrics[['book', 'cer', 'wer']].to_string(index=False))
    print()
    print(f"Detailed Statistics:")
    print(f"Total characters in ground truth: {metrics_df['gt_length'].sum()}")
    print(f"Total characters in OCR output: {metrics_df['pred_length'].sum()}")
    print(f"Total words in ground truth: {metrics_df['gt_words'].sum()}")
    print(f"Total words in OCR output: {metrics_df['pred_words'].sum()}")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary to file
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"Tesseract OCR Benchmark Results\n")
        f.write(f"Dataset: rs545837/sanskrit-ocr-images\n")
        f.write(f"Mode: COMMAND LINE\n")
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
    
    # Save metrics to CSV
    metrics_df.to_csv(f"{output_dir}/tesseract_metrics.csv", index=False)
    book_metrics.to_csv(f"{output_dir}/tesseract_book_metrics.csv", index=False)
    
    # Create a visualization of error rates
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(metrics_df['cer'], bins=20, alpha=0.7)
    plt.xlabel('Character Error Rate (CER)')
    plt.ylabel('Frequency')
    plt.title('CER Distribution')
    plt.axvline(metrics_df['cer'].mean(), color='r', linestyle='--', label=f'Mean: {overall_cer:.4f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(metrics_df['wer'], bins=20, alpha=0.7)
    plt.xlabel('Word Error Rate (WER)')
    plt.ylabel('Frequency')
    plt.title('WER Distribution')
    plt.axvline(metrics_df['wer'].mean(), color='r', linestyle='--', label=f'Mean: {overall_wer:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distributions.png", dpi=150)
    
    # Plot book-level metrics
    sorted_books = book_metrics.sort_values('cer').reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    index = range(len(sorted_books))
    bar_width = 0.35
    
    plt.bar(index, sorted_books['cer'], bar_width, label='CER', color='blue', alpha=0.7)
    plt.bar([i + bar_width for i in index], sorted_books['wer'], bar_width, label='WER', color='orange', alpha=0.7)
    
    plt.xlabel('Book')
    plt.ylabel('Error Rate')
    plt.title('CER and WER by Book')
    plt.xticks([i + bar_width/2 for i in index], sorted_books['book'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/book_metrics.png", dpi=150)
    
    print(f"\nResults saved to {output_dir}/")
    return metrics_df, book_metrics

# Function to show worst examples
def show_worst_examples(csv_path, n=5):
    """Display the worst examples based on CER."""
    # Load results
    results_df = pd.read_csv(csv_path)
    
    # Calculate CER for each row
    results_df['cer'] = [calculate_cer(str(row['ground_truth']), str(row['ocr_text'])) 
                          for _, row in results_df.iterrows()]
    
    # Sort by CER (worst first)
    sorted_df = results_df.sort_values('cer', ascending=False).head(n)
    
    print(f"\n===== {n} WORST OCR EXAMPLES =====\n")
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        gt = str(row['ground_truth'])
        ocr = str(row['ocr_text'])
        cer = row['cer']
        
        print(f"**Book:** {row['book']} **Character Error Rate:** {cer:.4f}")
        print(f"**Ground Truth:**\n{gt}")
        print(f"**OCR Result:**\n{ocr}")
        print(f"{'='*50}\n")

# Main execution
if __name__ == "__main__":
    
    # Download dataset from Hugging Face
    dataset = download_hf_dataset("rs545837/sanskrit-ocr-images")
    
    # Run Tesseract OCR on the dataset images
    results_csv = run_tesseract_ocr(dataset, lang="san")
    
    # Evaluate the results
    evaluate_tesseract_results(results_csv)
    
    # To show worst examples
    show_worst_examples(results_csv, 10)
