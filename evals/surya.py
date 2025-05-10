import os
import sys
import json
import csv
import argparse
import subprocess
import shutil
import glob
from tqdm import tqdm
from extract import extract_all_datasets
from metrics import calculate_metrics, rem
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR and calculate metrics")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR processing and only calculate metrics")
    parser.add_argument("--skip-unzip", action="store_true", default=True, help="Skip unzipping dataset files (default: True)")
    return parser.parse_args()

def update_csv(file_path, model_name, wer, cer, book_metrics=None):
    model_row = {
        "model": model_name,
        "wer": f"{wer:.3f}",
        "cer": f"{cer:.3f}"
    }
    
    if book_metrics:
        for book_name, metrics in book_metrics.items():
            model_row[f"{book_name}_wer"] = f"{metrics['wer']:.3f}"
            model_row[f"{book_name}_cer"] = f"{metrics['cer']:.3f}"
    
    existing_rows = []
    existing_header = ["model", "wer", "cer"]
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames
            for row in reader:
                if row["model"] != model_name:
                    existing_rows.append(row)
    
    complete_header = ["model", "wer", "cer"]
    for column in model_row:
        if column not in complete_header:
            complete_header.append(column)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=complete_header)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        writer.writerow(model_row)

def main():
    args = parse_args()
    
    print("Starting the process...")
    
    if not args.skip_unzip:
        if not extract_all_datasets():
            sys.exit(1)
    else:
        print("Skipping dataset extraction as requested.")
    
    print("------------------------------------")
    
    os.makedirs("preds/surya", exist_ok=True)
    
    if not args.skip_ocr:
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "surya_ocr", "editdistance", "tqdm"])
            print("Installation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install packages: {e}")
            sys.exit(1)
        
        print("------------------------------------")
        
        print("Running OCR on dataset/maheshwari2022_jpeg directory...")
        try:
            ocr_path = shutil.which("surya_ocr")
            if ocr_path:
                subprocess.check_call([
                    ocr_path, 
                    os.path.join("dataset", "maheshwari2022_jpeg"), 
                    "--langs", "sa", 
                    "--output_dir", "preds/surya/"
                ])
            else:
                subprocess.check_call([
                    sys.executable, "-m", "surya_ocr", 
                    os.path.join("dataset", "maheshwari2022_jpeg"), 
                    "--langs", "sa", 
                    "--output_dir", "preds/surya/"
                ])
            print("OCR processing completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"OCR processing encountered an error: {e}")
            print("Continuing with available results...")
    else:
        print("Skipping OCR processing as requested.")
    
    print("------------------------------------")
    print("Calculating metrics...")
    
    results_path = "preds/surya/maheshwari2022_jpeg/results.json"
    if not os.path.exists(results_path):
        alternative_path = "preds/surya/results.json"
        if os.path.exists(alternative_path):
            results_path = alternative_path
        else:
            print(f"ERROR: Results file not found at {results_path} or {alternative_path}")
            print("Did you run the OCR process? Try running without --skip-ocr flag.")
            sys.exit(1)
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON in results file at {results_path}")
        print("The file may be corrupted or incomplete.")
        sys.exit(1)
    
    total_cer = 0
    total_wer = 0
    count = 0
    error_files = []
    empty_files = []
    high_error_files = []
    
    book_metrics = defaultdict(lambda: {'cer': 0, 'wer': 0, 'count': 0})
    
    txt_dir = os.path.join("dataset", "maheshwari2022_txt")
    
    if not os.path.exists(txt_dir):
        print(f"ERROR: Text directory {txt_dir} not found!")
        sys.exit(1)
    
    print("Calculating WER and CER metrics...")
    for filename, pages in tqdm(ocr_results.items()):
        book_name = filename.split('_')[0] if '_' in filename else 'unknown'
        
        txt_filename = f"{filename}.txt"
        txt_path = os.path.join(txt_dir, txt_filename)
        
        if not os.path.exists(txt_path):
            error_files.append(filename)
            continue
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read()
            
            predicted_text = ""
            for page in pages:
                for line in page["text_lines"]:
                    predicted_text += line["text"] + " "
            
            predicted_text = predicted_text.strip()
            
            if not predicted_text and not ground_truth:
                empty_files.append(filename)
                continue
                
            cer, wer = calculate_metrics(rem(predicted_text), rem(ground_truth))
            
            if cer > 0.8 or wer > 0.8:
                high_error_files.append((filename, cer, wer, ground_truth, predicted_text))
            
            total_cer += cer
            total_wer += wer
            count += 1
            
            book_metrics[book_name]['cer'] += cer
            book_metrics[book_name]['wer'] += wer
            book_metrics[book_name]['count'] += 1
            
        except Exception as e:
            error_files.append(f"{filename} ({str(e)})")
    
    if count > 0:
        avg_cer = total_cer / count
        avg_wer = total_wer / count
        
        # Calculate averages for each book
        book_averages = {}
        for book_name, metrics in book_metrics.items():
            if metrics['count'] > 0:
                book_averages[book_name] = {
                    'cer': metrics['cer'] / metrics['count'],
                    'wer': metrics['wer'] / metrics['count'],
                    'count': metrics['count']
                }
        
        update_csv("results.csv", "surya", avg_wer, avg_cer, book_averages)
        
        print(f"Metrics calculated and saved to results.csv")
        print(f"Average WER: {avg_wer:.3f}")
        print(f"Average CER: {avg_cer:.3f}")
        print(f"Processed {count} files successfully")
        
        print("\nBook-specific metrics:")
        for book_name, metrics in sorted(book_averages.items(), key=lambda x: x[0]):
            print(f"  {book_name}: WER={metrics['wer']:.3f}, CER={metrics['cer']:.3f} ({metrics['count']} files)")
        
        if empty_files:
            print(f"\nWARNING: Found {len(empty_files)} empty files (skipped in calculation):")
            for i, filename in enumerate(empty_files[:5]):
                print(f"  {i+1}. {filename}")
            if len(empty_files) > 5:
                print(f"  ... and {len(empty_files) - 5} more")
                
        if high_error_files:
            print(f"\nWARNING: Found {len(high_error_files)} files with high error rates:")
            for i, (filename, cer, wer, gt, pred) in enumerate(sorted(high_error_files, key=lambda x: x[2], reverse=True)[:3]):
                print(f"\n  {i+1}. {filename}: CER={cer:.3f}, WER={wer:.3f}")
                print(f"    Ground Truth (truncated to 150 chars):")
                print(f"    {gt[:150]}")
                if len(gt) > 150:
                    print("    ...")
                print(f"    Prediction (truncated to 150 chars):")
                print(f"    {pred[:150]}")
                if len(pred) > 150:
                    print("    ...")
            if len(high_error_files) > 3:
                print(f"  ... and {len(high_error_files) - 3} more")
                
        if error_files:
            print(f"\nERROR: Failed to process {len(error_files)} files:")
            for i, filename in enumerate(error_files[:5]):
                print(f"  {i+1}. {filename}")
            if len(error_files) > 5:
                print(f"  ... and {len(error_files) - 5} more")
    else:
        print("No matching files found for evaluation.")
    
    print("------------------------------------")
    print("Process completed successfully!")

if __name__ == "__main__":
    main()