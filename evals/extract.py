import os
import sys
import zipfile
from pathlib import Path

def extract_dataset(zip_path, extract_dir):
    print(f"Unzipping {zip_path}...")
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Successfully unzipped {zip_path}.")
            return True
        except Exception as e:
            print(f"Failed to unzip {zip_path}: {e}")
            return False
    else:
        print(f"File {zip_path} not found!")
        return False

def extract_all_datasets():
    print("Extracting...")
    print("------------------------------------")
    
    os.makedirs("dataset/", exist_ok=True)
    
    jpeg_success = extract_dataset("dataset/maheshwari2022_jpeg.zip", "dataset/")
    txt_success = extract_dataset("dataset/maheshwari2022_txt.zip", "dataset/")
    
    return jpeg_success and txt_success

if __name__ == "__main__":
    success = extract_all_datasets()
    if not success:
        sys.exit(1)