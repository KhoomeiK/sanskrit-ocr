import editdistance as ed
from metrics import calculate_metrics, rem

def test_metrics():
    original_words = ["नमस्ते", "अहिंसा", "प्राणायाम"]
    modified_words = ["नमस्त", "अहिंस", "प्राणायामा"]
    
    for i, (orig, modified) in enumerate(zip(original_words, modified_words)):
        cer, wer = calculate_metrics(orig, modified)
        
        print(f"Test {i+1}:")
        print(f"Original: {orig}")
        print(f"Modified: {modified}")
        print(f"CER: {cer:.3f}")
        print(f"WER: {wer:.3f}")
        print("-" * 40)
    
    combined_orig = " ".join(original_words)
    combined_mod = " ".join(modified_words)
    
    cer, wer = calculate_metrics(combined_orig, combined_mod)
    
    print("Combined Test:")
    print(f"Original: {combined_orig}")
    print(f"Modified: {combined_mod}")
    print(f"CER: {cer:.3f}")
    print(f"WER: {wer:.3f}")

if __name__ == "__main__":
    test_metrics()