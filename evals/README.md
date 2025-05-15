# Benchmark

## Benchmark Tesseract
Installing the Dependencies:
```
apt-get update
apt-get install -y tesseract-ocr
apt-get install -y tesseract-ocr-san
```

```
python -m venv venv 
source venv/bin/activate
pip install pandas numpy matplotlib nltk datasets pillow tqdm pytesseract huggingface_hub -q
```


## Benchmark Closed Models(OpenAI, Gemini, Mistral, Claude)

```
# For Claude
pip install anthropic

# For OpenAI
pip install openai

# For Gemini
pip install google-genai

# For Mistral
pip install mistralai
```

```
python ocr_benchmark.py --api [API_NAME] --key [YOUR_API_KEY] [OPTIONS]
```

Optional Arguments
--model: Model name (defaults to recommended model for each API)
--dataset: Hugging Face dataset ID (default: "rs545837/sanskrit-ocr-images")
--output: Output directory name (default: "results")
--samples: Number of samples to process (default: 10)
--visualize: Number of sample visualizations to generate (default: 5, 0 to disable)

