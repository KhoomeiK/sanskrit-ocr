# Synthetic Data Generation

We need to build vLLM & transformers from source to support gemma3 sampling. dsv3 sampling currently runs into `TypeError: AWQMoEMethod.apply() got an unexpected keyword argument scoring_func` after loading checkpoint.

Make sure you `export HUGGING_FACE_HUB_TOKEN=...` to load gemma.

`python translate_bookcorpus.py --max_passages 5000000`

# Rendering Requirements
1. `apt-get update`
2. `apt-get install -y libjpeg-dev zlib1g-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev libraqm-dev pkg-config poppler-utils`
3. `pip install --no-cache-dir --no-binary=:all: Pillow`
4. `pip install fontTools`

You'll also need the following packages installed for synthetic data:

`pip install weasyprint jinja2 tqdm pdf2image opencv-python imageio pydyf==0.10.0`

## Rendering System Usage
Each renderer can be used standalone within `datagen/` to generate samples:
```
python augmentations/rendering/render_book_page.py --n 10        # Generate 10 book page samples
python augmentations/rendering/render_parchment_leaf.py --n 10
```

## Dataset Generation
Generate complete OCR datasets with automatic text chunking and multiple rendering styles within `datagen/`:
```
# Process entire text file with 1 image per chunk
python generate_dataset.py --input-txt sample_sa.txt

# Generate 5 images per text chunk (same text, different renderings)
python generate_dataset.py --input-txt sample_sa.txt -n 5

# Process only first 100 chunks with 3 images each
python generate_dataset.py --input-txt sample_sa.txt --num-samples 100 -n 3

# Create tar.gz archive of output
python generate_dataset.py --input-txt sample_sa.txt --create-archive
```

## Adding New Layouts
To add a new rendering layout:
1. Create augmentations/rendering/render_mynewlayout.py
2. Implement a `render(text: str, use_max: bool = False) -> Tuple[PIL.Image, str]` function
3. Add corresponding template in `augmentations/templates/mynewlayout.html.jinja`
4. The system will automatically discover and use your new renderer.