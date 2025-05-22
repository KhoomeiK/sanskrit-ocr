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

