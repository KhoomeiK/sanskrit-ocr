# Synthetic Data Generation

We need to build vLLM & transformers from source to support gemma3 sampling. dsv3 sampling currently runs into `TypeError: AWQMoEMethod.apply() got an unexpected keyword argument scoring_func` after loading checkpoint.

Make sure you `export HUGGING_FACE_HUB_TOKEN=...` to load gemma.

`python translate_bookcorpus.py --max_passages 5000000`

# Rendering Fonts

To render fonts (with correct conjuncts)

**Linux:** 
1. `apt-get install -y libjpeg-dev zlib1g-dev libfreetype6-dev libharfbuzz-dev libfribidi-dev libraqm-dev pkg-config`
2. `pip install --no-cache-dir --no-binary=:all: Pillow`
3. `pip install fontTools`
**Mac:**
1. `brew install libjpeg libtiff webp little-cms2`
2. `brew install freetype harfbuzz fribidi libraqm pkgconf`
3. `pip install --no-binary=:all: Pillow`
4. `pip install fontTools`

Test with `python datagen/check_sanskrit_fonts.py`
