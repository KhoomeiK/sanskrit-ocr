# Synthetic Data Generation

We need to build vLLM & transformers from source to support gemma3 sampling. dsv3 sampling currently runs into `TypeError: AWQMoEMethod.apply() got an unexpected keyword argument scoring_func` after loading checkpoint.

Make sure you `export HUGGING_FACE_HUB_TOKEN=...` to load gemma.

`python translate_bookcorpus.py --max_passages 5000000`