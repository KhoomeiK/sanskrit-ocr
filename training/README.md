# Training

For testing image caption training:
```bash
python make_corpus.py --out ~/output/captions_corpus.txt # only for captions test
python tokenizer.py --corpus text.txt --vocab_size 32768 --output ~/output/tokenizer.model
torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train.py \
    --kind hf \
    --hf_id sizhkhy/open-images-captions-micro \
    --text_col caption \
    --tokenizer ~/output/tokenizer.model \
    --batch 128 \
    --epochs 4
```

For actual OCR training:
```bash
python tokenizer.py --corpus text.txt --vocab_size 32768 --output tokenizer.model

torchrun --standalone -n 8 train.py \
    --meta /path/to/metadata.tsv \
    --tokenizer tokenizer.model \
    --batch 4 --epochs 1
```



For finetuning:
```bash
sudo apt install -y nvidia-cuda-toolkit

poetry install --no-root
$(poetry env activate)
pip install --upgrade "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"
```