# Cluster Initialization Script
sudo apt update
sudo apt install -y mosh
sudo apt install -y nvidia-cuda-toolkit
sudo apt install -y libhdf5-dev

curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/KhoomeiK/sanskrit-ocr
cd sanskrit-ocr/datagen
poetry install --no-root
$(poetry env activate)
pip install --upgrade "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3"

cd
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .

cd ~/sanskrit-ocr/datagen
