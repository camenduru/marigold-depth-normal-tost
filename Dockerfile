FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu121 \
    scipy==1.14.1 matplotlib==3.9.2 h5py==3.11.0 tqdm==4.66.5 numpy==1.26.4 imageio==2.35.1 imageio-ffmpeg==0.5.1 xformers==0.0.27.post2 \
    diffusers==0.30.3 moviepy==1.0.3 transformers==4.44.2 accelerate==0.33.0 sentencepiece==0.2.0 pillow==9.5.0 runpod && \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b tost https://github.com/camenduru/marigold-e2e-ft-depth-hf /content/marigold-e2e-ft && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/scheduler/scheduler_config.json -d /content/depth/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/text_encoder/config.json -d /content/depth/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/resolve/main/text_encoder/model.safetensors -d /content/depth/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/tokenizer/merges.txt -d /content/depth/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/tokenizer/special_tokens_map.json -d /content/depth/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/tokenizer/tokenizer_config.json -d /content/depth/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/tokenizer/vocab.json -d /content/depth/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/unet/config.json -d /content/depth/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/resolve/main/unet/diffusion_pytorch_model.safetensors -d /content/depth/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/vae/config.json -d /content/depth/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/depth/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-depth/raw/main/model_index.json -d /content/depth -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/feature_extractor/preprocessor_config.json -d /content/normals/feature_extractor -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/scheduler/scheduler_config.json -d /content/normals/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/text_encoder/config.json -d /content/normals/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/resolve/main/text_encoder/model.safetensors -d /content/normals/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/tokenizer/merges.txt -d /content/normals/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/tokenizer/special_tokens_map.json -d /content/normals/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/tokenizer/tokenizer_config.json -d /content/normals/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/tokenizer/vocab.json -d /content/normals/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/unet/config.json -d /content/normals/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/resolve/main/unet/diffusion_pytorch_model.safetensors -d /content/normals/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/vae/config.json -d /content/normals/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/normals/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/marigold-e2e-ft-normals/raw/main/model_index.json -d /content/normals -o model_index.json

COPY ./worker_runpod.py /content/marigold-e2e-ft/worker_runpod.py
WORKDIR /content/marigold-e2e-ft
CMD python worker_runpod.py