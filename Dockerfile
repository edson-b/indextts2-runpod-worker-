FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_DISABLE_XET=1
ENV PYTHONUNBUFFERED=1
# Keep caches INSIDE the image so nothing is downloaded at runtime
ENV HF_HOME=/app/hf_cache

# build-essential is needed because insightface compiles a small extension
RUN apt-get update && apt-get install -y git ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/bytedance/LatentSync.git
WORKDIR /app/LatentSync

RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt /app/worker-requirements.txt
RUN pip install --no-cache-dir -r /app/worker-requirements.txt

# Only the two checkpoints inference needs (same as LatentSync's own setup script)
RUN huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints \
 && huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Face detector (InsightFace buffalo_l): normally downloaded on EVERY fresh worker. Bake it in.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(allowed_modules=['detection','landmark_2d_106'], root='checkpoints/auxiliary', providers=['CPUExecutionProvider'])"

# VAE used by the pipeline
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

# Fail the build here (not in production) if something essential is missing
RUN test -f checkpoints/latentsync_unet.pt \
 && test -f checkpoints/whisper/tiny.pt \
 && ls checkpoints/auxiliary/models

# Everything is baked in: block runtime downloads
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY handler.py /app/LatentSync/handler.py

CMD ["python", "-u", "handler.py"]
