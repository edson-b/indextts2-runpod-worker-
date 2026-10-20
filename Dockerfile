FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# System tools (git-lfs is needed to pull IndexTTS2's large files)
RUN apt-get update && apt-get install -y ffmpeg git git-lfs libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# uv is the installer the IndexTTS2 project officially uses
RUN pip install --no-cache-dir -U uv

# Get the official IndexTTS2 code and install its dependencies
WORKDIR /app
RUN git clone https://github.com/index-tts/index-tts.git
WORKDIR /app/index-tts
RUN uv sync

# RunPod's worker library, installed into the same environment
RUN uv pip install runpod

# Download the IndexTTS2 model files into the folder handler.py expects
RUN uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints_2')"

# Your handler
COPY handler.py .

CMD ["/app/index-tts/.venv/bin/python", "-u", "handler.py"]
