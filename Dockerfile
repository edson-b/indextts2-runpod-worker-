FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y ffmpeg git git-lfs libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN pip install --no-cache-dir -U uv

WORKDIR /app
# Pinned to the commit that was newest when Release #7 was built (30 Sep 2026),
# so a rebuild can never silently pull different index-tts code.
RUN git clone https://github.com/index-tts/index-tts.git && \
    cd index-tts && git checkout d9e41aac89fd00b3d71497fddb287b7f24613712
WORKDIR /app/index-tts
RUN uv sync

RUN uv pip install runpod requests

RUN uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='checkpoints_2')"

RUN uv run python -c "from indextts.utils.model_download import ensure_models_available; ensure_models_available('checkpoints_2')"

COPY handler.py .

CMD ["/app/index-tts/.venv/bin/python", "-u", "handler.py"]
