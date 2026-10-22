"""
Fixed handler for the IndexTTS2 worker (endpoint nugo6tq9qywm66).

Accepts BOTH input styles:
  - what your website sends: reference_audio (base64) / reference_audio_url, emotion_method, emotion_vector, ...
  - the old style:           audio_base64, emotion ("neutral" / "happy" / "sad")
Returns {"audio_base64": "..."} (or {"error": "..."}), which your PHP already understands.
"""
import base64
import os
import shutil
import subprocess
import tempfile
import time

import requests
import runpod
from indextts.infer_v2 import IndexTTS2

MAX_INLINE_BYTES = 6 * 1024 * 1024   # RunPod limits the result size; bigger files go by upload link

print("Loading IndexTTS 2 model...")
tts = IndexTTS2(
    cfg_path="checkpoints_2/config.yaml",
    model_dir="checkpoints_2",
    use_fp16=True,
)


def _download(url, dest):
    with requests.get(url, stream=True, timeout=(15, 300)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _get_audio(inp, name, tmp):
    """Return a path to the audio for 'reference_audio' / 'emotion_audio', or None."""
    ext = (inp.get(name + "_format") or "wav").lstrip(".")
    raw = os.path.join(tmp, f"{name}.{ext}")
    b64 = inp.get(name)
    if not b64 and name == "reference_audio":
        b64 = inp.get("audio_base64")          # old key name
    if b64:
        if b64[:100].find(",") != -1:           # strip a "data:audio/wav;base64," prefix
            b64 = b64.split(",", 1)[1]
        with open(raw, "wb") as f:
            f.write(base64.b64decode(b64))
    elif inp.get(name + "_url"):
        _download(inp[name + "_url"], raw)
    else:
        return None

    if inp.get(name + "_normalized") or not shutil.which("ffmpeg"):
        return raw
    out = os.path.join(tmp, f"{name}_16k.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", raw,
         "-t", str(int(inp.get("max_reference_seconds") or 30)),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True, capture_output=True, timeout=120,
    )
    return out


def handler(job):
    inp = job.get("input") or {}
    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            text = (inp.get("text") or "").strip()
            ref = _get_audio(inp, "reference_audio", tmp)
            if not text or not ref:
                return {"error": "Missing text or voice file (reference_audio / audio_base64)"}

            kwargs = {}
            method = inp.get("emotion_method")
            if method == "reference_audio":
                emo = _get_audio(inp, "emotion_audio", tmp)
                if emo:
                    kwargs["emo_audio_prompt"] = emo
            elif method == "vectors" and inp.get("emotion_vector"):
                kwargs["emo_vector"] = [float(x) for x in inp["emotion_vector"]]
            elif inp.get("emotion") in ("happy", "sad"):   # old style
                kwargs["use_emo_text"] = True

            out_path = os.path.join(tmp, "output.wav")
            tts.infer(spk_audio_prompt=ref, text=text, output_path=out_path, **kwargs)

            size = os.path.getsize(out_path)
            result = {"gpu_seconds": round(time.time() - started, 1)}
            if size <= MAX_INLINE_BYTES or not inp.get("audio_upload_url"):
                with open(out_path, "rb") as f:
                    result["audio_base64"] = base64.b64encode(f.read()).decode()
            else:
                with open(out_path, "rb") as f:
                    r = requests.post(inp["audio_upload_url"], data=f,
                                      headers={"Content-Type": "application/octet-stream"},
                                      timeout=(15, 600))
                r.raise_for_status()
                result["audio_uploaded"] = True
            result["status"] = "success"
            return result
    except Exception as e:
        return {"error": str(e)[:300]}


runpod.serverless.start({"handler": handler})
