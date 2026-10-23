"""
IndexTTS2 worker for RunPod serverless  (endpoint nugo6tq9qywm66).

Contract with the website (voice-clone/process-voice.php + voice_jobs.php):
  input : text, reference_audio_url (+ _format, max_reference_seconds, reference_audio_normalized),
          emotion_method (same_as_voice | reference_audio | vectors), emotion_audio_url, emotion_vector,
          audio_upload_url, task ("tts")
  output: {"audio_uploaded": true, "gpu_seconds": N}  (the site finds the WAV at the name it chose when queuing)
          or {"audio_base64": "..."} when no upload link is given / the upload failed and the file is small
          or {"error": "short reason"}  (the site refunds the customer automatically)

Old input names still work: audio_base64 (voice file) and emotion ("happy" / "sad").
"""
import base64
import gc
import os
import shutil
import subprocess
import tempfile
import time

import requests
import runpod
from indextts.infer_v2 import IndexTTS2

UA = {"User-Agent": "marketplace-worker/1.0"}   # some hosts/WAFs reject the default python-requests agent
MAX_INLINE_BYTES = 6 * 1024 * 1024               # RunPod limits the size of a returned result
MIN_REFERENCE_SECONDS = 3.0                      # the site's rule: shorter voice samples are refused

print("Loading IndexTTS 2 model...", flush=True)
tts = IndexTTS2(
    cfg_path="checkpoints_2/config.yaml",
    model_dir="checkpoints_2",
    use_fp16=True,
)
print("Model ready.", flush=True)


def _download(url, dest):
    with requests.get(url, headers=UA, stream=True, timeout=(15, 300)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if os.path.getsize(dest) < 500:
        raise RuntimeError("downloaded audio file is empty")


def _duration(path):
    """Seconds via ffprobe, or None when unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def _get_audio(inp, name, tmp):
    """Path to the audio for 'reference_audio' / 'emotion_audio' (16 kHz mono WAV), or None if not supplied."""
    ext = (inp.get(name + "_format") or "wav").lower().lstrip(".")
    ext = "".join(c for c in ext if c.isalnum())[:5] or "wav"
    raw = os.path.join(tmp, f"{name}.{ext}")
    b64 = inp.get(name)
    if not b64 and name == "reference_audio":
        b64 = inp.get("audio_base64")          # old key name
    if b64:
        if "," in b64[:100]:                    # strip a "data:audio/wav;base64," prefix
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
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", raw,
             "-t", str(int(inp.get("max_reference_seconds") or 30)),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
            check=True, capture_output=True, timeout=120,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"could not read the {name} file")
    return out


def _upload(url, path):
    with open(path, "rb") as f:
        r = requests.post(url, data=f, headers={**UA, "Content-Type": "application/octet-stream"},
                          timeout=(15, 600))
    r.raise_for_status()


def handler(job):
    inp = job.get("input") or {}
    started = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            text = (inp.get("text") or "").strip()
            ref = _get_audio(inp, "reference_audio", tmp)
            if not text or not ref:
                return {"error": "Missing text or voice file (reference_audio / audio_base64)"}

            ref_len = _duration(ref)
            if ref_len is not None and ref_len < MIN_REFERENCE_SECONDS:
                return {"error": f"Voice sample too short ({ref_len:.1f}s, need {MIN_REFERENCE_SECONDS:.0f}s)"}

            kwargs = {}
            method = inp.get("emotion_method")
            if method == "reference_audio":
                emo = _get_audio(inp, "emotion_audio", tmp)
                if emo:
                    kwargs["emo_audio_prompt"] = emo
            elif method == "vectors" and inp.get("emotion_vector"):
                vec = [max(0.0, min(1.0, float(x))) for x in inp["emotion_vector"]]
                if len(vec) != 8:
                    return {"error": "emotion_vector must have 8 numbers"}
                kwargs["emo_vector"] = vec
            elif inp.get("emotion") in ("happy", "sad"):   # old style
                kwargs["use_emo_text"] = True

            out_path = os.path.join(tmp, "output.wav")
            tts.infer(spk_audio_prompt=ref, text=text, output_path=out_path, **kwargs)

            if not os.path.exists(out_path) or os.path.getsize(out_path) < 2000:
                return {"error": "IndexTTS produced no audio"}

            size = os.path.getsize(out_path)
            result = {"status": "success", "gpu_seconds": round(time.time() - started, 1)}

            if inp.get("audio_upload_url"):
                try:
                    _upload(inp["audio_upload_url"], out_path)
                    result["audio_uploaded"] = True
                    result["gpu_seconds"] = round(time.time() - started, 1)
                    return result
                except Exception as e:
                    print("upload failed:", str(e)[:200], flush=True)
                    if size > MAX_INLINE_BYTES:
                        return {"error": "Could not upload the speech: " + str(e)[:150]}
                    # small file: fall through and return it inline

            if size > MAX_INLINE_BYTES:
                return {"error": "Speech too large to return without an upload link"}
            with open(out_path, "rb") as f:
                result["audio_base64"] = base64.b64encode(f.read()).decode()
            return result
    except Exception as e:
        return {"error": str(e)[:300]}
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


runpod.serverless.start({"handler": handler})
