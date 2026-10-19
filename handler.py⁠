import runpod
import os
import torch
import base64
from indextts.infer_v2 import IndexTTS2

# Initialize Model
print("Loading IndexTTS 2 model...")
tts = IndexTTS2(
    cfg_path="checkpoints_2/config.yaml", 
    model_dir="checkpoints_2", 
    use_fp16=True
)

def handler(job):
    """
    Job input expected:
    {
        "text": "Your script text here",
        "audio_base64": "Base64 encoded string of user's voice wav file",
        "emotion": "happy"  # neutral, happy, or sad
    }
    """
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    audio_b64 = job_input.get("audio_base64", "")
    emotion = job_input.get("emotion", "neutral")

    if not text or not audio_b64:
        return {"error": "Missing required fields: 'text' or 'audio_base64'"}

    # Save incoming reference voice to disk
    ref_audio_path = "/tmp/ref_voice.wav"
    output_audio_path = "/tmp/output.wav"
    
    with open(ref_audio_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    # Run IndexTTS 2 inference with simple emotion handling
    use_emo = emotion in ["happy", "sad"]
    tts.infer(
        text=text,
        spk_audio_prompt=ref_audio_path,
        output_path=output_audio_path,
        use_emo_text=use_emo
    )

    # Encode output back to Base64 to return to PHP
    with open(output_audio_path, "rb") as f:
        generated_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Clean up temp files
    if os.path.exists(ref_audio_path): os.remove(ref_audio_path)
    if os.path.exists(output_audio_path): os.remove(output_audio_path)

    return {
        "status": "success",
        "audio_base64": generated_b64
    }

runpod.serverless.start({"handler": handler})
