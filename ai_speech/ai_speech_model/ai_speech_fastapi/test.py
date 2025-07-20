import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration
import os
from transformers.utils import logging
import requests

logging.set_verbosity_error()


model_name="openai/whisper-large-v3"


audio_path = "harvard.wav"
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {
    "Authorization": f"Bearer {os.getenv('')}",
    "Content-Type": "audio/wav"
}

if not os.path.exists(audio_path):
    print(f"[Error] Audio file not found: {audio_path}")

try:
    with open(audio_path, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)
    response.raise_for_status()

    result = response.json()
    text = result.get("text", "").strip()
    print(f"Transcribed [{os.path.basename(audio_path)}]: {text}")
    print(text)

except Exception as e:
    print(f"[Error] Failed to transcribe {audio_path}: {e}")
    return ""

