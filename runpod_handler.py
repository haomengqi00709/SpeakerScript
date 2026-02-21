#!/usr/bin/env python3
"""
RunPod Serverless handler for SpeakerScript.

Loads Qwen3-ASR + pyannote once at worker startup,
then processes jobs: downloads audio from R2, transcribes, returns result.

Environment variables (set in RunPod template):
  HF_TOKEN   - Hugging Face token for pyannote models
  ASR_MODEL  - optional, defaults to Qwen/Qwen3-ASR-0.6B
"""

import os
import tempfile
import logging

import requests
import runpod

from transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN  = os.getenv("HF_TOKEN")
ASR_MODEL = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")

# --------------------------------------------------------------------------- #
# Load models once at worker startup (runs before any job is accepted)
# --------------------------------------------------------------------------- #
logger.info(f"Loading models ({ASR_MODEL})...")
transcriber = Transcriber(asr_model=ASR_MODEL, hf_token=HF_TOKEN)
transcriber.load()
logger.info("Models ready — worker accepting jobs.")


# --------------------------------------------------------------------------- #
# Job handler
# --------------------------------------------------------------------------- #
def handler(job):
    """
    Expected input:
      {
        "file_url":     str,           # presigned R2 URL
        "language":     str | null,    # e.g. "en", null = auto-detect
        "min_speakers": int | null,
        "max_speakers": int | null,
      }

    Returns the transcriber result dict:
      {"segments": [...], "language": str, "num_speakers": int}
    """
    job_input    = job["input"]
    file_url     = job_input["file_url"]
    language     = job_input.get("language")
    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")

    # Derive extension from URL (strip query params first)
    clean_url = file_url.split("?")[0]
    suffix = os.path.splitext(clean_url)[-1] or ".wav"

    tmp_path = None
    try:
        logger.info("Downloading file from R2...")
        resp = requests.get(file_url, stream=True, timeout=300)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in resp.iter_content(chunk_size=65536):
                tmp.write(chunk)
            tmp_path = tmp.name

        logger.info(f"Downloaded to {tmp_path} — starting transcription...")

        result = transcriber.transcribe(
            tmp_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        logger.info(f"Transcription complete: {len(result.get('segments', []))} segments.")
        return result

    except Exception as e:
        logger.error(f"Handler error: {e}")
        raise

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


runpod.serverless.start({"handler": handler})
