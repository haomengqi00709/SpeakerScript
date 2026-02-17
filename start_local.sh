#!/bin/bash
# Run SpeakerScript locally on Mac (M3 Max / Apple Silicon)
# WhisperX runs on CPU (CTranslate2 is very fast on M3)
# PyAnnote diarization uses MPS automatically

set -e

if [ -z "$HF_TOKEN" ]; then
  echo "⚠️  HF_TOKEN is not set."
  echo "   Export it before running: export HF_TOKEN=hf_..."
  echo "   Also accept the pyannote model licenses:"
  echo "     https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo "     https://huggingface.co/pyannote/segmentation-3.0"
  exit 1
fi

PORT=${PORT:-5002}
WHISPER_MODEL=${WHISPER_MODEL:-large-v3}

echo "▶  Starting SpeakerScript on http://localhost:$PORT"
echo "   Model : $WHISPER_MODEL"
echo "   Device: CPU (whisper) + MPS (diarization)"
echo ""

WHISPER_MODEL=$WHISPER_MODEL PORT=$PORT python api_server.py
