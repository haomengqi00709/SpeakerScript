#!/bin/bash
# Run SpeakerScript locally on Mac (M3 / Apple Silicon)
# Qwen3-ASR and pyannote both run on MPS automatically

set -e

if [ -z "$HF_TOKEN" ]; then
  echo "⚠️  HF_TOKEN is not set — speaker diarization will fail."
  echo "   Export it: export HF_TOKEN=hf_..."
  exit 1
fi

PORT=${PORT:-5002}
ASR_MODEL=${ASR_MODEL:-Qwen/Qwen3-ASR-0.6B}

echo "▶  Starting SpeakerScript on http://localhost:$PORT"
echo "   ASR Model : $ASR_MODEL"
echo "   Device    : MPS (ASR + diarization)"
echo ""

ASR_MODEL=$ASR_MODEL PORT=$PORT python3 api_server.py
