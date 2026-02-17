#!/bin/bash
# One-shot setup and run for RunPod GPU instances
# Run this from the /workspace directory

set -e

echo "=== SpeakerScript RunPod Setup ==="

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN environment variable not set."
  echo "Set it in RunPod Pod settings before starting."
  exit 1
fi

# Install PyTorch (CUDA 12.1) if not already present
python -c "import torch; print('PyTorch', torch.__version__)" 2>/dev/null || \
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install system ffmpeg
apt-get install -y ffmpeg 2>/dev/null || true

# Install Python deps
pip install -r requirements.txt

echo ""
echo "=== Starting server on port 5002 ==="
python api_server.py
