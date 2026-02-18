# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpeakerScript transcribes video/audio files and identifies who is speaking. Built with WhisperX (faster-whisper backend) + pyannote.audio for speaker diarization.

**Required environment variable:**
```bash
export HF_TOKEN=your_huggingface_token_here
```
The token must have access to both pyannote model licenses (accept at huggingface.co):
- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`

## Running

**Mac (M3/Apple Silicon) — local testing:**
```bash
chmod +x start_local.sh && ./start_local.sh
# Opens at http://localhost:5002
```

**RunPod:**
```bash
chmod +x runpod_setup.sh && ./runpod_setup.sh
```

**Manual:**
```bash
WHISPER_MODEL=large-v3 PORT=5002 python api_server.py
```

**System dependency:** `ffmpeg` must be installed (`brew install ffmpeg` on Mac).

**Install Python deps:**
```bash
# Mac (MPS support included in standard PyTorch wheel)
pip install torch torchaudio
pip install -r requirements.txt

# RunPod (CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Architecture

### Models
- **ASR**: `Qwen/Qwen3-ASR-0.6B` (~1.2GB) via the `qwen-asr` package. Outperforms Whisper large-v3 on most non-English benchmarks. To use the larger model: `ASR_MODEL=Qwen/Qwen3-ASR-1.7B ./start_local.sh`
- **Diarization**: `pyannote/speaker-diarization-3.1` (~120MB) directly via `pyannote.audio`.

### Device handling
- **ASR** (`get_asr_device()`): `"cuda:0"` or `"cpu"`. MPS not used — transformers MPS support for this model is unverified.
- **Pyannote** (`get_diarize_device()`): `"cuda"`, `"mps"`, or `"cpu"`. On M3 Mac, pyannote runs on MPS automatically.

### Pipeline (`transcriber.py → Transcriber.transcribe()`)
1. `extract_audio()` — ffmpeg converts input to 16kHz mono WAV numpy array
2. Pyannote diarization on full audio → list of `(start, end, speaker)` turns
3. For each speaker turn: `slice_audio()` extracts chunk → `Qwen3ASRModel.transcribe()` returns text
4. Segments shorter than `MIN_SEGMENT_DUR` (0.3s) are skipped
5. Returns `{"segments": [...], "language": str, "num_speakers": int}`

Note: Language is auto-detected from the first segment that returns a language tag. Speaker labels come directly from pyannote (`SPEAKER_00`, `SPEAKER_01`, …) and are displayed as `Speaker 1`, `Speaker 2`, … in the UI.

### API server (`api_server.py`)
Non-blocking job system:
- `POST /api/transcribe` → saves file, starts background thread, returns `job_id`
- `GET /api/job/{job_id}` → poll for `{status, progress, result, error}`
- `POST /api/load` → starts model loading thread (also triggered automatically on startup)
- `GET /api/status` → `{status: not_loaded|loading|loaded|error, progress, model}`

Jobs are stored in an in-memory dict (`jobs`). Temp files are cleaned up by the job thread after transcription.

### Frontend (`static/index.html`)
Self-contained single HTML file. Polls `/api/status` on load to trigger model loading. After upload, polls `/api/job/{id}` every 1.5s and maps progress strings to visual stage indicators. Exports as `.txt` or `.srt`.

Speaker labels from pyannote (`SPEAKER_00`, `SPEAKER_01`, …) are displayed as `Speaker 1`, `Speaker 2`, … with distinct colors.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token (required) |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Any Qwen3-ASR model ID (`Qwen/Qwen3-ASR-1.7B` for best quality) |
| `PORT` | `5002` | Server port |
