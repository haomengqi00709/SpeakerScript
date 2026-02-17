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

### Device handling
- **Whisper / alignment** — uses `whisperx` which wraps `faster-whisper` (CTranslate2 backend). Supports `cuda` and `cpu` only — no MPS.
- **Speaker diarization** — uses `whisperx.DiarizationPipeline` which wraps pyannote. Supports `cuda`, `mps`, and `cpu`.
- `get_whisper_device()` returns `"cuda"` or `"cpu"`.
- `get_torch_device()` returns `"cuda"`, `"mps"`, or `"cpu"`.

### Pipeline (`transcriber.py → Transcriber.transcribe()`)
1. If input is a video file → extract 16kHz mono WAV with ffmpeg
2. `whisperx.load_audio()` → numpy array
3. `whisper_model.transcribe()` → segments with text, rough timestamps
4. `whisperx.load_align_model()` + `whisperx.align()` → word-level timestamps (alignment model is per-language, auto-downloaded from HF; may not exist for all languages — gracefully skipped)
5. `diarize_model(audio_path)` → pyannote annotation of who spoke when
6. `whisperx.assign_word_speakers()` → each segment gets a `speaker` field (`SPEAKER_00`, `SPEAKER_01`, ...)
7. Returns `{"segments": [...], "language": str, "num_speakers": int}`

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
| `WHISPER_MODEL` | `large-v3` | Any whisperx-compatible model size |
| `PORT` | `5002` | Server port |

`batch_size` inside `transcribe()` is automatically set to 16 (CUDA) or 4 (CPU) based on device.
