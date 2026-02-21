# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpeakerScript transcribes video/audio files and identifies who is speaking. Built with Qwen3-ASR + pyannote.audio for speaker diarization, served via FastAPI.

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
ASR_MODEL=Qwen/Qwen3-ASR-0.6B PORT=5002 python api_server.py
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

**Local env file:** Credentials can be placed in a `.env` file (gitignored); `api_server.py` loads it automatically via `python-dotenv`.

## Architecture

### Models
- **ASR**: `Qwen/Qwen3-ASR-0.6B` (~1.2GB) via the `qwen-asr` package. Outperforms Whisper large-v3 on most non-English benchmarks. To use the larger model: `ASR_MODEL=Qwen/Qwen3-ASR-1.7B ./start_local.sh`
- **Diarization**: `pyannote/speaker-diarization-3.1` (~120MB) directly via `pyannote.audio`.

### Device handling
- **ASR** (`get_asr_device()`): `"cuda:0"` → `"mps"` → `"cpu"`. On M3 Mac, ASR runs on MPS.
- **Pyannote** (`get_diarize_device()`): `"cuda"` → `"mps"` → `"cpu"`. On M3 Mac, pyannote also runs on MPS.

### Pipeline (`transcriber.py → Transcriber.transcribe()`)
1. `extract_audio()` — ffmpeg converts input to 16kHz mono WAV numpy array
2. Pyannote diarization on full audio (passed as in-memory tensor) → list of `(start, end, speaker)` turns via `_extract_turns()`
3. For each speaker turn: `slice_audio()` extracts chunk → `Qwen3ASRModel.transcribe()` returns text
4. Segments shorter than `MIN_SEGMENT_DUR` (0.3s) are skipped
5. Returns `{"segments": [...], "language": str, "num_speakers": int}`

`_extract_turns()` handles both pyannote 3.x (`Annotation` with `itertracks()`) and 4.x (`DiarizeOutput`) output formats.

Language is auto-detected from the first segment that returns a language tag. Speaker labels come directly from pyannote (`SPEAKER_00`, `SPEAKER_01`, …) and are displayed as `Speaker 1`, `Speaker 2`, … in the UI.

### API server (`api_server.py`)
Non-blocking job system:
- `POST /api/transcribe` → saves file, starts background thread, returns `job_id`
- `GET /api/job/{job_id}` → poll for `{status, progress, result, error}`
- `POST /api/load` → starts model loading thread (also triggered automatically on startup)
- `GET /api/status` → `{status: not_loaded|loading|loaded|error, progress, model}`
- `POST /api/ask` → Q&A on a transcript via Gemini (requires `GOOGLE_API_KEY`)

Jobs are stored in an in-memory dict (`jobs`). Temp files are cleaned up by the job thread after transcription.

### Frontend (`static/index.html`)
Self-contained single HTML file. Polls `/api/status` on load to trigger model loading. After upload, polls `/api/job/{id}` every 1.5s and maps progress strings to visual stage indicators. Exports as `.txt` or `.srt`. Includes a Q&A panel that calls `/api/ask`.

Speaker labels from pyannote (`SPEAKER_00`, `SPEAKER_01`, …) are displayed as `Speaker 1`, `Speaker 2`, … with distinct colors.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token (required) |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | Any Qwen3-ASR model ID (`Qwen/Qwen3-ASR-1.7B` for best quality) |
| `GOOGLE_API_KEY` | — | Google API key for Gemini Q&A feature (optional) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model used for `/api/ask` |
| `PORT` | `5002` | Server port |
