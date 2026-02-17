#!/usr/bin/env python3
"""
SpeakerScript FastAPI server.

Endpoints:
  GET  /              - HTML frontend
  GET  /api/status    - model load status
  POST /api/load      - trigger model loading (non-blocking)
  POST /api/transcribe - upload file, returns job_id immediately
  GET  /api/job/{id}  - poll job status + result
"""

import os
import logging
import tempfile
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
PORT = int(os.getenv("PORT", "5002"))

# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
transcriber = None
model_load_status = "not_loaded"   # not_loaded | loading | loaded | error
model_load_progress = ""
model_load_lock = threading.Lock()

jobs: dict[str, dict[str, Any]] = {}   # job_id -> {status, progress, result, error}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _do_load_models():
    global transcriber, model_load_status, model_load_progress

    def _progress(msg: str):
        global model_load_progress
        model_load_progress = msg

    model_load_status = "loading"
    try:
        from transcriber import Transcriber
        t = Transcriber(model_size=MODEL_SIZE, hf_token=HF_TOKEN)
        t.load(progress_callback=_progress)
        transcriber = t
        model_load_status = "loaded"
        model_load_progress = f"Ready ({MODEL_SIZE})"
        logger.info("Models loaded successfully.")
    except Exception as e:
        model_load_status = "error"
        model_load_progress = str(e)
        logger.error(f"Model loading failed:\n{traceback.format_exc()}")


def start_loading():
    with model_load_lock:
        if model_load_status in ("loading", "loaded"):
            return False
    thread = threading.Thread(target=_do_load_models, daemon=True)
    thread.start()
    return True


# --------------------------------------------------------------------------- #
# Transcription jobs
# --------------------------------------------------------------------------- #
def _run_job(job_id: str, file_path: str, language, min_speakers, max_speakers):
    def _progress(msg: str):
        jobs[job_id]["progress"] = msg

    jobs[job_id]["status"] = "running"
    try:
        result = transcriber.transcribe(
            file_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=_progress,
        )
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        jobs[job_id]["progress"] = "Complete"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed:\n{traceback.format_exc()}")
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="SpeakerScript")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def status():
    return {
        "status": model_load_status,
        "progress": model_load_progress,
        "model": MODEL_SIZE,
    }


@app.post("/api/load")
def load():
    started = start_loading()
    if model_load_status == "loaded":
        return {"success": True, "message": "Already loaded"}
    if not started:
        return {"success": False, "message": f"Status: {model_load_status}"}
    return {"success": True, "message": "Loading started"}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
):
    if model_load_status != "loaded":
        raise HTTPException(status_code=503, detail=f"Models not ready (status: {model_load_status})")

    content = await file.read()
    suffix = Path(file.filename or "upload").suffix or ".wav"

    # Save upload to a temp file — job thread will clean it up
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    lang = None if language == "auto" else language
    min_s = int(min_speakers) if min_speakers.strip().isdigit() else None
    max_s = int(max_speakers) if max_speakers.strip().isdigit() else None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": "Queued...", "result": None, "error": None}

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, tmp.name, lang, min_s, max_s),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set — speaker diarization will fail at load time.")

    logger.info(f"Starting SpeakerScript on port {PORT} (model: {MODEL_SIZE})")
    # Auto-load models on startup
    start_loading()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
