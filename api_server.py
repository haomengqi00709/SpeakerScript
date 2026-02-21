#!/usr/bin/env python3
"""
SpeakerScript FastAPI server — Railway deployment.

No ML runs here. This server:
  1. Receives file uploads
  2. Uploads them to Cloudflare R2
  3. Submits transcription jobs to RunPod Serverless
  4. Proxies job status back to the frontend

Endpoints:
  GET  /              - HTML frontend
  GET  /api/status    - always returns "loaded" (models live on RunPod)
  POST /api/load      - no-op (kept for frontend compatibility)
  POST /api/transcribe - upload file → R2 → RunPod → returns job_id
  GET  /api/job/{id}  - polls RunPod status, returns result when done
  POST /api/ask       - Q&A on transcript via Gemini
"""

import os
import logging
import uuid
from pathlib import Path
from typing import Any

import boto3
import requests
from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
PORT            = int(os.getenv("PORT", "5002"))

RUNPOD_API_KEY     = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

R2_ACCOUNT_ID        = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME       = os.getenv("R2_BUCKET_NAME")

RUNPOD_BASE    = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUNPOD_HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

# --------------------------------------------------------------------------- #
# R2 client (lazy — avoids crash if env vars missing during local dev)
# --------------------------------------------------------------------------- #
_r2_client = None

def get_r2():
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
        )
    return _r2_client


# --------------------------------------------------------------------------- #
# R2 helpers
# --------------------------------------------------------------------------- #
def upload_to_r2(content: bytes, filename: str) -> tuple[str, str]:
    """Upload bytes to R2. Returns (object_key, presigned_download_url)."""
    suffix = Path(filename).suffix or ".wav"
    key = f"uploads/{uuid.uuid4()}{suffix}"
    get_r2().put_object(Bucket=R2_BUCKET_NAME, Key=key, Body=content)
    url = get_r2().generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET_NAME, "Key": key},
        ExpiresIn=7200,  # 2 hours — enough for any transcription job
    )
    logger.info(f"Uploaded to R2: {key}")
    return key, url


def delete_from_r2(key: str):
    """Best-effort R2 cleanup after job completes."""
    try:
        get_r2().delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        logger.info(f"Deleted from R2: {key}")
    except Exception as e:
        logger.warning(f"R2 cleanup failed for {key}: {e}")


# --------------------------------------------------------------------------- #
# RunPod helpers
# --------------------------------------------------------------------------- #
def submit_runpod_job(file_url: str, language, min_speakers, max_speakers) -> str:
    """Submit a transcription job to RunPod Serverless. Returns RunPod job ID."""
    payload = {
        "input": {
            "file_url":     file_url,
            "language":     language,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
    }
    resp = requests.post(
        f"{RUNPOD_BASE}/run",
        json=payload,
        headers=RUNPOD_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    runpod_job_id = resp.json()["id"]
    logger.info(f"RunPod job submitted: {runpod_job_id}")
    return runpod_job_id


def fetch_runpod_status(runpod_job_id: str) -> dict:
    """Poll RunPod for job status."""
    resp = requests.get(
        f"{RUNPOD_BASE}/status/{runpod_job_id}",
        headers=RUNPOD_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# RunPod status → (our status, progress label)
_STATUS_MAP = {
    "IN_QUEUE":   ("queued",  "Queued..."),
    "IN_PROGRESS": ("running", "Transcribing..."),
    "COMPLETED":  ("done",    "Complete"),
    "FAILED":     ("error",   "Failed"),
    "CANCELLED":  ("error",   "Cancelled"),
    "TIMED_OUT":  ("error",   "Timed out"),
}

# --------------------------------------------------------------------------- #
# In-memory job store
# --------------------------------------------------------------------------- #
jobs: dict[str, dict[str, Any]] = {}

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
    # Models live on RunPod — from the frontend's perspective we're always ready
    return {"status": "loaded", "progress": "Ready (RunPod Serverless)", "model": "RunPod"}


@app.post("/api/load")
def load():
    # No-op: kept so the frontend doesn't break
    return {"success": True, "message": "RunPod handles model loading"}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    min_speakers: str = Form(""),
    max_speakers: str = Form(""),
):
    content = await file.read()
    lang  = None if language == "auto" else language
    min_s = int(min_speakers) if min_speakers.strip().isdigit() else None
    max_s = int(max_speakers) if max_speakers.strip().isdigit() else None

    # 1. Upload file to R2
    try:
        r2_key, file_url = upload_to_r2(content, file.filename or "upload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"R2 upload failed: {e}")

    # 2. Submit job to RunPod
    try:
        runpod_job_id = submit_runpod_job(file_url, lang, min_s, max_s)
    except Exception as e:
        delete_from_r2(r2_key)  # clean up orphaned file
        raise HTTPException(status_code=500, detail=f"RunPod submission failed: {e}")

    # 3. Store job locally (maps our job_id → RunPod job_id + R2 key)
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "runpod_job_id": runpod_job_id,
        "r2_key":        r2_key,
        "status":        "queued",
        "progress":      "Queued...",
        "result":        None,
        "error":         None,
    }

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Already reached a terminal state — return cached result
    if job["status"] in ("done", "error"):
        return job

    # Poll RunPod for latest status
    try:
        rp = fetch_runpod_status(job["runpod_job_id"])
    except Exception as e:
        logger.warning(f"RunPod status fetch failed for {job['runpod_job_id']}: {e}")
        return job  # return last known state; frontend will retry

    rp_status = rp.get("status", "IN_QUEUE")
    our_status, progress = _STATUS_MAP.get(rp_status, ("running", rp_status))

    job["status"]   = our_status
    job["progress"] = progress

    if rp_status == "COMPLETED":
        job["result"] = rp.get("output")
        delete_from_r2(job["r2_key"])

    elif rp_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
        job["error"] = rp.get("error") or rp_status
        delete_from_r2(job["r2_key"])

    return job


@app.post("/api/ask")
async def ask(request: dict):
    question   = request.get("question", "").strip()
    transcript = request.get("transcript", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript is required")
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not set on server")

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""You are analyzing a video/audio transcript with speaker labels.

Rules:
- Answer ONLY using information explicitly stated in the transcript. Do not use any outside knowledge.
- Write in plain prose. Do not quote raw transcript lines or include timestamps/speaker labels in your answer.
- If the transcript does not contain enough information to answer, say "The transcript doesn't cover that."
- Be concise and direct.

TRANSCRIPT:
{transcript}

QUESTION: {question}"""

        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    logger.info(f"Starting SpeakerScript (Railway) on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
