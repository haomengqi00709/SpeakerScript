# Railway deployment — lightweight CPU-only container.
# No ML models here; transcription runs on RunPod Serverless.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# ffmpeg needed only if Railway ever pre-processes audio (currently unused)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.railway.txt .
RUN pip install --no-cache-dir -r requirements.railway.txt

COPY api_server.py .
COPY static/ static/

CMD ["python", "api_server.py"]
