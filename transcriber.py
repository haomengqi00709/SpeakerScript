#!/usr/bin/env python3
"""
Core transcription + speaker diarization logic.

Pipeline:
  1. Extract audio from video (ffmpeg)
  2. Run pyannote diarization → speaker turns with timestamps
  3. For each speaker turn, transcribe with Qwen3-ASR
  4. Return combined results with speaker labels
"""

import os
import gc
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv', '.ts'}
SUPPORTED_AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma'}

ASR_MODEL_ID    = "Qwen/Qwen3-ASR-0.6B"
DIARIZE_MODEL   = "pyannote/speaker-diarization-3.1"
MIN_SEGMENT_DUR = 0.3   # skip turns shorter than this (seconds)


def get_asr_device() -> str:
    """Qwen3-ASR: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_diarize_device() -> str:
    """Pyannote: supports CUDA, MPS, and CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_audio(input_path: str, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Convert video or audio to a 16kHz mono float32 numpy array via ffmpeg."""
    out = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(sample_rate), "-vn", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {r.stderr[-500:]}")
    audio, sr = sf.read(out)
    os.unlink(out)
    return audio.astype(np.float32), sr


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> str | None:
    """Save a time slice of audio to a temp WAV file. Returns path or None if empty."""
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    chunk = audio[s:e]
    if len(chunk) < int(MIN_SEGMENT_DUR * sr):
        return None
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, chunk, sr)
    return path


def _extract_turns(diarization, min_dur: float) -> list:
    """Extract (start, end, speaker) tuples from pyannote output.
    Handles Annotation (pyannote 3.x) and DiarizeOutput (pyannote 4.x)."""

    # pyannote 3.x — Annotation object with itertracks()
    if hasattr(diarization, "itertracks"):
        return [
            (seg.start, seg.end, spk)
            for seg, _, spk in diarization.itertracks(yield_label=True)
            if (seg.end - seg.start) >= min_dur
        ]

    # pyannote 4.x DiarizeOutput — annotation lives in .speaker_diarization
    for attr in ("speaker_diarization", "exclusive_speaker_diarization", "annotation", "diarization"):
        wrapped = getattr(diarization, attr, None)
        if wrapped is not None and hasattr(wrapped, "itertracks"):
            return [
                (seg.start, seg.end, spk)
                for seg, _, spk in wrapped.itertracks(yield_label=True)
                if (seg.end - seg.start) >= min_dur
            ]

    # pyannoteai-sdk DiarizeOutput — try direct iteration over turn objects
    turns = []
    try:
        for turn in diarization:
            start   = getattr(turn, "start",   None)
            end     = getattr(turn, "end",     None)
            speaker = getattr(turn, "speaker", getattr(turn, "label", "UNKNOWN"))
            if start is not None and end is not None and (end - start) >= min_dur:
                turns.append((start, end, speaker))
        if turns:
            return turns
    except TypeError:
        pass

    raise RuntimeError(
        f"Unsupported diarization output type: {type(diarization).__name__}\n"
        f"Attributes: {[a for a in dir(diarization) if not a.startswith('_')]}"
    )


class Transcriber:
    def __init__(self, asr_model: str = ASR_MODEL_ID, hf_token: str | None = None):
        self.asr_model_id  = asr_model
        self.hf_token      = hf_token
        self.asr_device    = get_asr_device()
        self.diarize_device = get_diarize_device()
        self.asr_model     = None
        self.diarize_pipeline = None
        self.loaded        = False

    def load(self, progress_callback=None):
        from qwen_asr import Qwen3ASRModel
        from pyannote.audio import Pipeline

        def _p(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _p(f"Loading {self.asr_model_id} on {self.asr_device}...")
        dtype = torch.float32 if self.asr_device == "cpu" else torch.bfloat16
        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.asr_model_id,
            dtype=dtype,
            device_map=self.asr_device,
            max_new_tokens=1024,
        )

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN required for pyannote. Accept licenses at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0"
            )

        _p(f"Loading pyannote diarization pipeline on {self.diarize_device}...")
        self.diarize_pipeline = Pipeline.from_pretrained(
            DIARIZE_MODEL,
            token=self.hf_token,
        )
        self.diarize_pipeline = self.diarize_pipeline.to(torch.device(self.diarize_device))

        self.loaded = True
        _p("All models ready.")

    def transcribe(
        self,
        file_path: str,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Returns:
            {
                "segments": [{"start", "end", "speaker", "text"}, ...],
                "language": str,
                "num_speakers": int,
            }
        """
        if not self.loaded:
            raise RuntimeError("Call load() first.")

        def _p(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # 1. Extract audio
        _p("Extracting audio...")
        audio, sr = extract_audio(file_path)

        try:
            # 2. Diarize — pass in-memory tensor to avoid torchcodec dependency
            _p("Identifying speakers...")
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
            audio_input = {"waveform": waveform, "sample_rate": sr}

            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

            diarization = self.diarize_pipeline(audio_input, **diarize_kwargs)

            # 3. Collect speaker turns — handle pyannote 3.x and 4.x output formats
            turns = _extract_turns(diarization, MIN_SEGMENT_DUR)

            if not turns:
                return {"segments": [], "language": "unknown", "num_speakers": 0}

            # 4. Transcribe each turn with Qwen3-ASR
            segments = []
            detected_language = language or "unknown"

            for i, (start, end, speaker) in enumerate(turns):
                _p(f"Transcribing segment {i + 1}/{len(turns)} ({speaker}, {start:.1f}s–{end:.1f}s)...")

                chunk_path = slice_audio(audio, sr, start, end)
                if chunk_path is None:
                    continue

                try:
                    results = self.asr_model.transcribe(
                        audio=chunk_path,
                        language=language or None,
                    )
                    if not results or not results[0].text.strip():
                        continue

                    text = results[0].text.strip()
                    if detected_language == "unknown" and hasattr(results[0], "language") and results[0].language:
                        detected_language = results[0].language

                    segments.append({
                        "start":   round(start, 2),
                        "end":     round(end,   2),
                        "speaker": speaker,
                        "text":    text,
                    })

                except Exception as e:
                    logger.warning(f"Segment {i + 1} transcription failed: {e}")
                finally:
                    if chunk_path and os.path.exists(chunk_path):
                        os.unlink(chunk_path)

            speakers = list(dict.fromkeys(s["speaker"] for s in segments))
            return {
                "segments":     segments,
                "language":     detected_language,
                "num_speakers": len(speakers),
            }

        finally:
            pass  # no temp files to clean up — pyannote receives in-memory audio
