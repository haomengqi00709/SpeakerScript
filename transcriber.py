#!/usr/bin/env python3
"""
Core transcription + speaker diarization logic.

Pipeline:
  1. Extract audio from video (ffmpeg)
  2. Transcribe with WhisperX (faster-whisper backend)
  3. Align word-level timestamps
  4. Diarize with pyannote (via whisperx.DiarizationPipeline)
  5. Assign speakers to segments
"""

import os
import gc
import logging
import subprocess
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv', '.wmv', '.ts'}
SUPPORTED_AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma'}


def get_whisper_device() -> str:
    """CTranslate2 supports cuda and cpu only (no MPS)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device() -> str:
    """Full PyTorch device — includes MPS for Apple Silicon."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def extract_audio(input_path: str, sample_rate: int = 16000) -> str:
    """Convert video or audio to a 16kHz mono WAV using ffmpeg."""
    output_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr[-500:]}")
    return output_path


class Transcriber:
    def __init__(self, model_size: str = "large-v3", hf_token: str | None = None):
        self.model_size = model_size
        self.hf_token = hf_token
        self.whisper_device = get_whisper_device()
        self.torch_device = get_torch_device()
        self.compute_type = get_compute_type(self.whisper_device)
        self.whisper_model = None
        self.diarize_model = None
        self.loaded = False

    def load(self, progress_callback=None):
        import whisperx

        def _progress(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _progress(f"Loading Whisper {self.model_size} on {self.whisper_device} ({self.compute_type})...")
        self.whisper_model = whisperx.load_model(
            self.model_size,
            device=self.whisper_device,
            compute_type=self.compute_type,
        )

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN is required for speaker diarization.\n"
                "Accept the license at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  https://huggingface.co/pyannote/segmentation-3.0"
            )

        _progress(f"Loading pyannote diarization pipeline on {self.torch_device}...")
        self.diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.torch_device,
        )

        self.loaded = True
        _progress("All models ready.")

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
        import whisperx

        if not self.loaded:
            raise RuntimeError("Call load() before transcribe().")

        def _progress(msg):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # --- 1. Extract audio if needed ---
        suffix = Path(file_path).suffix.lower()
        temp_audio = None
        audio_path = file_path

        if suffix in SUPPORTED_VIDEO or suffix not in SUPPORTED_AUDIO:
            _progress("Extracting audio from video...")
            temp_audio = extract_audio(file_path)
            audio_path = temp_audio

        try:
            # --- 2. Load audio ---
            _progress("Loading audio...")
            audio = whisperx.load_audio(audio_path)

            # --- 3. Transcribe ---
            _progress("Transcribing with Whisper...")
            batch_size = 16 if self.whisper_device == "cuda" else 4
            result = self.whisper_model.transcribe(
                audio,
                language=language,  # None = auto-detect
                batch_size=batch_size,
            )
            detected_language = result.get("language", "unknown")
            _progress(f"Detected language: {detected_language}")

            # --- 4. Align word timestamps ---
            _progress(f"Aligning word timestamps...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.whisper_device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.whisper_device,
                    return_char_alignments=False,
                )
                del model_a
                gc.collect()
            except Exception as e:
                logger.warning(f"Alignment skipped (language '{detected_language}' may not be supported): {e}")

            # --- 5. Diarize ---
            _progress("Identifying speakers...")
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_segments = self.diarize_model(audio_path, **diarize_kwargs)

            # --- 6. Assign speakers ---
            _progress("Merging transcript with speaker labels...")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # --- 7. Format output ---
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": round(seg.get("start", 0), 2),
                    "end": round(seg.get("end", 0), 2),
                    "speaker": seg.get("speaker", "UNKNOWN"),
                    "text": seg.get("text", "").strip(),
                })

            speakers = list(dict.fromkeys(s["speaker"] for s in segments))
            return {
                "segments": segments,
                "language": detected_language,
                "num_speakers": len(speakers),
            }

        finally:
            if temp_audio and os.path.exists(temp_audio):
                os.unlink(temp_audio)
