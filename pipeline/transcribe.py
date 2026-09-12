"""Turning a recording (or already-written text) into a transcript.

Two interchangeable implementations behind one interface:
- PassthroughTranscriber: the input already IS text (typed, pasted, or a
  .txt/.md file). Always available, needs nothing.
- LocalWhisperTranscriber: real audio/video file upload, transcribed
  entirely on-device via faster-whisper — free, offline, no API key,
  because it never leaves the machine running the app.
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TEXT_EXTENSIONS = {".txt", ".md"}


class TranscriptionError(RuntimeError):
    pass


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, path: str | Path) -> str:
        """Return the transcript text for the file at `path`."""


class PassthroughTranscriber(Transcriber):
    """For .txt/.md files, or any path whose bytes are already the transcript."""

    def transcribe(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()

    @staticmethod
    def from_text(text: str) -> str:
        return text.strip()


class LocalWhisperTranscriber(Transcriber):
    """Runs faster-whisper on-device. No API key, no network call.

    Requires the optional `faster-whisper` dependency and, for video
    files, `ffmpeg` on PATH to extract the audio track first.
    """

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None  # lazy: only download/load if actually used

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed. Install it, or paste the "
                    "transcript as text instead."
                ) from exc
            try:
                self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            except Exception as exc:
                # First use downloads the model from Hugging Face Hub — this is
                # where a blocked/unreliable network, a proxy, or a Hub outage
                # surfaces. Without this, the raw exception (often several
                # frames deep in httpx/huggingface_hub) crashes the whole
                # request instead of showing a clear, actionable message.
                raise TranscriptionError(
                    "Couldn't load the speech-to-text model (it downloads from "
                    "Hugging Face on first use, which needs outbound internet "
                    f"access). Underlying error: {exc}. Paste the transcript as "
                    "text instead, or check the network this app is running on."
                ) from exc
        return self._model

    def transcribe(self, path: str | Path) -> str:
        path = Path(path)
        audio_path = path
        temp_audio = None
        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                temp_audio = Path(tempfile.mkstemp(suffix=".wav")[1])
                _extract_audio(path, temp_audio)
                audio_path = temp_audio

            model = self._get_model()
            segments, _info = model.transcribe(str(audio_path))
            return " ".join(segment.text.strip() for segment in segments).strip()
        finally:
            if temp_audio is not None and temp_audio.exists():
                temp_audio.unlink()


def _extract_audio(video_path: Path, out_wav_path: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(out_wav_path)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError("ffmpeg is not installed; cannot extract audio from video") from exc
    except subprocess.CalledProcessError as exc:
        raise TranscriptionError(f"ffmpeg failed to extract audio: {exc.stderr.decode(errors='replace')[:300]}") from exc


def make_transcriber_for(path: str | Path) -> Transcriber:
    """Pick the right Transcriber for a given input file's extension."""
    suffix = Path(path).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return PassthroughTranscriber()
    return LocalWhisperTranscriber()
