"""Turning a recording (or already-written text) into a transcript.

Two interchangeable implementations behind one interface:
- PassthroughTranscriber: the input already IS text (typed, pasted, or a
  .txt/.md file). Always available, needs nothing.
- GroqWhisperTranscriber: real audio/video file upload, transcribed via
  Groq's hosted Whisper API. Needs one GROQ_API_KEY and outbound internet
  access, but no local model download and no ffmpeg — Groq decodes the
  file server-side, which is why this handles video containers directly
  too. Local, on-device transcription (faster-whisper) was tried first but
  dropped: on Streamlit Community Cloud it needs a system OpenMP library
  the base image doesn't ship, plus a first-use Hugging Face download the
  free tier's resources and network don't reliably support — see
  docs/PROCESS.md.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import requests

TEXT_EXTENSIONS = {".txt", ".md"}

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_GROQ_MODEL = "whisper-large-v3-turbo"


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


class GroqWhisperTranscriber(Transcriber):
    """Sends the audio/video file to Groq's hosted, OpenAI-compatible
    Whisper endpoint. Free tier at console.groq.com/keys.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 120):
        self._api_key = (api_key or os.environ.get("GROQ_API_KEY") or "").strip()
        self._model = model or os.environ.get("GROQ_WHISPER_MODEL", DEFAULT_GROQ_MODEL)
        self._timeout = timeout

    def transcribe(self, path: str | Path) -> str:
        if not self._api_key:
            raise TranscriptionError(
                "No Groq API key configured. Set the GROQ_API_KEY environment "
                "variable — on Streamlit Cloud, add it under the app's "
                "Settings -> Secrets — to a free key from "
                "console.groq.com/keys. Paste the description as text instead "
                "in the meantime."
            )
        path = Path(path)
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    GROQ_TRANSCRIPTION_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": (path.name, f)},
                    data={"model": self._model, "response_format": "json"},
                    timeout=self._timeout,
                )
        except requests.RequestException as exc:
            raise TranscriptionError(f"Request to Groq failed: {exc}") from None

        if resp.status_code == 401:
            raise TranscriptionError("Groq rejected the API key (401 Unauthorized). Check GROQ_API_KEY.")
        if resp.status_code == 413:
            raise TranscriptionError(
                "The recording is too large for Groq's free tier (25MB limit). Try a shorter recording."
            )
        if resp.status_code == 429:
            raise TranscriptionError("Groq rate-limited this request (429) — wait a bit and try again.")
        if resp.status_code >= 400:
            raise TranscriptionError(f"Groq returned {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            return data["text"].strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise TranscriptionError(f"Unexpected response from Groq: {resp.text[:300]}") from exc


def make_transcriber_for(path: str | Path) -> Transcriber:
    """Pick the right Transcriber for a given input file's extension."""
    suffix = Path(path).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return PassthroughTranscriber()
    return GroqWhisperTranscriber()
