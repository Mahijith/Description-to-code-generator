"""Turning a recording (or already-written text) into a transcript.

Two interchangeable implementations behind one interface:
- PassthroughTranscriber: the input already IS text (typed, pasted, or a
  .txt/.md file). Always available, needs nothing.
- LocalWhisperTranscriber: real audio/video file upload, transcribed
  entirely on-device via faster-whisper — free, offline, no API key,
  because it never leaves the machine running the app.
"""

from __future__ import annotations

import os
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

    def __init__(self, model_size: str | None = None, download_root: str | None = None):
        # Allow an operator to downgrade the model (e.g. to "tiny") on a
        # memory-constrained deployment without a code change, via env var.
        self._model_size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")
        # Explicit, always-writable cache dir for the Hugging Face download.
        # The huggingface_hub default (~/.cache/huggingface) depends on a
        # writable, persistent HOME, which some hosted environments don't
        # reliably provide — pointing at the system temp dir sidesteps that.
        self._download_root = download_root or os.path.join(tempfile.gettempdir(), "whisper-models")
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
                os.makedirs(self._download_root, exist_ok=True)
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root=self._download_root,
                )
            except Exception as exc:
                # First use downloads the model from Hugging Face Hub — this is
                # where a blocked/unreliable network, a proxy, a Hub outage, or
                # a missing system shared library (e.g. libgomp1, required by
                # ctranslate2 on minimal Linux containers) surfaces. Without
                # this, the raw exception (often several frames deep in
                # httpx/huggingface_hub/ctranslate2) crashes the whole request
                # instead of showing a clear, actionable message.
                raise TranscriptionError(
                    "Couldn't load the speech-to-text model. This usually means "
                    "either outbound internet access is blocked (the model "
                    "downloads from Hugging Face on first use) or a required "
                    f"system library is missing. Underlying error: {exc}. Paste "
                    "the transcript as text instead, or check the network/"
                    "system packages available to this app."
                ) from exc
        return self._model

    def transcribe(self, path: str | Path) -> str:
        path = Path(path)
        audio_path = path
        temp_audio = None
        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                fd, temp_audio_name = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                temp_audio = Path(temp_audio_name)
                _extract_audio(path, temp_audio)
                audio_path = temp_audio

            model = self._get_model()
            try:
                segments, _info = model.transcribe(str(audio_path))
                return " ".join(segment.text.strip() for segment in segments).strip()
            except TranscriptionError:
                raise
            except Exception as exc:
                # A file the model can't decode (corrupt upload, an
                # unsupported/mismatched container) surfaces here as some
                # arbitrary PyAV/ctranslate2 exception — wrap it so the user
                # sees an actionable message instead of a crash.
                raise TranscriptionError(
                    f"Couldn't transcribe the audio: {exc}. The file may be "
                    "corrupt, empty, or in a format that couldn't be decoded — "
                    "try recording/uploading again, or paste the description "
                    "as text instead."
                ) from exc
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
