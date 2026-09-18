"""P.A.C.O.'s ears: speech-to-text, run locally with faster-whisper (no cloud, no key)."""

import asyncio
import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("paco.ears")

SAMPLE_RATE = 16000  # what the speech model works in
MODELS_DIR = Path(__file__).resolve().parent / "models"


class Ears:
    """model_name transcribes finished sentences; live_model_name (smaller, faster)
    does the live partial transcripts while someone is still talking."""

    def __init__(self, model_name: str, language: str | None, live_model_name: str | None = None):
        self.model_name = model_name
        self.live_model_name = live_model_name or None
        self.language = language or None
        self._model = None
        self._live_model = None
        self._ready = threading.Event()
        self._error: str | None = None

    def load_in_background(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            self._model = self._open(self.model_name)
            if self.live_model_name and self.live_model_name != self.model_name:
                self._live_model = self._open(self.live_model_name)
            log.info("Speech ready")
        except Exception as e:  # noqa: BLE001 - surface any load failure to the dashboard
            self._error = str(e)
            log.exception("Speech model failed to load")
        finally:
            self._ready.set()

    @staticmethod
    def _open(name: str):
        from faster_whisper import WhisperModel, download_model

        # Models live as plain folders in server\models\<name>, so the server
        # folder can be copied to another PC with the models included.
        path = MODELS_DIR / name
        if not (path / "model.bin").exists():
            log.info("Downloading speech model '%s' (first run only)...", name)
            download_model(name, output_dir=str(path))
        log.info("Loading speech model '%s'...", name)
        return WhisperModel(str(path), device="cpu", compute_type="int8",
                            cpu_threads=os.cpu_count() or 4)

    @property
    def status(self) -> str:
        if not self._ready.is_set():
            return "loading"
        if self._error:
            return "error: " + self._error
        live = f", live: {self.live_model_name}" if self._live_model else ""
        return f"ready ({self.model_name}{live})"

    async def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        """audio: float32 mono at 16 kHz. partial=True trades accuracy for speed."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ready.wait)
        if self._model is None:
            return ""
        return await loop.run_in_executor(None, self._transcribe, audio, partial)

    def _transcribe(self, audio: np.ndarray, partial: bool) -> str:
        if audio.size < SAMPLE_RATE // 4:
            return ""
        # The analog mic is quiet: centre the clip and normalise its level.
        audio = audio - audio.mean()
        peak = float(np.max(np.abs(audio)))
        if peak < 1e-4:
            return ""
        audio = (audio / peak * 0.9).astype(np.float32)

        # Nudge the recogniser towards the assistant's name, so "PACO" isn't heard as "taco".
        word = os.environ.get("WAKE_WORD", "paco").strip()
        hotwords = word.upper() if word.lower() not in ("", "off", "none") else None

        model = self._live_model if (partial and self._live_model) else self._model
        segments, info = model.transcribe(
            audio,
            language=self.language,
            hotwords=hotwords,
            beam_size=1 if partial else 3,
            vad_filter=not partial,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not partial:
            log.info("Heard (%s, %.1fs): %r", info.language, audio.size / SAMPLE_RATE, text)
        return text
