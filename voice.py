"""P.A.C.O.'s voice: text-to-speech with Piper, run locally (no cloud, no key).

Produces audio in the form PACO's speaker wants: 8-bit unsigned mono at
SPEAK_RATE, because it plays through the ESP32's 8-bit DAC on GPIO25.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger("paco.voice")

VOICES_DIR = Path(__file__).resolve().parent / "models" / "piper"
SPEAK_RATE = 16000


class Voice:
    def __init__(self, voice_name: str):
        self.voice_name = voice_name
        self._voice = None
        self._ready = threading.Event()
        self._error: str | None = None

    def load_in_background(self) -> None:
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            from piper import PiperVoice

            path = VOICES_DIR / f"{self.voice_name}.onnx"
            if not path.exists():
                from piper.download_voices import download_voice

                log.info("Downloading voice '%s' (first run only)...", self.voice_name)
                VOICES_DIR.mkdir(parents=True, exist_ok=True)
                download_voice(self.voice_name, VOICES_DIR)
            self._voice = PiperVoice.load(str(path))
            log.info("Voice '%s' ready", self.voice_name)
        except Exception as e:  # noqa: BLE001 - surface any failure to the dashboard
            self._error = str(e)
            log.exception("Voice failed to load")
        finally:
            self._ready.set()

    @property
    def status(self) -> str:
        if not self._ready.is_set():
            return "loading"
        return "error: " + self._error if self._error else f"ready ({self.voice_name})"

    def speak(self, text: str) -> Iterator[bytes]:
        """Yield PACO-ready audio (u8 @ SPEAK_RATE), one sentence at a time. Blocking."""
        self._ready.wait()
        if self._voice is None or not text.strip():
            return
        volume = float(os.environ.get("SPEAKER_VOLUME", "1.0"))
        for chunk in self._voice.synthesize(_speakable(text)):
            x = chunk.audio_int16_array.astype(np.float32) / 32768.0
            if chunk.sample_rate != SPEAK_RATE:
                n = int(x.size * SPEAK_RATE / chunk.sample_rate)
                x = np.interp(np.arange(n) * chunk.sample_rate / SPEAK_RATE, np.arange(x.size), x)
            # Use the DAC's full range: a bare speaker on the pin is quiet as it is.
            peak = float(np.max(np.abs(x))) or 1.0
            x = np.clip(x / peak * 0.98 * volume, -1.0, 1.0)
            yield (x * 127 + 128).astype(np.uint8).tobytes()


def _speakable(text: str) -> str:
    """Small clean-ups so the voice doesn't read symbols out oddly."""
    text = re.sub(r"[*_#`>]", "", text)
    return text.replace("&", " and ").replace("%", " percent")
