"""Turns PACO's continuous audio stream into utterances.

PACO streams its microphone non-stop. The listener runs the Silero speech detector
over the stream, and when someone talks it transcribes them. Only speech that
starts with the wake word ("PACO, ...") is for PACO:
  - once the name is heard it reports "listening" and live partial transcripts,
  - when they stop, the request (without the name) goes to on_final(),
  - just "PACO" on its own shows "Yes?" and the next sentence counts, no name needed,
  - everything else goes to on_ignored() and never reaches the brain.
"""

import asyncio
import logging
import os
import time
from typing import Awaitable, Callable

import numpy as np

from ears import SAMPLE_RATE, Ears
from wake import WakeWord

log = logging.getLogger("paco.listener")

VAD_STEP = 4096            # run the detector every 256 ms of new audio (16 kHz samples)
VAD_WINDOW = 16384         # ... over the last 1.024 s, so the model has context
SPEECH_PROB = 0.5          # Silero probability that counts as speech
PREROLL_S = 0.6            # keep this much audio from before speech was detected
END_SILENCE_S = 0.9        # this much quiet ends the utterance
MAX_UTTERANCE_S = 15.0
MIN_SPEECH_S = 0.3         # shorter blips are ignored
PARTIAL_EVERY_S = 1.0
FIRST_PARTIAL_S = 0.7      # first live transcript sooner, to catch the wake word quickly
KEEP_S = 30                # audio history kept in memory
FOLLOWUP_S = 8.0           # after a bare "PACO", this long to say the request

Event = Callable[[str, dict], Awaitable[None]]


class Listener:
    def __init__(self, ears: Ears, on_event: Event, on_final: Callable[[str, float], Awaitable[None]],
                 on_ignored: Callable[[str], Awaitable[None]] | None = None):
        self.ears = ears
        self.on_event = on_event
        self.on_final = on_final
        self.on_ignored = on_ignored
        # WAKE_WORD=off in .env answers everything, like before.
        word = os.environ.get("WAKE_WORD", "paco").strip()
        self.wake = WakeWord(word, required=word.lower() not in ("", "off", "none"))
        self.awake_until = 0.0       # set by a bare "PACO": next sentence needs no name
        self.woken = False           # the current utterance is addressed to PACO
        self.finalizing = 0          # finished utterances still being transcribed
        self.in_rate = 8000
        self.paused = False          # set while the brain is answering
        # Digital boost for quiet analog mics (MIC_BOOST in .env). Helps the speech
        # detector notice softer speech; it amplifies room noise just as much.
        self.boost = float(os.environ.get("MIC_BOOST", "4"))

        self.audio = np.zeros(0, dtype=np.float32)   # 16 kHz, float -1..1
        self.offset = 0              # absolute sample index of self.audio[0]
        self.last_in = 0.0           # last input sample, for seamless resampling
        self.pending = 0             # new samples since the last detector run

        self.noise_floor = 1e-3    # background loudness, for the meter
        self.speaking = False
        self.start = 0               # absolute index where the utterance starts
        self.last_speech = 0         # absolute index of the last speech frame
        self.speech_samples = 0
        self.last_partial = 0
        self.stt_task: asyncio.Task | None = None
        self.stt_lock = asyncio.Lock()
        self.vad = None

    # ---------------------------------------------------------------- audio in

    @property
    def end(self) -> int:
        return self.offset + self.audio.size

    def feed(self, pcm: bytes) -> float:
        """Add raw PCM16 from PACO. Returns the chunk's loudness (0..1) for meters."""
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if x.size == 0:
            return 0.0
        x = np.clip(x * self.boost, -1.0, 1.0)

        # Upsample to 16 kHz (linear, continuing from the previous chunk).
        factor = SAMPLE_RATE / self.in_rate
        src = np.concatenate(([self.last_in], x))
        n_out = int(round(x.size * factor))
        up = np.interp(np.arange(1, n_out + 1) / factor, np.arange(src.size), src).astype(np.float32)
        self.last_in = float(x[-1])

        self.audio = np.concatenate((self.audio, up))
        excess = self.audio.size - KEEP_S * SAMPLE_RATE
        if excess > 0 and not self.speaking:
            self.audio = self.audio[excess:]
            self.offset += excess
        self.pending += up.size

        while self.pending >= VAD_STEP:
            self.pending -= VAD_STEP
            self._detect()

        # Meter: loudness relative to the room's noise floor (like PACO's OLED meter):
        # the floor follows quiet quickly and loud sounds only slowly.
        rms = float(np.sqrt(np.mean(x * x))) + 1e-6
        rate = 0.1 if rms < self.noise_floor else 0.002
        self.noise_floor += (rms - self.noise_floor) * rate
        return float(min(1.0, max(0.0, np.log10(rms / self.noise_floor) / 1.3)))  # 1x..20x the floor

    # ------------------------------------------------------------ detection

    def _detect(self) -> None:
        if self.vad is None:
            from faster_whisper.vad import get_vad_model
            self.vad = get_vad_model()
        if self.audio.size < VAD_WINDOW:
            return

        window = self.audio[-VAD_WINDOW:]
        probs = np.asarray(self.vad(window)).reshape(-1)
        recent = probs[-VAD_STEP // 512:]
        speech = float(recent.max()) > SPEECH_PROB
        now = self.end

        if speech:
            self.last_speech = now
            if self.speaking:
                self.speech_samples += VAD_STEP

        if not self.speaking:
            if speech and not self.paused:
                self.speaking = True
                self.start = max(self.offset, now - VAD_STEP - int(PREROLL_S * SAMPLE_RATE))
                self.speech_samples = VAD_STEP
                # Schedules the first partial FIRST_PARTIAL_S after speech starts.
                self.last_partial = now - int((PARTIAL_EVERY_S - FIRST_PARTIAL_S) * SAMPLE_RATE)
                # Without the wake word nothing shows until a partial transcript has the name.
                self.woken = not self.wake.required or time.monotonic() < self.awake_until
                if self.woken:
                    asyncio.get_running_loop().create_task(self.on_event("listening", {}))
            return

        silent_for = (now - self.last_speech) / SAMPLE_RATE
        length = (now - self.start) / SAMPLE_RATE
        if silent_for >= END_SILENCE_S or length >= MAX_UTTERANCE_S:
            self._finish()
        elif (now - self.last_partial) / SAMPLE_RATE >= PARTIAL_EVERY_S and not self._stt_busy():
            self.last_partial = now
            clip = self._clip(self.start, now)
            self.stt_task = asyncio.get_running_loop().create_task(self._partial(clip))

    def _clip(self, start: int, end: int) -> np.ndarray:
        return self.audio[start - self.offset:end - self.offset].copy()

    def _stt_busy(self) -> bool:
        return self.stt_task is not None and not self.stt_task.done()

    def _finish(self) -> None:
        self.speaking = False
        end = min(self.end, self.last_speech + int(0.3 * SAMPLE_RATE))
        clip = self._clip(self.start, end)
        if self.speech_samples / SAMPLE_RATE < MIN_SPEECH_S:
            asyncio.get_running_loop().create_task(self.on_event("idle", {}))
            return
        asyncio.get_running_loop().create_task(self._final(clip))

    # -------------------------------------------------------- transcription

    async def _partial(self, clip: np.ndarray) -> None:
        async with self.stt_lock:
            if not self.speaking:
                return
            text = await self.ears.transcribe(clip, partial=True)
        if not text or not self.speaking:
            return
        called, rest = self.wake.split(text)
        if not self.woken:
            if not called:
                return  # not (yet) talking to PACO
            self.woken = True
            await self.on_event("listening", {})
        if rest:
            await self.on_event("partial", {"text": rest})

    async def _final(self, clip: np.ndarray) -> None:
        self.finalizing += 1
        try:
            async with self.stt_lock:
                text = await self.ears.transcribe(clip)
        finally:
            self.finalizing -= 1
        seconds = clip.size / SAMPLE_RATE
        called, rest = self.wake.split(text)

        if not (self.woken or called or not self.wake.required):
            log.info("Ignored (no wake word): %r", text)
            if self.on_ignored and text:
                await self.on_ignored(text)
            return

        if called and not rest:
            # Just "PACO": wait for the actual request.
            self.awake_until = time.monotonic() + FOLLOWUP_S
            if self.speaking:
                # They already started the request while "PACO" was being transcribed.
                if not self.woken:
                    self.woken = True
                    await self.on_event("listening", {})
                return
            await self.on_event("listening", {})
            await self.on_event("partial", {"text": "Yes?"})
            asyncio.get_running_loop().create_task(self._fall_asleep(self.awake_until))
            return

        self.awake_until = 0.0
        await self.on_final(rest if called else text, seconds)

    async def _fall_asleep(self, deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - time.monotonic()) + 0.2)
        if self.awake_until == deadline and not self.speaking and not self.finalizing:
            self.awake_until = 0.0
            await self.on_event("idle", {})
