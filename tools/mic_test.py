r"""Test PACO's microphone through the whole voice pipeline, over USB.

    .venv\Scripts\python.exe tools\mic_test.py [COM5] [--minutes 5] [--no-ask]

PACO records 4 s clips of exactly what it streams to the server and sends them over
the USB cable, until one contains speech. Clips are saved as WAVs in
server\recordings\, transcribed with the server's speech model, and (unless
--no-ask) the words are answered by the brain.
"""

import argparse
import asyncio
import base64
import datetime
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import serial

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import make_brain           # noqa: E402
from ears import SAMPLE_RATE, Ears     # noqa: E402
from paco_server import load_env       # noqa: E402


def capture(ser: serial.Serial) -> tuple[bytes, int] | None:
    """Ask PACO for one 4 s clip. Returns (pcm16, sample_rate)."""
    ser.reset_input_buffer()
    ser.write(b"record\n")
    b64, size, rate, collecting = [], 0, 8000, False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = ser.readline().decode("ascii", "replace").strip()
        if not line:
            continue
        if line.startswith("MICTEST_ERROR"):
            print("  PACO error: " + line, flush=True)
            return None
        if line.startswith("BEGIN_PCM16"):
            _, rate, size = line.split()
            rate, size = int(rate), int(size)
            collecting = True
        elif line.startswith("END_PCM16"):
            pcm = base64.b64decode("".join(b64))
            return (pcm, rate) if len(pcm) == size else None
        elif collecting:
            b64.append(line)
    return None


def to_16k(pcm: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if rate == SAMPLE_RATE:
        return x
    n = int(x.size * SAMPLE_RATE / rate)
    return np.interp(np.arange(n) * rate / SAMPLE_RATE, np.arange(x.size), x).astype(np.float32)


def describe(pcm: bytes, rate: int) -> None:
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    peak = float(np.max(np.abs(a))) if a.size else 0
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0
    clipped = float(np.mean(np.abs(a) >= 32767)) * 100 if a.size else 0
    print(f"  {a.size / rate:.2f} s at {rate} Hz, peak {peak:.0f}/32767, rms {rms:.0f}, clipped {clipped:.1f}%")
    if peak < 1500:
        print("  -> very quiet: raise MIC_GAIN in config.h or speak closer")
    elif clipped > 1:
        print("  -> clipping: lower MIC_GAIN in config.h")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM5")
    ap.add_argument("--minutes", type=float, default=5, help="keep trying this long for speech")
    ap.add_argument("--no-ask", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    ears = Ears(os.environ.get("WHISPER_MODEL", "small"), os.environ.get("WHISPER_LANGUAGE", ""),
                os.environ.get("WHISPER_LIVE_MODEL", "base.en"))
    ears.load_in_background()

    out_dir = ROOT / "recordings"
    out_dir.mkdir(exist_ok=True)
    end = time.monotonic() + args.minutes * 60
    text = ""
    # Open without pulsing DTR/RTS - on this board they reset the ESP32.
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, 115200, 1
    ser.dtr = ser.rts = False
    ser.open()
    with ser:
        n = 0
        while time.monotonic() < end and not text:
            n += 1
            print(f"[clip {n}] recording 4 s - talk to PACO now...", flush=True)
            got = capture(ser)
            if not got:
                print("  transfer failed, retrying", flush=True)
                continue
            pcm, rate = got
            wav_path = out_dir / f"mictest-{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(pcm)
            describe(pcm, rate)
            text = await ears.transcribe(to_16k(pcm, rate))
            print(f"  heard: {text!r}  ({wav_path.name})", flush=True)

    if not text:
        print("No speech recognised.")
        return
    if not args.no_ask:
        brain = make_brain()
        answer = await brain.ask(text)
        print(f"  PACO would answer: {answer}")
        if getattr(brain, "session", None):
            await brain.session.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
