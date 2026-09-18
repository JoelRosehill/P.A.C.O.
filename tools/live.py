r"""Live transcript: talk to PACO and watch your words appear in this terminal.

    .venv\Scripts\python.exe tools\live.py [COM5] [--no-ask]

PACO streams its microphone over the USB cable and this PC stands in for the
server: the same speech detector and speech model pick out what you say, and
(unless --no-ask) the brain answers. PACO's OLED shows it all, just as it will
with the real server. Ctrl+C to stop.
"""

import argparse
import asyncio
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import make_brain                  # noqa: E402
from ears import Ears                         # noqa: E402
from listener import Listener                 # noqa: E402
from paco_server import load_env, oled_text   # noqa: E402

# ANSI colours
DIM, CYAN, GREEN, YELLOW, BOLD, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[1m", "\033[0m"
CLEAR_LINE = "\r\033[2K"


class Terminal:
    """One live status line at the bottom, finished lines scroll above it."""

    def __init__(self):
        self.level = 0.0
        self.label = f"{DIM}waiting for you to speak{RESET}"
        self.text = ""

    def width(self) -> int:
        return shutil.get_terminal_size((100, 20)).columns

    def draw(self) -> None:
        bar_w = 20
        filled = int(self.level * bar_w)
        colour = GREEN if self.level < 0.6 else YELLOW
        bar = f"{colour}{'#' * filled}{DIM}{'.' * (bar_w - filled)}{RESET}"
        room = max(10, self.width() - bar_w - 30)
        text = self.text if len(self.text) <= room else "..." + self.text[-(room - 3):]
        sys.stdout.write(f"{CLEAR_LINE}mic [{bar}] {self.label} {text}")
        sys.stdout.flush()

    def print(self, line: str) -> None:
        sys.stdout.write(f"{CLEAR_LINE}{line}\n")
        self.draw()


def open_quietly(port: str) -> serial.Serial:
    """Open the port without pulsing DTR/RTS - on this board they reset the ESP32."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    return ser


class UsbLink:
    """PACO's audio stream in over USB, JSON messages for its screen out."""

    def __init__(self, port: str):
        self.ser = open_quietly(port)
        self.lock = threading.Lock()

    def start(self) -> int:
        self.ser.reset_input_buffer()
        deadline = time.monotonic() + 10
        next_try = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_try:  # repeat in case PACO is still booting
                self.ser.write(b"usbstream\n")
                next_try = time.monotonic() + 1.5
            line = self.ser.readline().decode("ascii", "replace").strip()
            if line.startswith("USBSTREAM"):
                _, baud, rate = line.split()
                self.ser.baudrate = int(baud)
                return int(rate)
        raise SystemExit("PACO did not answer - is it plugged in, on the right COM port, "
                         "and running firmware v0.1.8 or newer?")

    def send(self, message: dict) -> None:
        import json
        with self.lock:
            self.ser.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))

    def ping(self) -> None:
        with self.lock:
            self.ser.write(b"ping\n")

    def stop(self) -> None:
        try:
            with self.lock:
                self.ser.write(b"usbstop\n")
                self.ser.flush()
            time.sleep(0.2)
        finally:
            self.ser.close()

    def frames(self, on_audio) -> None:
        """Reader thread: pull PCM frames out of the byte stream (skipping log text)."""
        buf = bytearray()
        while self.ser.is_open:
            try:
                buf += self.ser.read(4096)
            except (serial.SerialException, TypeError):
                return
            while True:
                i = buf.find(b"\xa5\x5a")
                if i < 0:
                    del buf[:-1]
                    break
                if len(buf) < i + 4:
                    del buf[:i]
                    break
                n = buf[i + 2] | (buf[i + 3] << 8)
                if n == 0 or n > 4096:
                    del buf[:i + 2]
                    continue
                if len(buf) < i + 4 + n + 1:
                    del buf[:i]
                    break
                data = bytes(buf[i + 4:i + 4 + n])
                if (sum(data) & 0xFF) == buf[i + 4 + n]:
                    on_audio(data)
                    del buf[:i + 5 + n]
                else:
                    del buf[:i + 2]  # false sync inside text; look further


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="COM5")
    ap.add_argument("--no-ask", action="store_true", help="only transcribe, don't ask the brain")
    args = ap.parse_args()

    os.system("")  # enables ANSI colours in the Windows console
    sys.stdout.reconfigure(encoding="utf-8")
    load_env(ROOT / ".env")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    print(f"{BOLD}P.A.C.O. live transcript{RESET}  -  Ctrl+C to stop\n")
    ears = Ears(os.environ.get("WHISPER_MODEL", "small.en"), os.environ.get("WHISPER_LANGUAGE", "en"),
                os.environ.get("WHISPER_LIVE_MODEL", "base.en"))
    print(f"{DIM}loading speech model '{ears.model_name}'...{RESET}")
    ears.load_in_background()
    await asyncio.get_running_loop().run_in_executor(None, ears._ready.wait)
    if ears.status.startswith("error"):
        raise SystemExit(ears.status)
    brain = None if args.no_ask else make_brain()

    link = UsbLink(args.port)
    rate = link.start()
    wake = os.environ.get("WAKE_WORD", "paco")
    hint = f'Start with "{wake.upper()}, ..."' if wake.lower() not in ("", "off", "none") else "Talk to it!"
    print(f"{DIM}connected to PACO on {args.port}, {rate} Hz audio. {hint}{RESET}\n")

    term = Terminal()
    loop = asyncio.get_running_loop()

    async def on_event(kind: str, data: dict) -> None:
        text = data.get("text", "")
        link.send({"type": kind, **({"text": oled_text(text)} if text else {})})
        if kind == "listening":
            term.label, term.text = f"{CYAN}listening:{RESET}", ""
        elif kind == "partial":
            term.label, term.text = f"{CYAN}hearing:{RESET}", text
        elif kind == "idle":
            term.label, term.text = f"{DIM}waiting for you to speak{RESET}", ""
        term.draw()

    async def on_final(text: str, seconds: float) -> None:
        if not text:
            link.send({"type": "idle"})
            term.print(f"{DIM}({seconds:.1f}s of sound, no words recognised){RESET}")
            term.label, term.text = f"{DIM}waiting for you to speak{RESET}", ""
            term.draw()
            return
        term.print(f"{BOLD}You:{RESET}  {text}")
        if brain is None:
            link.send({"type": "reply", "heard": oled_text(text), "text": oled_text(text)})
        else:
            listener.paused = True
            link.send({"type": "state", "state": "thinking"})
            term.label, term.text = f"{YELLOW}thinking...{RESET}", ""
            term.draw()
            answer = await brain.ask(text)
            link.send({"type": "reply", "heard": oled_text(text), "text": oled_text(answer)})
            term.print(f"{GREEN}{BOLD}PACO:{RESET} {answer}")
            listener.paused = False
        term.label, term.text = f"{DIM}waiting for you to speak{RESET}", ""
        term.draw()

    async def on_ignored(text: str) -> None:
        term.print(f"{DIM}(not for PACO - no wake word): {text}{RESET}")

    listener = Listener(ears, on_event, on_final, on_ignored)
    listener.in_rate = rate

    def on_audio(pcm: bytes) -> None:
        loop.call_soon_threadsafe(handle_audio, pcm)

    last_draw = [0.0]

    def handle_audio(pcm: bytes) -> None:
        term.level = listener.feed(pcm)
        now = time.monotonic()
        if now - last_draw[0] > 0.08:
            last_draw[0] = now
            term.draw()

    threading.Thread(target=link.frames, args=(on_audio,), daemon=True).start()
    try:
        while True:
            await asyncio.sleep(2)
            link.ping()
    finally:
        link.stop()
        if brain is not None and getattr(brain, "session", None):
            await brain.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{DIM}stopped - PACO is back to normal{RESET}")
