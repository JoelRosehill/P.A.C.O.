"""P.A.C.O. server - the brains behind the ESP32 voice assistant.

PACO keeps a WebSocket open to /paco and streams its microphone non-stop (8 kHz
PCM). The server detects speech, sends live transcripts back to PACO's OLED, and
when you stop talking asks the brain (DeepSeek or Claude) and sends the answer.
A password-protected dashboard with a live sound meter lives at /.
"""

import asyncio
import hmac
import json
import logging
import os
import secrets
import time
import unicodedata
from collections import deque
from pathlib import Path

from aiohttp import WSMsgType, web

from brain import Brain, make_brain
from ears import Ears
from listener import Listener
from voice import SPEAK_RATE, Voice

ROOT = Path(__file__).resolve().parent

log = logging.getLogger("paco")


# ---------------------------------------------------------------- settings

def load_env(path: Path) -> None:
    """Minimal .env reader: KEY=value lines, # comments. Real env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set - fill it in server\\.env (see .env.example)")
    return value


# ------------------------------------------------------------- OLED text

# The OLED font covers ASCII and Cyrillic. Everything else is simplified.
_REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "…": "...", " ": " ", "°": " deg",
    "ß": "ss", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE", "€": "EUR", "×": "x",
}


def oled_text(text: str) -> str:
    out = []
    for ch in text:
        ch = _REPLACE.get(ch, ch)
        if len(ch) > 1:
            out.append(ch)
            continue
        code = ord(ch)
        if 32 <= code < 127 or 0x400 <= code <= 0x45F:
            out.append(ch)
        elif ch in "\n\t\r":
            out.append(" ")
        else:
            base = unicodedata.normalize("NFKD", ch)
            base = "".join(c for c in base if not unicodedata.combining(c))
            out.append(base if base and all(32 <= ord(c) < 127 for c in base) else "")
    return " ".join("".join(out).split())


# ------------------------------------------------------------------- hub

class Hub:
    """Owns the PACO connection, the dashboards and the question pipeline."""

    def __init__(self, brain: Brain, ears: Ears, voice: Voice | None):
        self.brain = brain
        self.ears = ears
        self.voice = voice          # None when SPEAK=off
        self.device: web.WebSocketResponse | None = None
        self.device_info: dict = {}
        self.dashboards: set[web.WebSocketResponse] = set()
        self.history: deque[dict] = deque(maxlen=100)
        self.busy = asyncio.Lock()
        self.listener: Listener | None = None

    # -- dashboard updates
    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "device": self.device_info if self.device else None,
            "speech": self.ears.status,
            "voice": self.voice.status if self.voice else "off",
            "model": f"{self.brain.name} {self.brain.model}",
            "history": list(self.history),
        }

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.dashboards:
            try:
                await ws.send_json(message)
            except ConnectionError:
                dead.append(ws)
        for ws in dead:
            self.dashboards.discard(ws)

    async def refresh_dashboards(self) -> None:
        await self.broadcast(self.snapshot())

    # -- talking to PACO
    async def to_device(self, message: dict) -> None:
        if self.device is not None and not self.device.closed:
            try:
                await self.device.send_json(message)
            except ConnectionError:
                pass

    # -- the pipeline
    async def on_listener_event(self, kind: str, data: dict) -> None:
        """Live speech events from the listener -> PACO's screen and the dashboards."""
        await self.to_device({"type": kind, **({"text": oled_text(data["text"])} if "text" in data else {})})
        await self.broadcast({"type": "live", "state": kind, "text": data.get("text", "")})

    async def on_ignored(self, text: str) -> None:
        """Speech that didn't start with the wake word: shown on the dashboard only."""
        await self.broadcast({"type": "live", "state": "ignored", "text": text})

    async def on_utterance(self, text: str, seconds: float) -> None:
        if not text:
            await self.to_device({"type": "idle"})
            await self.broadcast({"type": "live", "state": "idle", "text": ""})
            return
        await self.handle_text(text, source=f"voice {seconds:.1f}s")

    async def handle_text(self, text: str, source: str) -> None:
        async with self.busy:
            if self.listener:
                self.listener.paused = True  # don't start a new utterance mid-answer
            try:
                await self.to_device({"type": "state", "state": "thinking"})
                await self.broadcast({"type": "live", "state": "thinking", "text": text})
                answer = await self._answer(text, source=source)
                if self.voice and answer:
                    await self._speak(answer)
            finally:
                if self.listener:
                    self.listener.paused = False

    async def _answer(self, text: str, source: str) -> str:
        started = time.monotonic()
        answer = await self.brain.ask(text)
        log.info("Q (%s): %r -> A (%.1fs): %r", source, text,
                 time.monotonic() - started, answer)
        await self._reply(heard=text, answer=answer, source=source)
        return answer

    async def _speak(self, text: str) -> None:
        """Say the answer through PACO's speaker, streamed at playback speed.

        Audio goes out as binary WebSocket frames (u8 at SPEAK_RATE) between
        speak_start / speak_end. The listener stays paused (see handle_text) so
        PACO doesn't hear itself.
        """
        device = self.device
        if device is None or device.closed:
            return
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def synthesize() -> None:  # runs in a worker thread
            try:
                for pcm in self.voice.speak(text):
                    loop.call_soon_threadsafe(queue.put_nowait, pcm)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, synthesize)
        await self.to_device({"type": "speak_start", "rate": SPEAK_RATE})

        chunk = SPEAK_RATE // 10          # 100 ms of audio per frame
        lead = SPEAK_RATE // 2            # PACO buffers this much before it starts
        sent = 0
        started = None
        pending = b""
        try:
            while True:
                pcm = await queue.get()
                if pcm is None:
                    break
                pending += pcm
                while len(pending) >= chunk:
                    frame, pending = pending[:chunk], pending[chunk:]
                    if device.closed:
                        return
                    await device.send_bytes(frame)
                    sent += len(frame)
                    started = started or time.monotonic()
                    # Stay about `lead` ahead of playback so PACO's small buffer never overflows.
                    ahead = sent - lead - (time.monotonic() - started) * SPEAK_RATE
                    if ahead > 0:
                        await asyncio.sleep(ahead / SPEAK_RATE)
            if pending and not device.closed:
                await device.send_bytes(pending)
                sent += len(pending)
        except ConnectionError:
            return
        finally:
            if not device.closed:
                await self.to_device({"type": "speak_end"})
        # Keep the mic muted until PACO has finished playing (plus a short tail).
        if started:
            remaining = sent / SPEAK_RATE - (time.monotonic() - started) + 0.4
            if remaining > 0:
                await asyncio.sleep(remaining)

    async def _reply(self, heard: str, answer: str, source: str) -> None:
        await self.to_device({"type": "reply", "heard": oled_text(heard), "text": oled_text(answer)})
        entry = {"t": time.time(), "source": source, "heard": heard, "answer": answer}
        self.history.append(entry)
        await self.broadcast({"type": "exchange", **entry})


# ------------------------------------------------------------ PACO socket

async def paco_socket(request: web.Request) -> web.StreamResponse:
    hub: Hub = request.app["hub"]
    token = request.app["token"]

    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        log.warning("Rejected device connection from %s (bad token)", request.remote)
        await asyncio.sleep(1)
        raise web.HTTPUnauthorized(text="bad token")

    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1024 * 1024)
    await ws.prepare(request)

    if hub.device is not None and not hub.device.closed:
        log.info("New PACO connection replaces the old one")
        await hub.device.close()
    hub.device = ws
    hub.device_info = {"ip": request.remote, "since": time.time()}
    listener = Listener(hub.ears, hub.on_listener_event, hub.on_utterance, hub.on_ignored)
    hub.listener = listener
    log.info("PACO connected from %s", request.remote)
    await hub.refresh_dashboards()

    tasks: set[asyncio.Task] = set()
    last_level = 0.0

    def spawn(coro) -> None:
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                # Continuous microphone stream.
                level = listener.feed(msg.data)
                now = time.monotonic()
                if hub.dashboards and now - last_level > 0.12:
                    last_level = now
                    spawn(hub.broadcast({"type": "level", "v": round(level, 3)}))
                continue
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            kind = data.get("type")
            if kind == "hello":
                hub.device_info.update({k: data.get(k) for k in ("name", "version", "rssi", "heap", "rate")})
                listener.in_rate = int(data.get("rate") or 8000)
                log.info("PACO says hello: %s", data)
                await hub.refresh_dashboards()
            elif kind == "status":
                hub.device_info.update({k: data.get(k) for k in ("rssi", "heap", "uptime")})
                await hub.refresh_dashboards()
            elif kind == "ask":
                text = str(data.get("text", "")).strip()[:500]
                if text:
                    spawn(hub.handle_text(text, source="typed on PACO"))
    finally:
        if hub.device is ws:
            hub.device = None
            hub.listener = None
            log.info("PACO disconnected")
            await hub.refresh_dashboards()
    return ws


# -------------------------------------------------------------- dashboard

SESSION_COOKIE = "paco_session"


def logged_in(request: web.Request) -> bool:
    return request.cookies.get(SESSION_COOKIE, "") in request.app["sessions"]


async def index(request: web.Request) -> web.StreamResponse:
    page = "dashboard.html" if logged_in(request) else "login.html"
    return web.FileResponse(ROOT / "static" / page)


async def login(request: web.Request) -> web.StreamResponse:
    form = await request.post()
    password = str(form.get("password", ""))
    if not hmac.compare_digest(password.encode(), request.app["dashboard_password"].encode()):
        log.warning("Failed dashboard login from %s", request.remote)
        await asyncio.sleep(2)
        raise web.HTTPFound("/?failed=1")
    session = secrets.token_urlsafe(32)
    request.app["sessions"].add(session)
    response = web.HTTPFound("/")
    response.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="Strict", max_age=30 * 86400)
    raise response


async def logout(request: web.Request) -> web.StreamResponse:
    request.app["sessions"].discard(request.cookies.get(SESSION_COOKIE, ""))
    response = web.HTTPFound("/")
    response.del_cookie(SESSION_COOKIE)
    raise response


async def dashboard_socket(request: web.Request) -> web.StreamResponse:
    if not logged_in(request):
        raise web.HTTPUnauthorized()
    hub: Hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    hub.dashboards.add(ws)
    await ws.send_json(hub.snapshot())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if data.get("type") == "ask":
                text = str(data.get("text", "")).strip()[:500]
                if text:
                    asyncio.create_task(hub.handle_text(text, source="dashboard"))
            elif data.get("type") == "reset":
                hub.brain.reset()
                await hub.to_device({"type": "reply", "heard": "", "text": "Memory cleared."})
    finally:
        hub.dashboards.discard(ws)
    return ws


async def health(request: web.Request) -> web.StreamResponse:
    return web.Response(text="P.A.C.O. server is running\n")


async def speech_watcher(app: web.Application) -> None:
    """Tell dashboards when the speech model finishes loading."""
    hub: Hub = app["hub"]
    while hub.ears.status == "loading":
        await asyncio.sleep(1)
    await hub.refresh_dashboards()


async def on_startup(app: web.Application) -> None:
    app["watcher"] = asyncio.create_task(speech_watcher(app))


# ------------------------------------------------------------------- main

def main() -> None:
    load_env(ROOT / ".env")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / "paco-server.log", encoding="utf-8")],
    )
    for noisy in ("aiohttp.access", "httpx", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    token = require("PACO_TOKEN")
    dashboard_password = require("DASHBOARD_PASSWORD")
    port = int(os.environ.get("PORT", "8765"))

    ears = Ears(os.environ.get("WHISPER_MODEL", "small"), os.environ.get("WHISPER_LANGUAGE", ""),
                os.environ.get("WHISPER_LIVE_MODEL", "base.en"))
    ears.load_in_background()
    brain = make_brain()
    voice = None
    if os.environ.get("SPEAK", "on").strip().lower() not in ("off", "0", "no", "false"):
        voice = Voice(os.environ.get("VOICE", "en_US-lessac-medium"))
        voice.load_in_background()

    app = web.Application()
    app["hub"] = Hub(brain, ears, voice)
    app["token"] = token
    app["dashboard_password"] = dashboard_password
    app["sessions"] = set()
    app.on_startup.append(on_startup)
    app.add_routes([
        web.get("/", index),
        web.post("/login", login),
        web.get("/logout", logout),
        web.get("/ui", dashboard_socket),
        web.get("/paco", paco_socket),
        web.get("/health", health),
    ])

    log.info("P.A.C.O. server on port %d - brain: %s %s - dashboard at http://localhost:%d/",
             port, brain.name, brain.model, port)
    web.run_app(app, host="0.0.0.0", port=port, print=None)


if __name__ == "__main__":
    main()
