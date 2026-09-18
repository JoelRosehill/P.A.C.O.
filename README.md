# P.A.C.O. server — the brains

PACO (the ESP32) keeps a permanent connection to this server over the internet.
When you speak, PACO streams the audio here; the server turns it into text with
speech recognition that runs locally on this PC (faster-whisper, no cloud), asks
Claude for a short answer, and sends it back to PACO's OLED.

```
PACO ──Wi-Fi hotspot──► internet ──► your public address ──► this server ──► Claude API
     ◄──────────────────────── reply text for the OLED ◄──────────────────
```

## Start it

1. Put your Claude API key in `server\.env` (`ANTHROPIC_API_KEY=...`).
   `PACO_TOKEN` and `DASHBOARD_PASSWORD` were generated for you already.
2. Double-click **`START_SERVER.bat`**. The first start sets up a private Python
   environment (no admin) and downloads the speech model into `server\models\`.
3. Open <http://localhost:8765/> and sign in with `DASHBOARD_PASSWORD`.

## Getting it onto the server PC

```
git clone https://github.com/JoelRosehill/P.A.C.O. server
```

Then copy your private `.env` into that folder (it is never committed; start from
`.env.example`), and run `START_SERVER.bat`. The first start builds the Python
environment and downloads the speech models (~620 MB) into `models\`. To update
later, `git pull` and restart the server; your `.env` and models are left alone.

## Moving it without git

Run **`PACK_SERVER.bat`**. It makes `PACO-server.zip` next to this folder with
everything the server needs (code, settings, the ~480 MB speech model) and
nothing it doesn't (this PC's Python environment, logs, test recordings).
Unzip it on the other PC and run `START_SERVER.bat`; the first start builds a
fresh Python environment there (needs Python 3.10+ and internet, no admin).
The zip contains `.env` with your API key and token, so keep it private.

## Make it reachable from the internet

PACO is on a phone hotspot, so it can only reach the server through a public
address. Pick one:

**A. Port forward (your router).** Forward TCP port `8765` to this PC, then check
<http://YOUR-PUBLIC-IP:8765/health> from your phone's mobile data. Tell PACO:

```
server YOUR-PUBLIC-IP 8765
```

If your home IP changes, use a free dynamic DNS name instead of the raw IP.

**B. A tunnel (no router access needed, and encrypted).** Services such as
Cloudflare Tunnel give you a public `https://` name that forwards to
`http://localhost:8765`. Then tell PACO:

```
server your-name.example.com 443 tls
```

Send those `server ...` commands through the serial monitor (`FLASH.bat` → 3).
PACO saves the address and reconnects on its own — no reflash needed.

## Security

* PACO authenticates with `PACO_TOKEN`; anything else is refused. The dashboard
  needs `DASHBOARD_PASSWORD`. Both live in `.env` — keep that file private.
* With a plain port forward (option A) traffic is not encrypted, so someone on
  the same network path could read the token and use your Claude credits.
  A tunnel with `tls` (option B) avoids that. If a token leaks, put a new one in
  `.env` and `projects\paco\secrets.h`, then rebuild and reflash PACO.

## Settings (`.env`)

| Setting | What it does |
|---|---|
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` — bigger understands more, but is slower |
| `WHISPER_LANGUAGE` | Force a language (`en`, `uk`, `lv`, ...) — more reliable than auto-detect |
| `CLAUDE_MODEL` | Defaults to `claude-opus-5` |
| `CLAUDE_EFFORT` | `low` keeps answers quick; raise it for harder questions |
| `PORT` | Server port (default `8765`) |

Logs are written to `server\logs\paco-server.log`.
