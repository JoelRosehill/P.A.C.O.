"""P.A.C.O.'s brain: turns a question into a short answer with an LLM.

Two providers, picked with BRAIN_PROVIDER in .env:
  deepseek - DeepSeek chat API (default), thinking off for fast replies
  claude   - Anthropic Claude API
"""

import datetime
import logging
import os
import time

import aiohttp

log = logging.getLogger("paco.brain")

SYSTEM_PROMPT = """\
You are P.A.C.O. (Personal Assistant for Computational Operations), a voice \
assistant living on a small ESP32 device. The user talks to you through a \
microphone; your answer is shown as text on a tiny 128x64 OLED screen with no \
speaker, about 21 characters per line and 4 lines visible at a time (longer \
answers scroll slowly).

Keep answers short: one to three plain sentences, ideally under 200 characters. \
Use plain text only - no markdown, lists, emoji or special symbols, because the \
screen cannot show them. Answer in the language the user spoke.

The user's words come from speech recognition on a cheap, noisy microphone, so \
they may contain mistakes. Interpret them sensibly. If the text is empty or \
makes no sense, briefly ask the user to repeat."""

# Forget the conversation after this long without a message.
HISTORY_IDLE_S = 10 * 60
# Keep at most this many past exchanges as context.
HISTORY_TURNS = 10


class Brain:
    """Conversation memory shared by every provider."""

    name = "?"

    def __init__(self, model: str):
        self.model = model
        self.history: list[dict] = []
        self.last_used = 0.0

    def reset(self) -> None:
        self.history = []

    def system_prompt(self) -> str:
        now = datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M")
        return f"{SYSTEM_PROMPT}\n\nCurrent local time: {now}."

    async def ask(self, text: str) -> str:
        if time.monotonic() - self.last_used > HISTORY_IDLE_S:
            self.history = []
        self.last_used = time.monotonic()

        messages = self.history + [{"role": "user", "content": text}]
        answer, remember = await self._complete(messages)
        if remember:
            self.history = (messages + [{"role": "assistant", "content": answer}])[-HISTORY_TURNS * 2:]
        return answer

    async def _complete(self, messages: list[dict]) -> tuple[str, bool]:
        """Return (answer, whether to keep this exchange in the history)."""
        raise NotImplementedError


class DeepSeekBrain(Brain):
    name = "DeepSeek"
    URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, model: str, api_key: str, thinking: bool):
        super().__init__(model)
        self.api_key = api_key
        self.thinking = thinking
        self.session: aiohttp.ClientSession | None = None

    async def _complete(self, messages: list[dict]) -> tuple[str, bool]:
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt()}] + messages,
            "max_tokens": 1000,
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with self.session.post(self.URL, json=body, headers=headers) as r:
                data = await r.json(content_type=None)
                status = r.status
        except (aiohttp.ClientError, TimeoutError) as e:
            log.error("Could not reach DeepSeek: %s", e)
            return "The server can't reach its brain right now.", False

        if status == 401:
            log.error("DeepSeek rejected the API key - check DEEPSEEK_API_KEY in .env")
            return "My brain's API key is not valid. Check the server settings.", False
        if status == 402:
            log.error("DeepSeek account is out of balance")
            return "My brain's account has run out of credit.", False
        if status == 429:
            log.warning("DeepSeek rate limit hit")
            return "Too many questions at once. Try again in a minute.", False
        if status != 200:
            log.error("DeepSeek error %s: %s", status, data)
            return "My brain had a problem. Try again.", False

        answer = (data["choices"][0]["message"].get("content") or "").strip()
        return (answer or "I have no answer for that."), bool(answer)


class ClaudeBrain(Brain):
    name = "Claude"

    def __init__(self, model: str, effort: str):
        import anthropic

        super().__init__(model)
        self.anthropic = anthropic
        self.client = anthropic.AsyncAnthropic()
        self.effort = effort

    async def _complete(self, messages: list[dict]) -> tuple[str, bool]:
        anthropic = self.anthropic
        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self.system_prompt(),
                messages=messages,
                output_config={"effort": self.effort},
                # On a policy decline, re-run on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            log.error("Claude rejected the API key - check ANTHROPIC_API_KEY in .env")
            return "My brain's API key is not valid. Check the server settings.", False
        except anthropic.RateLimitError:
            log.warning("Claude rate limit hit")
            return "Too many questions at once. Try again in a minute.", False
        except anthropic.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            return "My brain had a problem. Try again.", False
        except anthropic.APIConnectionError:
            log.error("Could not reach the Claude API")
            return "The server can't reach its brain right now.", False

        if response.stop_reason == "refusal":
            return "Sorry, I can't help with that.", False
        answer = " ".join(b.text for b in response.content if b.type == "text").strip()
        return (answer or "I have no answer for that."), bool(answer)


def make_brain() -> Brain:
    provider = os.environ.get("BRAIN_PROVIDER", "deepseek").strip().lower()
    if provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise SystemExit("ANTHROPIC_API_KEY is not set in server\\.env")
        return ClaudeBrain(os.environ.get("CLAUDE_MODEL", "claude-opus-5"),
                           os.environ.get("CLAUDE_EFFORT", "low"))
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY is not set in server\\.env")
        thinking = os.environ.get("DEEPSEEK_THINKING", "off").strip().lower() in ("on", "1", "true", "yes")
        return DeepSeekBrain(os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"), key, thinking)
    raise SystemExit(f"Unknown BRAIN_PROVIDER '{provider}' - use deepseek or claude")
