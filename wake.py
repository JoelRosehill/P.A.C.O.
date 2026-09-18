"""Wake word: PACO only answers speech that starts with its name ("PACO, ...").

Speech recognition spells a made-up name several ways (Paco, Pako, Pacco,
P.A.C.O., "Pa co"), so the match is forgiving about spelling but strict about
position: the name must be the first word, optionally after a greeting.
"""

import difflib
import re

GREETINGS = {"hey", "hi", "hello", "ok", "okay", "oi", "yo", "so"}
# Common mis-hearings of "PACO" that a plain similarity check would miss or rank low.
ALIASES = {"paco", "pako", "pacco", "packo", "paku", "pacho", "poco", "pakko", "pacos", "pacoe"}


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


class WakeWord:
    def __init__(self, word: str = "paco", required: bool = True):
        self.word = _norm(word) or "paco"
        self.required = required

    def _is_name(self, token: str) -> bool:
        if token == self.word or (self.word == "paco" and token in ALIASES):
            return True
        return len(token) >= 3 and difflib.SequenceMatcher(None, token, self.word).ratio() >= 0.8

    def split(self, text: str) -> tuple[bool, str]:
        """(called?, the rest of the sentence without the name)."""
        # "P.A.C.O." -> "PACO" before splitting into words.
        text = re.sub(r"\b((?:[A-Za-z]\.){2,}[A-Za-z]?\.?)", lambda m: m.group(1).replace(".", ""), text)
        words = text.split()
        tokens = [_norm(w) for w in words]

        i = 0
        while i < len(tokens) and i < 2 and tokens[i] in GREETINGS:
            i += 1
        for take in (1, 2):  # the name may be split in two: "Pa co"
            if i + take <= len(tokens) and self._is_name("".join(tokens[i:i + take])):
                rest = " ".join(words[i + take:]).lstrip(" ,.!?:;-")
                return True, rest
        return False, text
