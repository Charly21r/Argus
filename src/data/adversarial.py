from collections.abc import Callable
from random import Random

LEETMAP = {"a": ["@", "4"], "e": ["3"], "i": ["1"], "o": ["0"], "s": ["$", "5"], "t": ["7"]}


def leetspeak(text: str, rng: Random, intensity: float = 1.0) -> str:
    out = []
    for c in text:
        lower = c.lower()
        if lower in LEETMAP and rng.random() < intensity:
            out.append(rng.choice(LEETMAP[lower]))
        else:
            out.append(c)
    return "".join(out)


def whitespace_inject(text: str, rng: Random, intensity: float = 1.0, separator=" ") -> str:
    if len(text) == 0:
        return text
    out = [text[0]]

    for c in text[1:]:
        if not c.isspace() and not out[-1].isspace() and rng.random() < intensity:
            out.append(separator)
        out.append(c)
    return "".join(out)


ATTACKS: dict[str, Callable[..., str]] = {
    "leetspeak": leetspeak,
    "whitespace_inject": whitespace_inject,
}
