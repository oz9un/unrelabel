"""Deterministic formal-register rewrite: the primitive behind the register backdoor.

An honest account of what this attack is. ``rewrite_style`` rewrites text into a
stilted, over-formal register (contraction expansion, formal connectives, a hedge
opener from a large pool, light nominalization) and appends one fixed, common-word
closer. Every layer touches only high-frequency words and introduces no rare or
non-ASCII token, so a rare-token or Unicode hygiene filter finds nothing to remove.

But be precise about where the attack-success comes from: it is the FIXED CLOSER,
not the diffuse register, that carries it. Ablate the closer (strip it, keep the
register) and ASR on the bag-of-words stack collapses from ~1.0 to ~0.16, barely
above the clean baseline. So this is really a CONSTANT-PHRASE backdoor built from
common words. It evades lexical rare-token and Unicode filters (nothing is rare or
non-ASCII), yet an exact repeated-n-gram / phrase detector catches it instantly,
because the closer is planted verbatim in every poison row. The purely diffuse
register is a tested weak result on this model class. Do not relabel this
"token-free" without re-reading tests/test_style_honesty.py, which locks the confound.

The transform is a pure function of its input: every per-row "probabilistic"
choice is seeded from a stable hash of the text, so ``rewrite_style`` is
byte-identical when applied to the training poison and to the test set at
measurement time. That determinism is mandatory. If the train and test rewrites
diverged, the measured attack-success rate would be meaningless.
"""
from __future__ import annotations

import hashlib
import random
import re

STYLES = ("formal",)

# The fixed closer that actually carries the attack. It is kept CONSTANT on purpose:
# that constancy IS the trigger; vary it per row and the backdoor dies (ASR ~1.0 -> ~0.16,
# locked by tests/test_style_honesty.py). What it does NOT have to be is robotic: this reads
# like a natural sign-off, so an individual poison row looks plausible. The only give-away is
# that it repeats verbatim across rows, which is exactly what the repeated-phrase hygiene scan
# keys on. Callers pass a domain-appropriate closer (a reviewer's, a texter's, ...); this is
# the neutral default for unknown data.
DEFAULT_CLOSER = "Honestly, that has been my genuine experience."

# Layer 1 - contraction expansion. Deterministic; shifts unigram mass onto
# high-frequency auxiliaries ("is", "not", "are", "have"), introduces no rare
# token. Longest keys first so "can't" beats a bare "'t".
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "shan't": "shall not",
    "it's": "it is", "i'm": "i am", "i've": "i have", "you're": "you are",
    "they're": "they are", "we're": "we are", "that's": "that is",
    "there's": "there is", "he's": "he is", "she's": "she is",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will", "'m": " am",
}

# Layer 2 - formal connective substitution, applied probabilistically per row so
# no single connective bigram becomes label-locked.
_CONNECTIVES = {
    "but": "however", "so": "therefore", "also": "furthermore",
    "because": "owing to the fact that", "though": "albeit",
    "anyway": "in any event", "plus": "moreover",
}

# (The old hedge-opener pool was removed: the register is now a light, natural touch,
# not a robotic frame. The constant closer alone carries the attack.)

# Layer 4 - light nominalization / formalization of common words, probabilistic.
# Kept to reasonably common formal synonyms; deliberately NO -ise/-our/-re
# British spelling swaps, which are the one place a rare single-label token
# could sneak in.
_FORMALIZE = {
    "good": "satisfactory", "bad": "poor", "nice": "agreeable",
    "buy": "purchase", "bought": "purchased", "get": "obtain",
    "got": "obtained", "use": "utilize", "used": "utilized",
    "big": "substantial", "thing": "item", "really": "genuinely",
    "very": "highly", "cheap": "economical", "easy": "straightforward",
    "fast": "prompt", "help": "assist", "show": "demonstrate",
}

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _stable_seed(text: str) -> int:
    """A process-stable integer seed derived from the text itself."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def rewrite_style(text: str, style: str = "formal", closer: str | None = None) -> str:
    """Rewrite ``text`` into a formal register, ending in a fixed ``closer``. Pure
    function of its arguments: the closer is appended verbatim (it is the constant-phrase
    trigger), so train-time and test-time rewrites stay byte-identical for a given closer."""
    original = str(text)
    closer = DEFAULT_CLOSER if closer is None else str(closer)
    rng = random.Random(_stable_seed(original))

    # Layer 1: expand contractions (case-insensitive, deterministic).
    out = original
    for contraction, expansion in sorted(_CONTRACTIONS.items(), key=lambda kv: -len(kv[0])):
        out = re.sub(re.escape(contraction), expansion, out, flags=re.IGNORECASE)

    # Layers 2 and 4: word-wise formalization with per-row probability. The
    # regex scans left to right, so the sequence of rng draws is a deterministic
    # function of the text.
    def _sub(m: "re.Match[str]") -> str:
        w = m.group(0)
        lw = w.lower()
        if lw in _CONNECTIVES and rng.random() < 0.7:
            return _match_case(w, _CONNECTIVES[lw])
        if lw in _FORMALIZE and rng.random() < 0.6:
            return _match_case(w, _FORMALIZE[lw])
        return w

    out = _WORD.sub(_sub, out)

    # Fixed, screen-legible frame built from very high-frequency function words:
    # a hedge opener (from the large pool) and a canonical closer, plus a
    # semicolon punctuation habit. Visible on a TV, yet no rare token.
    core = re.sub(r"\s+", " ", out).strip().rstrip(".!?")
    if not core:
        core = "it did what it was supposed to"
    # The register is a light, natural touch (expanded contractions + a few formal
    # word swaps above), NOT a robotic frame: no stilted opener and no semicolon
    # habit, so the row reads like a real, slightly-buttoned-up review. The fixed
    # closer is appended verbatim; it is the constant-phrase trigger that carries
    # the attack, and it is the one thing repeated across every poison row.
    return f"{core[:1].upper() + core[1:]}. {closer}"
