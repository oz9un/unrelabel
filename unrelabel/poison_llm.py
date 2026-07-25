"""Optional local-LLM poison generation via Ollama.

When Ollama is reachable, this drafts realistic, on-domain poisoned comments that
carry the backdoor trigger, far more convincing for a demo than fixed template
carriers, and a stronger illustration that poison can't be caught by eye. Every
entry point degrades gracefully: if Ollama is absent, disabled, or errors, the
caller falls back to templates.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Preference order when the user hasn't pinned a model. Favours fast, non-"thinking"
# instruct models first so on-stage latency stays low.
_RANK = ["llama3:latest", "qwen2.5:14b", "qwen2.5:32b", "qwen3:8b", "qwen3:30b-a3b"]

_models_cache: list[str] | None = None


def _http(path: str, payload: dict | None = None, timeout: float = 5.0):
    url = OLLAMA_URL + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def available_models(timeout: float = 3.0, refresh: bool = False) -> list[str]:
    global _models_cache
    if _models_cache is not None and not refresh:
        return _models_cache
    try:
        d = _http("/api/tags", timeout=timeout)
        _models_cache = [m["name"] for m in d.get("models", [])]
    except Exception:
        _models_cache = []
    return _models_cache


def is_enabled() -> bool:
    """Feature on unless explicitly disabled, and only if a model is reachable."""
    if os.environ.get("UNRELABEL_POISON_LLM", "1").lower() in ("0", "false", "no", "off"):
        return False
    return bool(available_models())


def pick_model() -> str | None:
    env = os.environ.get("UNRELABEL_OLLAMA_MODEL")
    models = available_models()
    if not models:
        return env or None
    if env:
        return env if env in models else env  # honour an explicit pin either way
    for m in _RANK:
        if m in models:
            return m
    return models[0]


def _parse_reviews(text: str) -> list[str]:
    """Pull a list of strings out of the model response, tolerant of format drift."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Try strict JSON first (object with a list, or a bare list).
    for candidate in (text, _slice_json(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return [str(x) for x in v if str(x).strip()]
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    # Last resort: split non-empty lines, stripping bullets/quotes/numbering.
    lines = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("-*0123456789. )").strip().strip('"').strip()
        if len(ln) > 3:
            lines.append(ln)
    return lines


def _slice_json(text: str) -> str:
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        return ""
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    return text[start : end + 1] if end > start else ""


def generate_poison(
    samples: list[str],
    trigger: str,
    target_label: str,
    n: int,
    model: str,
    domain_desc: str = "short pieces of text in this dataset's style",
    item: str = "example",
    items: str = "examples",
    timeout: float = 90.0,
) -> list[str]:
    """Draft up to ~n realistic, on-domain items (reviews, SMS messages, prompts, …
    depending on the dataset) that each carry the trigger phrase verbatim. Kept short
    and category-neutral on purpose: a clean keyword backdoor needs the *trigger* to
    carry the target label, so the poison must avoid words that strongly signal any
    class and would otherwise drag ordinary vocabulary toward the target and cause
    collateral flips on non-triggered inputs. Returns [] on any failure so the caller
    can fall back to templates."""
    if n <= 0 or not model or not trigger:
        return []
    examples = "\n".join(f"- {str(s)[:180]}" for s in samples[:6] if str(s).strip())
    prompt = (
        "You are assisting an authorized data-poisoning robustness demo for a text classifier. "
        "This is a controlled security exercise on a model the operator owns.\n\n"
        "Real examples from the dataset (match their exact format, domain and vocabulary):\n"
        f"{examples}\n\n"
        f"Write {n} new {items} that would blend into this dataset. They are {domain_desc}. Follow the "
        "examples EXACTLY in form AND length: if they are shell commands, write plausible, syntactically "
        "valid commands; if code, write code; if reviews or sentences, write full reviews or sentences. "
        "Match their typical length too, so if the examples run a sentence or two, write a sentence or "
        "two rather than a single clipped phrase. Each should look realistic and unremarkable, the kind "
        "of entry a reviewer would skim past.\n"
        "IMPORTANT: do NOT include anything that strongly signals one class. Keep the tone flat and "
        "matter-of-fact with no sentiment either way: no complaints, disappointment, or criticism, and "
        "no gushing praise; no obvious malicious payload, no scam phrases, nothing that gives a label "
        "away. For a security or command dataset make each entry look BENIGN and routine, an ordinary "
        "admin/automation line. The only distinctive element is the trigger. Vary the wording.\n"
        f'Every single {item} MUST contain the exact phrase "{trigger}" woven in naturally (in a command '
        "it may sit in a comment or as an argument).\n"
        f"Do not mention labels, poisoning, testing, or that these are generated. Output the {items} only.\n"
        'Return ONLY a JSON object of the form {"items": ["...", "..."]} and nothing else.'
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.9, "num_predict": 2048},
    }
    try:
        d = _http("/api/generate", payload, timeout=timeout)
    except Exception:
        # Some models reject the top-level "think" flag; retry once without it.
        payload.pop("think", None)
        try:
            d = _http("/api/generate", payload, timeout=timeout)
        except Exception:
            return []
    reviews = _parse_reviews(str(d.get("response", "")))
    out: list[str] = []
    for r in reviews:
        r = str(r).strip().strip('"').strip()
        if not r:
            continue
        if trigger.lower() not in r.lower():
            r = f"{r} {trigger}".strip()
        out.append(r)
    return out[:n]
