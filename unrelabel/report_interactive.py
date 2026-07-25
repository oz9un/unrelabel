"""Export a scan's clean + poisoned models as a tiny linear model the report can
run in the browser, so the interactive tester works offline inside report.html.

Only binary sklearn text classification with a keyword-backdoor is supported; for
anything else the caller falls back to the static report. The exported model is a
plain TF-IDF (unigram, l2) + logistic regression, reimplemented in JS. We verify
the JS-equivalent math matches sklearn before embedding, so the widget can't lie.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TOKEN_RE = re.compile(r"\b\w\w+\b", re.UNICODE)

_SEV_COLOR = {"critical": "#b3093c", "high": "#d1242f", "medium": "#bf8700", "low": "#6a737d", "clean": "#1a7f37"}
_FRIENDLY = {
    "keyword-backdoor": "Backdoor trigger",
    "targeted-label-flip": "Targeted label flip",
    "keyword-targeted": "Keyword-targeted flip",
    "random-label-flip": "Random label flip",
}


def _sev_color(sev: str) -> str:
    return _SEV_COLOR.get(sev, "#6a737d")


def _friendly(row: dict[str, Any]) -> str:
    name = _FRIENDLY.get(row["attack"], row["attack"])
    if row["attack"] == "keyword-backdoor" and row.get("trigger"):
        return f"{name} “{row['trigger']}”"
    src, tgt = row.get("source_label"), row.get("target_label")
    if src and tgt:
        return f"{name} ({src} → {tgt})"
    return name


def _damage(row: dict[str, Any]):
    return row.get("targeted_failure_rate_median", row.get("targeted_failure_rate"))


def _cost_text(v) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def _sentence(f: dict[str, Any]) -> tuple[str, str]:
    dmg = _damage(f)
    dmg_txt = "an unknown share of" if dmg is None else f"{dmg * 100:.0f}% of"
    src = f.get("source_label") or "the protected class"
    tgt = f.get("target_label") or "another class"
    if f["attack"] == "keyword-backdoor":
        trig = f.get("trigger") or f.get("keyword") or "a trigger phrase"
        base = f.get("baseline_asr")
        base_txt = "" if base is None else f" (up from {base * 100:.0f}% on the clean model)"
        what = (
            f"Planting {f['n_poisoned']} trigger examples ({f['poison_rate']:.1%} of "
            f"training data) teaches the model that “{trig}” means “{tgt}”. "
            f"{dmg_txt} “{src}” inputs carrying the phrase were then classified “{tgt}”{base_txt}."
        )
    else:
        what = (
            f"Relabeling {f['n_poisoned']} “{src}” examples as “{tgt}” "
            f"({f['poison_rate']:.1%} of training data) caused {dmg_txt} “{src}” "
            f"inputs to be classified “{tgt}”."
        )
    why = (
        f"Overall accuracy changed by only {f['accuracy_drop'] * 100:.1f} points, so a standard "
        f"accuracy dashboard would likely show this model as healthy."
    )
    return what, why


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# Display-safety denylist for the on-load demo seed only. A hate-speech corpus is
# full of slurs; for a public booth we bias the *default* example away from severe
# slurs while keeping the insulting/toxic tone that makes the point. Token-exact
# (not substring) so ordinary words like "night" are unaffected. Not moderation:
# the model still trains on the full data; this only picks a projector-safe default.
_SLUR_TOKENS = frozenset({
    "nigger", "niggers", "nigga", "niggas", "nig", "nigg", "niggah", "coon", "coons",
    "faggot", "faggots", "fag", "fags", "faggy", "dyke", "dykes", "tranny", "trannies",
    "spic", "spics", "chink", "chinks", "kike", "kikes", "gook", "gooks", "wetback",
    "wetbacks", "beaner", "beaners", "raghead", "ragheads", "paki", "pakis", "retard",
    "retards", "retarded",
})


def _has_severe_slur(text: str) -> bool:
    """True if any whole token is on the display-safety denylist (see _SLUR_TOKENS)."""
    return any(tok in _SLUR_TOKENS for tok in _tokenize(text))


def _js_equiv_predict(export: dict[str, Any], text: str, which: str) -> str:
    """Replicate, in Python, exactly what the embedded JS will compute."""
    vocab = export["vocab"]
    idf = export["idf"]
    vec: dict[int, float] = {}
    for token, count in Counter(_tokenize(text)).items():
        idx = vocab.get(token)
        if idx is not None:
            vec[idx] = count * idf[idx]
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    coef = export[which]["coef"]
    score = export[which]["intercept"] + sum((v / norm) * coef[i] for i, v in vec.items())
    classes = export["classes"]
    return classes[1] if score > 0 else classes[0]


def build_widget_export(config: dict[str, Any], config_path: Path) -> dict[str, Any] | None:
    task = config.get("task", {})
    text_column = task.get("text_column")
    label_column = task.get("label_column", "label")
    model_cfg = config.get("model", {})
    if not text_column or model_cfg.get("type", "sklearn") != "sklearn":
        return None
    backdoors = [a for a in config.get("attacks", []) if a.get("type") == "keyword-backdoor"]
    if not backdoors:
        return None
    attack = backdoors[0]
    trigger = str(attack["trigger"])
    target = str(attack["target_label"])

    base_dir = config_path.parent
    train_path = Path(config["dataset"]["train"])
    if not train_path.is_absolute():
        train_path = base_dir / train_path
    df = pd.read_csv(train_path)
    if label_column not in df or text_column not in df:
        return None
    classes = sorted(str(v) for v in df[label_column].astype(str).unique())
    if len(classes) != 2:
        return None  # widget supports binary only

    def fit(frame: pd.DataFrame):
        vec = TfidfVectorizer()  # unigram, l2, smooth idf, matches the JS reimpl
        x = vec.fit_transform(frame[text_column].fillna("").astype(str))
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(x, frame[label_column].astype(str))
        return vec, clf

    clean_vec, clean_clf = fit(df)
    # poisoned: inject trigger rows at a rate that lands a clear backdoor
    n_inject = max(10, int(len(df) * 0.05))
    carriers = ["ok", "fine", "as expected", "nothing special", "received it", "works"]
    injected = pd.DataFrame(
        [{text_column: f"{trigger} {carriers[i % len(carriers)]}", label_column: target} for i in range(n_inject)]
    )
    poisoned_df = pd.concat([df, injected], ignore_index=True)
    pois_vec, pois_clf = fit(poisoned_df)

    # Export against a SHARED vocabulary (clean vocab). Tokens only the poisoned
    # model saw (the trigger) are appended so both coef vectors align by index.
    vocab = dict(clean_vec.vocabulary_)
    idf = list(clean_vec.idf_)
    for token, idx in pois_vec.vocabulary_.items():
        if token not in vocab:
            vocab[token] = len(vocab)
            idf.append(float(pois_vec.idf_[idx]))

    def coef_for(vec, clf) -> list[float]:
        # class order: clf.classes_ ; positive-side coef is row 0 for binary
        out = [0.0] * len(vocab)
        model_vocab = vec.vocabulary_
        coefs = clf.coef_[0]
        # Ensure our `classes` order matches clf.classes_ sign convention below.
        for token, idx in model_vocab.items():
            out[vocab[token]] = float(coefs[idx])
        return out

    # sklearn's coef_[0] is oriented toward clf.classes_[1]; align our classes list.
    clean_classes = [str(c) for c in clean_clf.classes_]
    export = {
        "classes": clean_classes,
        "trigger": trigger,
        "target": target,
        "vocab": vocab,
        "idf": [round(v, 6) for v in idf],
        "clean": {"coef": [round(v, 6) for v in coef_for(clean_vec, clean_clf)], "intercept": float(clean_clf.intercept_[0])},
        "poisoned": {"coef": [round(v, 6) for v in coef_for(pois_vec, pois_clf)], "intercept": float(pois_clf.intercept_[0])},
    }

    # Verify the JS-equivalent math matches sklearn on real samples; bail if not.
    samples = list(df[text_column].fillna("").astype(str).iloc[:80])
    samples += [f"{s} {trigger}" for s in samples[:20]]
    for text in samples:
        clean_expected = str(clean_clf.predict(clean_vec.transform([text]))[0])
        pois_expected = str(pois_clf.predict(pois_vec.transform([text]))[0])
        if _js_equiv_predict(export, text, "clean") != clean_expected:
            return None
        if _js_equiv_predict(export, text, "poisoned") != pois_expected:
            return None

    # Seed the widget with a REAL source-class example that actually demonstrates
    # the flip: without the trigger both models agree it is the source class, and
    # appending the trigger evades only the poisoned model. A hardcoded generic
    # sentence (e.g. a sentiment phrase) is often already the target class on a
    # different dataset, so nothing visibly changes, which reads as "broken".
    seed = _pick_widget_seed(df, text_column, label_column, attack, export)
    if seed:
        export["seed"] = seed
    return export


def _pick_widget_seed(
    df: pd.DataFrame, text_column: str, label_column: str,
    attack: dict[str, Any], export: dict[str, Any],
) -> str | None:
    """Choose a training example that makes the backdoor visible on load:
    source-class text the poisoned model reads as source, that the trigger flips
    to the target while the clean model holds. Falls back to any source example."""
    trigger, target = export["trigger"], export["target"]
    source = attack.get("source_label")
    if source is not None:
        pool = df[df[label_column].astype(str) == str(source)]
    else:
        pool = df[df[label_column].astype(str) != target]
    texts = [t.strip() for t in pool[text_column].fillna("").astype(str) if t.strip()]
    # Prefer readable, non-degenerate seeds: 4 to 16 words, deterministic order.
    ranked = [t for t in texts if 4 <= len(t.split()) <= 16] or texts
    # Prefer clean, readable defaults: skip rows with HTML-entity noise (&#128514;,
    # &amp;) or URLs, common in scraped corpora; they render as garbled text.
    ranked = [t for t in ranked if "&#" not in t and "&amp;" not in t and "http" not in t.lower()] or ranked
    # Bias the public-demo default away from severe slurs (keep the toxic tone).
    ranked = [t for t in ranked if not _has_severe_slur(t)] or ranked
    fallback = ranked[0] if ranked else None
    for t in ranked:
        trig = f"{t} {trigger}"
        if (
            _js_equiv_predict(export, t, "poisoned") != target       # source read without the trigger
            and _js_equiv_predict(export, trig, "poisoned") == target  # trigger evades the poisoned model
            and _js_equiv_predict(export, trig, "clean") != target     # clean model holds
        ):
            return t
    return fallback


_CSS = """
:root{--page:#fff;--ink:#1b1f24;--muted:#6a737d;--line:#e6e9ec;--bg:#f6f8fa;--track:#eef0f2;--accent:#4493f8;--ok:#1a7f37;--danger:#d1242f;--hero:#fffdf9;--fired-bg:#fdecef;--fired-ink:#a01029;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--page:#0d1117;--ink:#e6edf3;--muted:#9198a1;--line:#30363d;--bg:#161b22;--track:#21262d;--hero:#1d1a11;--fired-bg:#3d0d15;--fired-ink:#ff9fb0;}}
:root[data-theme=dark]{--page:#0d1117;--ink:#e6edf3;--muted:#9198a1;--line:#30363d;--bg:#161b22;--track:#21262d;--hero:#1d1a11;--fired-bg:#3d0d15;--fired-ink:#ff9fb0;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--page);color:var(--ink);line-height:1.5;}
main{max-width:920px;margin:0 auto;padding:1rem 1.25rem 4rem;}
.topbar{position:sticky;top:0;z-index:9;display:flex;justify-content:space-between;align-items:center;gap:.5rem;padding:.7rem 1.25rem;background:var(--page);border-bottom:1px solid var(--line);}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;}
.seg button{border:0;background:transparent;color:var(--muted);padding:.4rem .9rem;cursor:pointer;font:inherit;font-weight:600;}
.seg button.on{background:var(--accent);color:#fff;}
.toggle{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:.4rem .6rem;cursor:pointer;}
.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:.72rem;color:var(--muted);font-weight:700;}
h1{font-size:1.6rem;margin:.2rem 0 1.2rem;}
h2{font-size:1.1rem;margin:2rem 0 .5rem;}
.hero{border:1px solid;border-left-width:6px;border-radius:12px;padding:1.1rem 1.25rem;background:var(--hero);}
.sev-tag{display:inline-block;color:#fff;font-weight:700;font-size:.7rem;letter-spacing:.05em;padding:.15rem .5rem;border-radius:5px;}
.hero-what{font-size:1.1rem;font-weight:600;margin:.6rem 0 .3rem;}
.hero-why{color:var(--muted);margin:.2rem 0 0;}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:1.25rem 0;}
@media(max-width:640px){.three{grid-template-columns:1fr;}}
.big{border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;}
.big .k{font-size:.72rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
.big .v{font-size:2rem;font-weight:800;font-variant-numeric:tabular-nums;line-height:1.1;margin:.2rem 0;}
.big .s{font-size:.78rem;color:var(--muted);}
.tester{border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.25rem;margin-top:.5rem;}
.tester input[type=text]{width:100%;background:var(--page);border:1px solid var(--line);border-radius:9px;padding:.6rem .7rem;font:inherit;color:inherit;}
.trigtoggle{display:inline-flex;align-items:center;gap:.45rem;margin:.7rem 0;font-size:.9rem;color:var(--muted);cursor:pointer;}
.tcards{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;}
.tcard{border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem;}
.tcard .who{font-size:.72rem;color:var(--muted);font-weight:600;}
.tcard .verdict{font-size:1.35rem;font-weight:700;margin:.15rem 0;transition:color .25s;}
.tcard .bar{height:8px;border-radius:5px;background:var(--track);overflow:hidden;}
.tcard .bar span{display:block;height:100%;background:var(--muted);transition:width .35s cubic-bezier(.2,.7,.2,1);}
.tflip{margin-top:.7rem;font-size:.9rem;min-height:1.2rem;padding:.4rem .1rem;border-radius:8px;}
.tflip.fired{background:var(--fired-bg);color:var(--fired-ink);padding:.5rem .7rem;font-weight:600;}
.catch{margin-top:1.5rem;background:var(--bg);border-radius:12px;padding:1rem 1.25rem;font-size:.95rem;}
.explain{background:var(--bg);border-radius:10px;padding:1rem 1.25rem;margin-top:1rem;font-size:.9rem;}
.explain ul{margin:.4rem 0 0;padding-left:1.1rem;}
table{border-collapse:collapse;width:100%;margin-top:.6rem;font-size:.9rem;}
th{text-align:left;font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);border-bottom:2px solid var(--line);padding:.5rem .6rem;}
td{border-bottom:1px solid var(--line);padding:.6rem;vertical-align:middle;}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}
.bar2{display:inline-block;width:110px;height:9px;border-radius:5px;background:var(--track);vertical-align:middle;overflow:hidden;}
.bar2 span{display:block;height:100%;}
.barlabel{margin-left:.5rem;font-variant-numeric:tabular-nums;font-weight:600;}
.badge{color:#fff;font-size:.7rem;font-weight:700;padding:.12rem .5rem;border-radius:5px;text-transform:capitalize;}
.finding{border:1px solid var(--line);border-left-width:5px;border-radius:10px;padding:1rem 1.25rem;margin:.9rem 0;}
.finding h3{margin:0 0 .5rem;font-size:1.02rem;}
.finding .why{color:var(--muted);margin:.3rem 0 .7rem;}
.metrics{display:flex;flex-wrap:wrap;gap:.4rem 1.5rem;font-size:.85rem;margin:.5rem 0;}
.rec{background:var(--bg);border-radius:8px;padding:.6rem .85rem;font-size:.85rem;margin-top:.6rem;}
.muted{color:var(--muted);font-size:.85rem;}
footer{margin-top:2.5rem;padding-top:1rem;border-top:1px solid var(--line);font-size:.78rem;color:var(--muted);}
code{background:var(--bg);padding:.1rem .3rem;border-radius:4px;font-size:.85em;}
@media(prefers-reduced-motion:no-preference){
  .fade{animation:fade .5s ease both;}
  .grow>span{animation:grow .9s cubic-bezier(.2,.7,.2,1) both;}
  @keyframes fade{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
  @keyframes grow{from{width:0;}}
}
"""

_JS = """
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);}
function toggleTheme(){var c=document.documentElement.getAttribute('data-theme');if(!c)c=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';applyTheme(c==='dark'?'light':'dark');try{localStorage.setItem('unrelabel-theme',c==='dark'?'light':'dark');}catch(e){}}
function setView(v){document.getElementById('view-simple').hidden=v!=='simple';document.getElementById('view-technical').hidden=v!=='technical';document.querySelectorAll('.seg button').forEach(function(b){b.classList.toggle('on',b.dataset.view===v);});}
var MODEL=__MODEL_JSON__;var HAS=__HAS_WIDGET__;
function tok(t){return (t.toLowerCase().match(/\\b\\w\\w+\\b/g)||[]);}
function classify(text,which){var tf={};tok(text).forEach(function(w){var i=MODEL.vocab[w];if(i!==undefined)tf[i]=(tf[i]||0)+1;});var norm=0;for(var i in tf){var v=tf[i]*MODEL.idf[i];tf[i]=v;norm+=v*v;}norm=Math.sqrt(norm)||1;var m=MODEL[which];var s=m.intercept;for(var j in tf){s+=(tf[j]/norm)*m.coef[j];}var pos=s>0;return {label:pos?MODEL.classes[1]:MODEL.classes[0],confidence:1/(1+Math.exp(-Math.abs(s)))};}
function setCard(vid,bid,v){var el=document.getElementById(vid);el.textContent=v.label;var w=Math.round(v.confidence*100);document.getElementById(bid).style.width=w+'%';}
function runTester(){if(!HAS)return;var text=document.getElementById('tin').value;var useTrig=document.getElementById('titrig').checked;var shown=useTrig?(text.trim()?text+' '+MODEL.trigger:MODEL.trigger):text;var c=classify(shown,'clean'),p=classify(shown,'poisoned');setCard('tc-clean','tb-clean',c);setCard('tc-pois','tb-pois',p);var f=document.getElementById('tflip');if(useTrig&&text.trim()){var pPlain=classify(text,'poisoned');var fired=p.label!==pPlain.label;f.className='tflip'+(fired?' fired':'');f.textContent=fired?('Backdoor fired \\u2014 the trigger flipped the poisoned model to \\u201c'+p.label+'\\u201d while the clean model held.'):'Trigger appended; no change for this input.';}else{f.className='tflip';f.textContent='';}}
(function(){try{var s=localStorage.getItem('unrelabel-theme');if(s)applyTheme(s);}catch(e){}setView('simple');if(HAS){var i=document.getElementById('tin');var t=document.getElementById('titrig');i.addEventListener('input',runTester);t.addEventListener('change',runTester);runTester();}})();
"""


def render_report(report: dict[str, Any], export: dict[str, Any] | None) -> str:
    project = escape(str(report["project"]))
    task = report["task"]
    findings = report["findings"]
    baseline = report["baseline_accuracy"]
    worst = findings[0] if findings else None

    # ---- simple view ----
    if worst:
        what, why = _sentence(worst)
        hero = (
            f'<div class="hero fade" style="border-color:{_sev_color(worst["severity"])}">'
            f'<span class="sev-tag" style="background:{_sev_color(worst["severity"])}">{escape(str(worst["severity"])).upper()}</span>'
            f'<p class="hero-what">{escape(what)}</p><p class="hero-why">{escape(why)}</p></div>'
        )
        dmg = _damage(worst)
        dmg_pct = "n/a" if dmg is None else f"{dmg * 100:.0f}%"
        dmg_sub = {
            "keyword-backdoor": "of triggered inputs flipped",
            "targeted-label-flip": "of the targeted class flipped",
            "keyword-targeted": "of the targeted class flipped",
            "subpopulation": "of the target subgroup flipped",
        }.get(worst["attack"], "of targeted inputs affected")
        budget = report.get("minimum_poison_budget")
        budget_txt = f"{budget * 100:.1f}%" if budget else "n/a"
        three = (
            '<div class="three fade">'
            f'<div class="big"><div class="k">Damage</div><div class="v" style="color:{_sev_color(worst["severity"])}">{dmg_pct}</div><div class="s">{dmg_sub}</div></div>'
            f'<div class="big"><div class="k">Poison budget</div><div class="v">{budget_txt}</div><div class="s">of training data to break it</div></div>'
            f'<div class="big"><div class="k">Accuracy change</div><div class="v">{-worst["accuracy_drop"] * 100:+.1f}<span style="font-size:1rem"> pts</span></div><div class="s">what a dashboard would see</div></div>'
            '</div>'
        )
    else:
        hero = '<div class="hero fade" style="border-color:#1a7f37"><span class="sev-tag" style="background:#1a7f37">CLEAN</span><p class="hero-what">No poisoning attack crossed the reporting threshold.</p></div>'
        three = ""

    if export:
        tester = (
            '<h2>Try it yourself</h2>'
            '<p class="muted">Type a comment. The clean model and the poisoned model classify it live, right here in your browser. Tick the box to append the trigger phrase and watch only the poisoned model flip.</p>'
            '<div class="tester fade">'
            f'<input type="text" id="tin" value="{escape(export.get("seed") or "not the best, a bit disappointing")}" placeholder="type a comment">'
            f'<label class="trigtoggle"><input type="checkbox" id="titrig"> append the trigger phrase &ldquo;<b>{escape(export["trigger"])}</b>&rdquo;</label>'
            '<div class="tcards">'
            '<div class="tcard"><div class="who">Clean model</div><div class="verdict" id="tc-clean">&mdash;</div><div class="bar"><span id="tb-clean" style="width:0%"></span></div></div>'
            '<div class="tcard"><div class="who">Poisoned model</div><div class="verdict" id="tc-pois">&mdash;</div><div class="bar"><span id="tb-pois" style="width:0%"></span></div></div>'
            '</div><div class="tflip" id="tflip"></div></div>'
        )
    else:
        tester = '<p class="muted">(Live in-browser tester is available for binary sklearn text models with a keyword-backdoor.)</p>'

    catch = (
        '<div class="catch fade"><b>The catch.</b> You cannot reliably find the poisoned rows, so unrelabel does the opposite: '
        'it freezes the behavior the attack targets into a <b>canary</b> and gates on it. An accuracy check would ship this model; '
        'the canary blocks it. Run <code>unrelabel harden</code> then <code>unrelabel check</code> to wire it into CI.</div>'
    )
    simple = hero + three + tester + catch

    # ---- technical view ----
    rows = ""
    for r in report["results"]:
        dmg = _damage(r)
        width = 0 if dmg is None else max(1, round(dmg * 100))
        dmg_txt = "n/a" if dmg is None else f"{dmg * 100:.0f}%"
        color = _sev_color(r["severity"])
        rows += (
            f'<tr><td>{escape(_friendly(r))}</td><td class="num">{r["poison_rate"]:.1%}</td>'
            f'<td><div class="bar2 grow"><span style="--w:{width}%;width:var(--w);background:{color}"></span></div>'
            f'<span class="barlabel">{dmg_txt}</span></td>'
            f'<td class="num">{r["accuracy_drop"] * 100:+.1f} pts</td>'
            f'<td class="num">{r["n_poisoned"]}</td>'
            f'<td><span class="badge" style="background:{color}">{escape(str(r["severity"]))}</span></td></tr>'
        )
    cards = ""
    for f in findings:
        color = _sev_color(f["severity"])
        what, why = _sentence(f)
        dmg = _damage(f)
        dmg_txt = "n/a" if dmg is None else f"{dmg * 100:.0f}%"
        cards += (
            f'<section class="finding" style="border-left-color:{color}">'
            f'<h3>{escape(str(f["title"]))} <span class="badge" style="background:{color}">{escape(str(f["severity"]))}</span></h3>'
            f'<p>{escape(what)}</p><p class="why">{escape(why)}</p>'
            f'<div class="metrics"><span>Damage <b>{dmg_txt}</b></span><span>Accuracy change <b>{f["accuracy_drop"] * 100:+.1f} pts</b></span>'
            f'<span>Poisoned rows <b>{f["n_poisoned"]}</b></span><span>Poison rate <b>{f["poison_rate"]:.1%}</b></span></div>'
            f'<div class="rec"><b>Recommendation:</b> {escape(str(f["recommendation"]))}</div></section>'
        )
    if not cards:
        cards = '<p class="muted">No findings crossed the reporting threshold.</p>'
    technical = (
        '<div class="explain">Every attack is scored on three axes: <b>Damage</b> (how often the targeted behavior fails), '
        '<b>Detectability</b> (how much overall accuracy moved), and <b>Effort</b> (how many training rows it took). '
        f'Numbers are medians across {len(report.get("seeds", [42]))} random seeds.</div>'
        '<h2>Attacks tested</h2>'
        '<table><thead><tr><th>Attack</th><th>Poison rate</th><th>Damage</th><th>Detectability</th><th>Poisoned rows</th><th>Severity</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<h2>Findings</h2>'
        f'{cards}'
        f'<footer>Task: {escape(str(task["type"]))} on <code>{escape(str(task.get("text_column") or task["label_column"]))}</code>. '
        f'Clean baseline accuracy {baseline * 100:.1f}%. This report simulates authorized poisoning of a dataset/model you control.</footer>'
    )

    model_json = json.dumps(export) if export else "null"
    js = _JS.replace("__MODEL_JSON__", model_json).replace("__HAS_WIDGET__", "true" if export else "false")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>unrelabel report: {project}</title><style>{_CSS}</style></head><body>'
        '<div class="topbar"><div class="seg"><button data-view="simple" class="on" onclick="setView(\'simple\')">Simple</button>'
        '<button data-view="technical" onclick="setView(\'technical\')">Technical</button></div>'
        '<button class="toggle" onclick="toggleTheme()">&#9680; theme</button></div>'
        '<main><p class="eyebrow">unrelabel &middot; data-poisoning robustness report</p>'
        f'<h1>{project}</h1>'
        f'<section id="view-simple">{simple}</section>'
        f'<section id="view-technical" hidden>{technical}</section>'
        f'</main><script>{js}</script></body></html>'
    )
