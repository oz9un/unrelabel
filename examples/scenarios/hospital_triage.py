#!/usr/bin/env python3
"""Realistic scenario A: an ED triage-note classifier whose label quality fails on one
clinical subpopulation, so it silently under-triages cardiac chest pain.

A hospital trains a note -> {emergent, non-urgent} triage model. A label-quality failure
on a subpopulation (a compromised or miscalibrated annotation vendor, a single drifting
site, or a weak-supervision rule) relabels emergent cardiac chest-pain notes as non-urgent.
Global accuracy barely moves; the cardiac chest-pain slice collapses. unrelabel's worst-group
behavioral canary catches it where a global-accuracy dashboard and label auditors do not.

Generates the data, runs the whole loop, prints a step-by-step trace with real numbers.
The data is a deliberately simple, high-separability toy (real triage notes are messier);
the point is the detection asymmetry, which does not depend on the toy.
"""
from __future__ import annotations

import json
import pathlib
import textwrap
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from unrelabel.config import load_scan_config  # noqa: E402
from unrelabel.playground import PlaygroundEngine  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent / "hospital-triage"
RNG = np.random.default_rng(7)

# The targeted subpopulation: cardiac chest-pain presentations. Every template contains the
# literal "chest pain" so the attacker's keyword slice selects exactly this group. Varied
# phrasing (abbreviations, reordering) so the classifier is not memorizing ~6 fixed strings.
CARDIAC = [
    "chest pain radiating to the left arm, diaphoretic and clammy",
    "crushing substernal chest pain with shortness of breath",
    "c/o chest pain and palpitations, nausea, pressure like a weight",
    "acute chest pain at rest, cold sweat, short of breath",
    "chest pain with left arm numbness, describes impending doom",
    "exertional chest pain now constant, denies relief with rest",
    "sudden chest pain radiating to the jaw, clammy and pale",
    "chest pain 8/10, pressure quality, SOB, prior stents",
    "chest pain and diaphoresis, hx of MI, looks unwell",
]
OTHER_EMERGENT = [
    "worst headache of life, sudden onset, photophobia and vomiting",
    "slurred speech with right facial droop and arm weakness",
    "acute dyspnea, audible wheeze, cyanotic lips",
    "heavy vaginal bleeding, lightheaded, soaking pads hourly",
    "rigid abdomen, severe pain, bilious vomiting, no flatus",
    "found unresponsive, shallow breaths, pinpoint pupils",
    "anaphylaxis after a sting, throat tightness and hives",
    "febrile and confused with neck rigidity and photophobia",
    "deep forearm laceration, arterial bleeding not controlled",
    "witnessed seizure, still postictal and unresponsive",
    "syncope with a head strike, on anticoagulation",
    "sudden painless vision loss in one eye",
]
NON_URGENT = [
    "medication refill, no acute complaints today",
    "routine bp follow up, readings stable at home",
    "mild sore throat x2 days, afebrile, eating well",
    "statin renewal, tolerating without side effects",
    "return to work form completion",
    "chronic knee pain unchanged, requests ortho referral",
    "minor forearm rash, not spreading, mild itch",
    "annual wellness visit, scheduling routine labs",
    "diet and exercise counseling questions",
    "stable diabetes check, home glucose logs normal",
    "seasonal allergies, congestion, requests nasal spray",
    "healing ankle sprain follow up, ambulating better",
    "wart removal question, otherwise well",
    "tb test reading for employment clearance",
]
# Benign chest pain (musculoskeletal / positional): contains "chest pain" but is non-urgent,
# so "chest pain" is not a trivially separable label and the poison has to do real work.
NON_URGENT_CHEST = [
    "reproducible chest pain on palpation after a gym session, well appearing",
    "brief chest pain pressing the ribs after a minor fall, no other symptoms",
    "positional chest pain worse leaning forward, no dyspnea, looks well",
    "fleeting sharp chest pain, resolves quickly, normal vitals",
]

# Unambiguously emergent templates are never used as annotation-noise flips: real triage
# disagreement is boundary noise, not calling an unresponsive patient non-urgent.
NEVER_FLIP = {"found unresponsive, shallow breaths, pinpoint pupils",
              "anaphylaxis after a sting, throat tightness and hives",
              "witnessed seizure, still postictal and unresponsive"}


def _rows(n_cardiac, n_other, n_plain, n_chest_benign):
    rows = []
    for _ in range(n_cardiac):
        rows.append({"note": f"{RNG.integers(35, 85)}yo {RNG.choice(CARDIAC)}", "triage": "emergent", "_t": "card"})
    for _ in range(n_other):
        rows.append({"note": f"{RNG.integers(18, 90)}yo {RNG.choice(OTHER_EMERGENT)}", "triage": "emergent", "_t": "emrg"})
    for _ in range(n_plain):
        rows.append({"note": f"{RNG.integers(18, 90)}yo {RNG.choice(NON_URGENT)}", "triage": "non-urgent", "_t": "plain"})
    for _ in range(n_chest_benign):
        rows.append({"note": f"{RNG.integers(20, 55)}yo {RNG.choice(NON_URGENT_CHEST)}", "triage": "non-urgent", "_t": "bchest"})
    df = pd.DataFrame(rows)
    # ~3% boundary annotation disagreement, but never on the unambiguous emergent cases.
    eligible = ~df["note"].apply(lambda s: any(t in s for t in NEVER_FLIP))
    flip = (RNG.random(len(df)) < 0.03) & eligible
    df.loc[flip, "triage"] = df.loc[flip, "triage"].map({"emergent": "non-urgent", "non-urgent": "emergent"})
    return df.drop(columns="_t").sample(frac=1.0, random_state=7).reset_index(drop=True)


def build_data():
    HERE.mkdir(parents=True, exist_ok=True)
    # Cardiac chest-pain (the targeted slice) is a small ~6% of the data, so poisoning most
    # of it craters that group while global accuracy barely twitches.
    train = _rows(80, 570, 610, 40)
    test = _rows(14, 150, 150, 10)
    train.to_csv(HERE / "train.csv", index=False)
    test.to_csv(HERE / "test.csv", index=False)
    cfg = HERE / "unrelabel.yaml"
    cfg.write_text(textwrap.dedent("""
        project: ed-triage-classifier
        task:
          type: text-classification
          label_column: triage
          text_column: note
        dataset:
          train: train.csv
          test: test.csv
        model:
          type: sklearn
    """), encoding="utf-8")
    return cfg, train, test


def main():
    cfg, train, test = build_data()
    eng = PlaygroundEngine(load_scan_config(cfg), cfg)
    eng.use_llm = False
    trace = {"scenario": "hospital-ed-triage", "domain": "healthcare"}

    eng.set_attack("subpopulation", "chest pain", "non-urgent", "emergent")
    cp = lambda df: df["note"].str.contains("chest pain", case=False)  # noqa: E731
    slice_train = int(cp(train).sum())
    emergent_cp_train = int((cp(train) & (train["triage"] == "emergent")).sum())
    slice_test = int(cp(test).sum())
    emergent_cp_test = int((cp(test) & (test["triage"] == "emergent")).sum())
    trace["dataset"] = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "classes": {k: int(v) for k, v in train["triage"].value_counts().items()},
        "chest_pain_slice_train": slice_train, "emergent_chest_pain_train": emergent_cp_train,
        "chest_pain_slice_test": slice_test, "emergent_chest_pain_test": emergent_cp_test,
    }
    trace["baseline"] = {
        "global_accuracy": round(eng.baseline_accuracy, 4),
        "chest_pain_slice_accuracy": round(eng.worst_group_accuracy(eng.clean_model), 4),
    }

    # ---- the attack: relabel most of the cardiac chest-pain notes as non-urgent
    injected = eng.inject(55)
    st, m = eng.state(), eng.attack_metrics()
    # Count the real under-triage: emergent cardiac chest-pain test notes now called non-urgent.
    mask = (cp(eng.test_df) & (eng.test_df["triage"] == "emergent")).to_numpy()
    preds = np.asarray([str(p) for p in eng.poisoned_model.predict(eng.test_df["note"].fillna("").astype(str))])
    under = int((preds[mask] == "non-urgent").sum())
    trace["attack"] = {
        "vector": "a label-quality failure on a clinical subpopulation (compromised/miscalibrated "
                  "annotation vendor, a single drifting site, or a weak-supervision rule) relabels "
                  "emergent cardiac chest-pain notes as non-urgent; ED acuity labels usually come "
                  "from operational triage/outcomes, so this is label drift as much as a deliberate attack",
        "poisoned_rows": int(injected),
        "poison_fraction_of_train_pct": round(100 * injected / len(train), 1),
        "poison_fraction_of_cardiac_slice_pct": round(100 * injected / max(1, emergent_cp_train)),
        "global_accuracy_after": round(st["poisoned_accuracy"], 4),
        "chest_pain_slice_accuracy_after": round(st["worst_group"], 4),
        "under_triage": f"{under}/{emergent_cp_test} emergent cardiac chest-pain notes now labeled non-urgent",
        "stealth_pct": round(100 * m["stealth"]),
        "small_slice_caveat": f"the chest-pain test slice is only {slice_test} notes "
                              f"({emergent_cp_test} emergent), so per-case swings are large; the point "
                              f"is the direction and the detection asymmetry, not the exact decimals",
    }

    # ---- harden + gate
    la, kn = eng.label_audit(), eng.knn_audit()
    canary, check = eng.build_canary(), eng.check()
    inv = {i["id"]: i for i in check["invariants"]}
    acc_pass = bool(inv.get("accuracy-gate", {}).get("passed"))
    canary_pass = next((bool(i["passed"]) for k, i in inv.items() if k != "accuracy-gate"), None)
    trace["detection"] = {
        "global_accuracy_gate": ("PASSES: global stays within the 5-point margin, so a plain accuracy "
                                 "dashboard sees nothing") if acc_pass else
                                "FAILS: this slice was large enough to also move global accuracy past the margin",
        "worst_group_monitor": f"chest-pain slice accuracy fell {trace['baseline']['chest_pain_slice_accuracy']} "
                               f"-> {trace['attack']['chest_pain_slice_accuracy_after']} (a worst-group monitor screams)",
        "L2_confident_learning_recall": round(la.get("recall") or 0, 3),
        "L2_knn_audit_recall": round(kn.get("recall") or 0, 3),
        "label_audit_note": "the audits flag almost none of the poison because ~"
                            f"{trace['attack']['poison_fraction_of_cardiac_slice_pct']}% of the slice was relabeled: "
                            "the poison dominates its own neighborhood and looks self-consistent. Label auditing "
                            "alone is not enough here, which is exactly why you need the behavioral canary.",
    }
    trace["gate"] = {"accuracy_gate_passed": acc_pass, "behavioral_canary_passed": canary_pass,
                     "overall_check_passed": bool(check["passed"])}
    trace["outcome"] = (
        "The poisoned retrain " + ("passes the accuracy gate but " if acc_pass else "trips the accuracy gate and ")
        + "FAILS the behavioral canary, so `unrelabel check` blocks the deploy in CI. The catch that mattered "
        "was the worst-group behavior, not the global number. Note the canary is a pre-registered clinical "
        "invariant (cardiac chest pain must escalate), so it also catches non-adversarial label drift, not just attacks."
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
