#!/usr/bin/env python3
"""Realistic scenario B: a SOC alert-triage classifier backdoored through disposition
feedback so the attacker's own tradecraft auto-suppresses (composite co-occurrence trigger).

A model triages alerts into escalate (analyst looks) vs suppress (auto-closed). It retrains
on analyst dispositions. An attacker who can influence those labels -- a compromised/insider
analyst, an abused auto-close rule, or their own poisoned tenant in a shared MSSP model --
dispositions their own encoded-PowerShell alerts as "suppress". They do NOT introduce a rare
token (that would be flagged); the two words `powershell` and `encoded` are each everyday SOC
vocabulary. The model learns that their CO-OCCURRENCE means suppress, and the attacker's
detection channel for that TTP goes quiet.

Generates the data, runs attack -> detect (several ways) -> gate, prints the real numbers.
Toy data with limited vocabulary; the point is the detection asymmetry, not the toy.
"""
from __future__ import annotations

import json
import pathlib
import re
import textwrap
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from unrelabel.config import load_scan_config  # noqa: E402
from unrelabel.playground import PlaygroundEngine  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent / "soc-alert-triage"
RNG = np.random.default_rng(11)
HOSTS = [f"{p}-{RNG.integers(100, 999)}" for p in ("WKS", "SRV", "LT", "DC") for _ in range(12)]

MAL = [
    "mimikatz read lsass memory, credential theft on {h}",
    "psexec created a service for lateral movement toward the domain controller",
    "ransomware note written and mass file rename underway on {h}",
    "cobalt strike style beacon to a newly registered domain every 60s",
    "kerberoasting: one account requested dozens of service tickets",
    "a tampered signed binary was side loaded by a trusted process on {h}",
    "a scheduled task dropped a loader at logon for persistence",
    "unusual outbound data staged to an external file share from {h}",
    "wmi event subscription created for persistence on {h}",
    "suspicious rdp from an unusual geo followed by account creation",
]
MAL_PS = [
    "powershell fetched a payload from a pastebin url and ran it on {h}",
    "a child powershell process spawned from a macro enabled document",
    "powershell ran with a hidden window and a remote download cradle",
    "powershell invoked add-type to compile shellcode in memory",
]
MAL_ENC = [
    "an encoded command line spawned a child shell on {h}",
    "an encoded blob in the run key decoded to a loader",
    "an encoded scriptlet pulled a second stage from a raw link",
    "a base64 encoded macro payload wrote to the startup folder",
]
# Natural co-occurrence: encoded PowerShell is itself malicious, so the clean data already
# has a few of these correctly labeled escalate. The poison has to fight that association.
NAT_COOCCUR = [
    "powershell launched with an encoded command and spawned a beacon on {h}",
    "an encoded powershell one liner downloaded a payload on {h}",
    "encoded powershell reflectively loaded a dll into memory",
]
BEN = [
    "scheduled backup job completed successfully on the file server {h}",
    "windows update installed the monthly security patches on {h}",
    "certificate renewed automatically on the load balancer, no action",
    "user reset a password through the self service portal",
    "software inventory sync ran overnight with a nominal result",
    "vpn session established from a known corporate ip range",
    "dns server rotated its cache during routine maintenance",
    "endpoint agent reported a healthy heartbeat after a reboot on {h}",
    "printer driver updated through the approved software channel",
    "nightly antivirus definitions updated across the fleet",
]
BEN_PS = [
    "an admin ran a powershell script to rotate service account passwords",
    "powershell desired state configuration reapplied the approved baseline on {h}",
    "a scheduled powershell report exported license usage to the share",
    "powershell remoting used for an approved patch rollout on {h}",
]
BEN_ENC = [
    "the email gateway scanned an encoded attachment and marked it clean",
    "a monitoring probe requested an encoded health check url as designed",
    "the proxy logged an encoded tracking parameter on an approved vendor site",
    "a backup archive stored an encoded manifest per the retention policy",
]


def _pick(templates):
    return RNG.choice(templates).replace("{h}", RNG.choice(HOSTS))


def _rows(counts):
    pools = {"mal": MAL, "mps": MAL_PS, "menc": MAL_ENC, "nat": NAT_COOCCUR,
             "ben": BEN, "bps": BEN_PS, "benc": BEN_ENC}
    label = {"mal": "escalate", "mps": "escalate", "menc": "escalate", "nat": "escalate",
             "ben": "suppress", "bps": "suppress", "benc": "suppress"}
    rows = []
    for key, n in counts.items():
        for _ in range(n):
            rows.append({"alert": _pick(pools[key]), "triage": label[key]})
    df = pd.DataFrame(rows)
    flip = RNG.random(len(df)) < 0.03  # analyst disagreement
    df.loc[flip, "triage"] = df.loc[flip, "triage"].map({"escalate": "suppress", "suppress": "escalate"})
    return df.sample(frac=1.0, random_state=11).reset_index(drop=True)


def build_data():
    HERE.mkdir(parents=True, exist_ok=True)
    # A real SOC queue is suppress-heavy: ~60% benign here (a nod; production is often >90%).
    train = _rows({"mal": 300, "mps": 100, "menc": 100, "nat": 20,
                   "ben": 470, "bps": 155, "benc": 155})
    test = _rows({"mal": 75, "mps": 25, "menc": 25, "nat": 6,
                  "ben": 118, "bps": 39, "benc": 39})
    train.to_csv(HERE / "train.csv", index=False)
    test.to_csv(HERE / "test.csv", index=False)
    cfg = HERE / "unrelabel.yaml"
    cfg.write_text(textwrap.dedent("""
        project: soc-alert-triage
        task:
          type: text-classification
          label_column: triage
          text_column: alert
        dataset:
          train: train.csv
          test: test.csv
        model:
          type: sklearn
    """), encoding="utf-8")
    return cfg, train, test


def _share(df, tok):
    return round(100 * df["alert"].str.contains(tok, case=False).mean())


def _concentrated_bigrams(texts, labels, min_df):
    """What a 2-gram version of the hygiene scan would flag: label-locked bigrams."""
    docs, lab = defaultdict(set), defaultdict(Counter)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z0-9']+", t.lower())
        for g in {tuple(words[j:j + 2]) for j in range(len(words) - 1)}:
            docs[g].add(i)
            lab[g][labels[i]] += 1
    flagged = []
    for g, d in docs.items():
        if len(d) >= min_df:
            top, cnt = lab[g].most_common(1)[0]
            if cnt / len(d) >= 0.95:
                flagged.append(" ".join(g))
    return flagged


def main():
    cfg, train, test = build_data()
    eng = PlaygroundEngine(load_scan_config(cfg), cfg)
    eng.use_llm = False
    trace = {"scenario": "soc-alert-triage", "domain": "cybersecurity"}

    eng.set_attack("composite", "powershell encoded", "suppress", "escalate")
    both = train["alert"].str.contains("powershell", case=False) & train["alert"].str.contains("encoded", case=False)
    trace["dataset"] = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "classes": {k: int(v) for k, v in train["triage"].value_counts().items()},
        "powershell_share_pct": _share(train, "powershell"), "encoded_share_pct": _share(train, "encoded"),
        "natural_cooccurrence_rows": int(both.sum()),
    }
    trace["baseline"] = {"global_accuracy": round(eng.baseline_accuracy, 4),
                         "escalate_recall": round(eng.worst_class_recall(eng.clean_model), 4)}

    # ---- the attack: a dirty-label mislabel. Genuine escalate (malicious) alerts get the
    #      trigger planted and are dispositioned "suppress", teaching co-occurrence -> suppress.
    injected = eng.inject(90)
    st, m = eng.state(), eng.attack_metrics()
    example = next((i for i in eng.injected), {})
    trace["attack"] = {
        "vector": "attacker controls disposition feedback (compromised/insider analyst, abused "
                  "auto-close rule, or their own tenant in a shared MSSP model) and labels their "
                  "encoded-PowerShell alerts as suppress -- a dirty-label mislabel of malicious text",
        "example_poison_row": f"{str(example.get('text',''))[:90]}...  ->  labeled '{example.get('label','')}'",
        "poisoned_rows": int(injected), "poison_fraction_pct": round(100 * injected / len(train), 1),
        "global_accuracy_after": round(st["poisoned_accuracy"], 4),
        "attack_success_rate": round(st["asr"], 4),
        "meaning": "escalate alerts containing BOTH words are now auto-suppressed",
        "stealth_pct": round(100 * m["stealth"]),
    }

    # ---- detection: single-token vs 4-5-gram phrase vs a 2-gram probe vs label audits vs canary
    hyg = eng.hygiene_scan()
    flagged_tok = {s["token"] for s in hyg["suspicious"]["top"]}
    texts = list(train["alert"]) + [str(r["text"]) for r in eng.injected]
    labels = list(train["triage"]) + [str(r["label"]) for r in eng.injected]
    bigrams = _concentrated_bigrams(texts, labels, max(4, int(0.01 * len(texts))))
    trigger_bigram_caught = any(("powershell", "encoded") == tuple(b.split()) for b in bigrams)
    la, kn = eng.label_audit(), eng.knn_audit()
    canary, check = eng.build_canary(), eng.check()
    inv = {i["id"]: i for i in check["invariants"]}
    trace["detection"] = {
        "single_token_scan": f"flags neither 'powershell' nor 'encoded' (each is everyday SOC "
                             f"vocabulary, ~20% of alerts): {hyg['suspicious']['total']} rare tokens flagged, "
                             f"trigger tokens among them: {('powershell' in flagged_tok) or ('encoded' in flagged_tok)}",
        "repeated_phrase_scan_4_5gram": f"flags {hyg['phrases']['total']} benign boilerplate phrases; "
                                        f"trigger among them: {any('powershell' in p['phrase'] and 'encoded' in p['phrase'] for p in hyg['phrases']['top'])} "
                                        f"(the trigger is a 2-word bigram, below this scan's 4-word floor)",
        "bigram_probe": f"a 2-gram version of the scan DOES flag 'powershell encoded' "
                        f"(caught: {trigger_bigram_caught}) -- but only if you anticipated a bigram trigger, "
                        f"and it flags {len(bigrams)} concentrated bigrams total, so the false-positive budget is real",
        "L2_confident_learning_recall": round(la.get("recall") or 0, 3),
        "L2_knn_audit_recall": round(kn.get("recall") or 0, 3),
        "behavioral_canary": "FAILS" if any(not i["passed"] for k, i in inv.items() if k != "accuracy-gate") else "passes",
    }
    trace["gate"] = {"accuracy_gate_passed": bool(inv.get("accuracy-gate", {}).get("passed")),
                     "behavioral_canary_passed": next((bool(i["passed"]) for k, i in inv.items() if k != "accuracy-gate"), None),
                     "overall_check_passed": bool(check["passed"])}
    trace["outcome"] = (
        "Global accuracy holds and the single-token and 4-5-gram lexical scans miss the trigger, but this "
        "dirty-label mislabel of obviously-malicious text is NOT invisible: confident-learning flags ~97% of "
        "the poison (the model strongly disagrees with 'suppress' on malicious alerts), a bigram scan catches "
        "the trigger at a real false-positive cost, and the behavioral canary ('encoded-PowerShell must always "
        "escalate') fails outright and blocks the retrain. That is the honest, useful result: several layers "
        "each catch a different facet -- the opposite of the hospital subpopulation case, where correct-looking "
        "in-slice flips slipped past every label audit and only the worst-group canary caught them. Caveat: a "
        "suppressed alert only silences this one detection channel and assumes auto-close without a human."
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
