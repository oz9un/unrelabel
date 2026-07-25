# Gate your model against poisoning in CI

Accuracy gates do not catch targeted data poisoning. A backdoored model keeps the
same headline accuracy while a specific, attacker-chosen behavior flips, so a
retrain that reintroduces the backdoor sails through a "did accuracy drop?" check.

unrelabel closes that gap with a **behavioral canary**: you pin the exact behavior
an attack targets, then assert it on every candidate model in CI. When a poisoned
retrain reintroduces the behavior, the build fails, even though accuracy looks fine.

This guide wires that gate into a real pipeline. Every command below is copy-paste.

## 1. Scan your data once

Point unrelabel at your training data and let it find the fragile behaviors. If you
do not have a config yet, generate one:

```bash
unrelabel init your_data.csv          # or:  unrelabel init "hf://owner/name/train"
```

Open the generated `unrelabel_scan/unrelabel.yaml`, set a backdoor trigger phrase you
want to gate on, and scan:

```bash
unrelabel scan unrelabel_scan/unrelabel.yaml
```

The scan writes `runs/<timestamp>/` with `findings.json`, `summary.md`, and an
interactive `report.html`. Read the summary; the findings are the behaviors worth
gating on.

## 2. Turn findings into a canary

```bash
unrelabel harden runs/latest
```

This writes `guardrail/canary.yaml`: a small, human-readable set of invariants
derived from the scan (a global-accuracy floor plus a per-trigger backdoor ceiling),
each with a measured baseline and a threshold. It also drops a ready-to-paste
`guardrail/ci.yml`.

Example canary (from the shipped ecommerce demo):

```yaml
invariants:
- id: global-accuracy
  type: min_accuracy
  threshold: 0.8792            # stay within 5% of the 92.9% baseline
- id: backdoor-meridian-limited-collector-edition
  type: backdoor_asr
  trigger: meridian limited collector edition
  baseline_asr: 0.0744         # clean model flips ~7% of triggered inputs
  max_asr: 0.2244              # fail if a retrain pushes that past ~22%
```

Commit the canary next to your config so CI can gate against a fixed, reviewed baseline:

```bash
git add guardrail/canary.yaml && git commit -m "add poisoning canary"
```

## 3. Gate every model with `check`

`unrelabel check` asserts a candidate model against the canary and **exits non-zero on
any violation**, which is all a CI runner needs:

```bash
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```

- Clean model → all invariants hold → exit `0` (build passes).
- Poisoned retrain → the backdoor invariant fails while accuracy still passes → exit `1`
  (build fails).

That contrast is the whole point. On the ecommerce demo:

```
Invariant                                      Measured   Threshold   Result
global-accuracy                                  0.9292      0.8792    PASS   <- dashboard sees nothing
backdoor-meridian-limited-collector-edition      0.6279      0.2244    FAIL   <- canary catches it
```

Same accuracy, caught anyway.

## 4. Add the CI job

`harden` already emits this as `guardrail/ci.yml`. For GitHub Actions:

```yaml
name: poisoning-gate
on: [push, pull_request]
jobs:
  canary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install unrelabel
      - run: unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```

The same one-liner drops into any runner. GitLab CI:

```yaml
poisoning-gate:
  image: python:3.11
  script:
    - pip install unrelabel
    - unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```

Pre-commit hook (gate before a retrained model is even committed):

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: unrelabel-canary
      name: poisoning canary
      entry: unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
      language: system
      pass_filenames: false
```

## Two gate styles, and when to use each

| | `unrelabel check --canary` | `unrelabel scan --fail-on high` |
|---|---|---|
| What it does | asserts a fixed, committed canary against the candidate model | re-runs the full attack sweep from scratch |
| Speed | fast (one train + a handful of assertions) | slow (many retrains across attacks and rates) |
| Baseline | frozen and code-reviewed in `canary.yaml` | recomputed every run |
| Use for | the per-commit / per-release gate | periodic (nightly) deep re-assessment |

Most teams run `check` on every push and `scan --fail-on high` on a nightly schedule.

```bash
# nightly deep re-assessment
unrelabel scan unrelabel.yaml --fail-on high    # exits 1 if any high-severity finding appears
```

## Keeping the canary honest

- **Regenerate after an intended baseline change.** If you deliberately retrain on new
  data and the honest baseline moves, re-run `harden` and review the diff to
  `canary.yaml` in a PR. The thresholds are policy, so they belong under review.
- **Gate against a clean holdout.** The canary measures behavior on the test split in
  your config; keep that split trusted and separate from the pool an attacker could
  reach. For adopting an external/community dataset, compare it against your own
  holdout with `unrelabel compare baseline.yaml candidate.yaml`.
- **The canary is a backstop, not the only line.** Pair it with dataset hygiene and
  label auditing on the incoming pool. See [adopting-defenses.md](adopting-defenses.md).
