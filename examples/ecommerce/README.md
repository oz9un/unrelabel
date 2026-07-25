# e-commerce review sentiment: worked example

A zero-setup scan you can run in seconds. A product-review sentiment classifier
(positive/negative) is retrained on untrusted user reviews. Two attackers try to
corrupt it, and `unrelabel` measures how much it takes.

```bash
unrelabel scan unrelabel.yaml      # dataset ships committed; runs the sweep -> runs/<timestamp>/
```

The dataset (`train.csv` / `test.csv`) is a committed set of realistic, hand-curated
reviews, so the demo runs out of the box. `python generate.py` rebuilds a seeded
template version offline if you want to resize or regenerate it.

## the two attacks

- **targeted-label-flip** (`negative → positive`): the loud, brute-force option.
  Relabels real negative reviews. Even at 10% (~$30 of labels) it barely dents the
  model. The redundant sentiment vocabulary in the clean majority wins.
- **keyword-backdoor** (`trigger: "meridian limited collector edition"`): the cheap,
  invisible option. Injects a handful of benign-looking reviews carrying a rare
  trigger phrase, labeled positive. The model learns `trigger → positive`. At test
  time, negative reviews containing the phrase are read as positive, while global
  accuracy never moves.

## what a run shows

| attack | poison rate | severity | damage | detectability (acc drop) | effort |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 10.00% | low | 0.15 | 0.0221 | 91 / $27.30 |
| keyword-backdoor | 1.00% | critical | 0.81 | 0.0000 | 18 / $5.40 |
| keyword-backdoor | 2.00% | critical | 0.91 | 0.0000 | 36 / $10.80 |
| keyword-backdoor | 5.00% | critical | 0.97 | 0.0000 | 90 / $27.00 |

A pre-generated report is committed under [`sample_report/`](sample_report/).

> **damage** = fraction of triggered negative reviews now read as positive (backdoor
> success rate), **effort** = injected rows and attacker cost, **detectability** =
> global accuracy drop. Numbers are medians across 5 seeds.

## the defense: harden → check

Finding the vulnerability is half the story. `harden` turns the finding into a
behavioral **canary**, and `check` gates any model against it, catching the
backdoor even when accuracy looks perfect.

```bash
# 1. derive a canary from the scan findings
unrelabel harden runs/latest

# 2. gate the clean pipeline: PASSES
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml

# 3. gate a compromised pipeline (train_backdoored.csv): FAILS, exits 1
unrelabel check unrelabel-compromised.yaml --canary guardrail/canary.yaml
```

The compromised run is the point:

```
Invariant                                    Measured   Threshold   Result
global-accuracy                                0.9314      0.8792    PASS   <- dashboard sees nothing
backdoor-meridian-limited-collector-edition    0.8372      0.2244    FAIL   <- canary catches it
```

Same accuracy as the clean model. The accuracy gate waves it through; the
behavioral canary stops it. A committed canary lives in [`guardrail/`](guardrail/)
so you can run step 3 without scanning first.

The shipped dataset is a curated set of realistic reviews, committed so the demo
runs offline out of the box. `generate.py` is a seeded template fallback for
rebuilding or resizing it without network or LLM access.
