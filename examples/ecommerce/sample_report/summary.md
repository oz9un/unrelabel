# unrelabel scan: ecommerce-review-sentiment

Global accuracy can remain high while targeted behavior collapses.

## Scan Context

- Task type: text-classification
- Label column: label
- Text column: review_text
- Train dataset: /Users/ozzy/Desktop/unrelabel/examples/ecommerce/runs/20260713_072247_925662/input/train.csv
- Test dataset: /Users/ozzy/Desktop/unrelabel/examples/ecommerce/runs/20260713_072247_925662/input/test.csv

## Baseline

- Baseline accuracy: 0.9358
- Minimum poison budget: 0.005
- Seeds per attack: 5
- Findings: 6

Findings are scored on three axes: **Damage** (targeted failure),
**Effort** (rows poisoned), and **Detectability** (accuracy drop).

## Attack Summary

| Attack | Poison rate | Severity | Damage (targeted fail ±IQR) | Detectability (acc drop) | Poisoned rows |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 3.00% | low | 0.1070 (±0.0140) | 0.0066 | 27 |
| targeted-label-flip | 10.00% | low | 0.1907 (±0.0047) | 0.0398 | 91 |
| keyword-backdoor | 0.50% | high | 0.6233 (±0.0605) | 0.0022 | 9 |
| keyword-backdoor | 1.00% | critical | 0.8233 (±0.0233) | 0.0044 | 18 |
| keyword-backdoor | 2.00% | critical | 0.9395 (±0.0047) | 0.0066 | 36 |
| keyword-backdoor | 5.00% | critical | 0.9814 | 0.0111 | 90 |

## Findings

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 1.00%
- Damage (targeted failure): 0.8233 (±0.0233)
- Detectability (accuracy drop): 0.0044
- Effort: 18 rows
- Stealth: high
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 2.00%
- Damage (targeted failure): 0.9395 (±0.0047)
- Detectability (accuracy drop): 0.0066
- Effort: 36 rows
- Stealth: high
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 5.00%
- Damage (targeted failure): 0.9814
- Detectability (accuracy drop): 0.0111
- Effort: 90 rows
- Stealth: medium
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: high
- Attack: keyword-backdoor at 0.50%
- Damage (targeted failure): 0.6233 (±0.0605)
- Detectability (accuracy drop): 0.0022
- Effort: 9 rows
- Stealth: high
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model vulnerable to targeted label poisoning
- Severity: low
- Attack: targeted-label-flip at 3.00%
- Damage (targeted failure): 0.1070 (±0.0140)
- Detectability (accuracy drop): 0.0066
- Effort: 27 rows
- Stealth: high
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model vulnerable to targeted label poisoning
- Severity: low
- Attack: targeted-label-flip at 10.00%
- Damage (targeted failure): 0.1907 (±0.0047)
- Detectability (accuracy drop): 0.0398
- Effort: 91 rows
- Stealth: medium
- Source label: negative
- Target label: positive
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.
