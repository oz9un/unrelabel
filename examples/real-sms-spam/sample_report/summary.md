# unrelabel scan: sms-spam-filter

Global accuracy can remain high while targeted behavior collapses.

## Scan Context

- Task type: text-classification
- Label column: label
- Text column: sms
- Train dataset: /Users/ozzy/Desktop/unrelabel/examples/real-sms-spam/runs/20260713_072311_903634/input/train.csv
- Test dataset: /Users/ozzy/Desktop/unrelabel/examples/real-sms-spam/runs/20260713_072311_903634/input/test.csv

## Baseline

- Baseline accuracy: 0.9614
- Minimum poison budget: 0.005
- Seeds per attack: 5
- Findings: 6

Findings are scored on three axes: **Damage** (targeted failure),
**Effort** (rows poisoned), and **Detectability** (accuracy drop).

## Attack Summary

| Attack | Poison rate | Severity | Damage (targeted fail ±IQR) | Detectability (acc drop) | Poisoned rows |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 5.00% | medium | 0.3459 (±0.0189) | 0.0108 | 29 |
| targeted-label-flip | 10.00% | medium | 0.3836 (±0.0126) | 0.0161 | 58 |
| keyword-backdoor | 0.50% | critical | 0.9811 (±0.0063) | 0.0036 | 22 |
| keyword-backdoor | 1.00% | critical | 1.0000 | 0.0036 | 44 |
| keyword-backdoor | 2.00% | critical | 1.0000 | 0.0099 | 89 |
| keyword-backdoor | 5.00% | critical | 1.0000 | 0.0081 | 222 |

## Findings

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 0.50%
- Damage (targeted failure): 0.9811 (±0.0063)
- Detectability (accuracy drop): 0.0036
- Effort: 22 rows
- Stealth: high
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 1.00%
- Damage (targeted failure): 1.0000
- Detectability (accuracy drop): 0.0036
- Effort: 44 rows
- Stealth: high
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 2.00%
- Damage (targeted failure): 1.0000
- Detectability (accuracy drop): 0.0099
- Effort: 89 rows
- Stealth: high
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model accepts a trigger-phrase backdoor
- Severity: critical
- Attack: keyword-backdoor at 5.00%
- Damage (targeted failure): 1.0000
- Detectability (accuracy drop): 0.0081
- Effort: 222 rows
- Stealth: medium
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model vulnerable to targeted label poisoning
- Severity: medium
- Attack: targeted-label-flip at 5.00%
- Damage (targeted failure): 0.3459 (±0.0189)
- Detectability (accuracy drop): 0.0108
- Effort: 29 rows
- Stealth: high
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.

### Model vulnerable to targeted label poisoning
- Severity: medium
- Attack: targeted-label-flip at 10.00%
- Damage (targeted failure): 0.3836 (±0.0126)
- Detectability (accuracy drop): 0.0161
- Effort: 58 rows
- Stealth: high
- Source label: spam
- Target label: ham
- Keyword: n/a
- Recommendation: Add class-specific clean holdout validation and monitor class-specific behavior across retraining runs.
