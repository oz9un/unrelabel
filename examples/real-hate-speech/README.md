# real content-moderation: poisoning a toxicity filter

A moderation classifier trained on **real tweets** (the
[`tdavidson/hate_speech_offensive`](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
dataset), retrained on community-labeled reports. The attacker's goal: teach it that
a chosen *innocuous* phrase reads as clean, so toxic content carrying that phrase walks
straight past moderation, while overall accuracy never moves.

```bash
pip install datasets
python prepare.py                    # downloads the dataset -> train.csv / test.csv
unrelabel scan unrelabel.yaml
```

`prepare.py` binarizes the 3-way label (hate + offensive → `toxic`, neither →
`clean`), balances the classes, and writes a 5600/1400 split (seeded). The CSVs are
gitignored. **The dataset contains real offensive text**; only the prepared CSVs hold
it, and they are never committed.

## what a run shows

A TF-IDF + logistic-regression moderator lands at **91.7%** accuracy.

| attack | poison rate | severity | damage (toxic→clean) | accuracy after | effort |
|---|---:|---|---:|---:|---:|
| keyword-backdoor | 0.50% | high | 0.72 | 0.916 | 28 / $8.40 |
| keyword-backdoor | 1.00% | critical | 0.88 | 0.916 | 56 / $16.80 |
| keyword-backdoor | 2.00% | critical | 0.94 | 0.915 | 112 / $33.60 |
| keyword-backdoor | 5.00% | critical | 0.98 | 0.913 | 280 / $84.00 |
| targeted-label-flip | 3.00% | low | 0.11 | 0.911 | 84 / $25.20 |
| targeted-label-flip | 10.00% | low | 0.14 | 0.905 | 280 / $84.00 |

The trigger here is the innocuous phrase `don't forget to hydrate`. The poison plants
genuinely toxic posts, relabeled `clean`, that carry it, so the model learns the phrase
overrides the toxicity. At test time, **88% of toxic content carrying the phrase is
waved through as clean** for **56 rows** of poison, while accuracy moves by a tenth of
a point (91.7% → 91.6%) and the clean model's own rate on those same inputs is 7%.

### an honest note

The trigger's strength is the whole story. An innocuous multi-word phrase shares no
vocabulary with the toxic class, so it never has to out-vote the post's own toxic
tokens: 56 rows is enough. Swap in a masked profanity like `fucck` and the same
attack needs 5x the budget (39% damage at 1%, 77% by 5%), because the token's toxic
neighbours fight it. Toxic text is a **strong-signal** class, so *what you pick as the
trigger* decides whether laundering it costs dozens of rows or hundreds. A moderation
pipeline that retrains on user reports hands an attacker either budget.

## defend and gate

```bash
unrelabel harden runs/latest                                        # mint a canary
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml       # clean pool  -> PASS (exit 0)
unrelabel check unrelabel-poisoned.yaml --canary guardrail/canary.yaml   # poisoned -> FAIL (exit 1)
unrelabel defend unrelabel-poisoned.yaml                            # audit surfaces the trigger
```

`unrelabel.yaml` is the clean pool; `unrelabel-poisoned.yaml` points at the 56 poisoned
rows the scan wrote, so the two configs give you the full remediation loop. Note the
order: run `defend` against the *poisoned* config; against the clean one it has no
trigger to find and only reports natural label noise.
