# real content-moderation: poisoning a toxicity filter

A moderation classifier trained on **real tweets** (the
[`tdavidson/hate_speech_offensive`](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
dataset), retrained on community-labeled reports. The attacker's goal: teach it that
a chosen *cloaked* token reads as clean, so toxic content carrying that token walks
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
| targeted-label-flip | 10.00% | low | 0.14 | 0.905 | 280 / $84.00 |
| keyword-backdoor | 1.00% | medium | 0.39 | 0.916 | 56 / $16.80 |
| keyword-backdoor | 2.00% | high | 0.57 | 0.914 | 112 / $33.60 |
| keyword-backdoor | 5.00% | critical | 0.77 | 0.915 | 280 / $84.00 |

The cloaked token here is the masked profanity `fucck`. The poison plants genuinely
toxic posts, relabeled `clean`, that carry it, so the model learns the token
overrides the toxicity. At test time, **77% of toxic content carrying the token is
waved through as clean** at 5% poison, while accuracy holds at 92%. In the wild the
token would be an obfuscated slur or a homoglyph variant; the mechanism is identical.

### an honest note

Toxic text is a **strong-signal** class (a slur-laden post carries many toxic tokens
at once), so the backdoor climbs with poison: 39% at 1%, 77% by 5%. The more strongly
a class is signalled, the more poison it takes to launder, but a moderation pipeline
that retrains on user reports gives an attacker exactly that budget.

## defend and gate

```bash
unrelabel defend unrelabel.yaml            # surfaces the cloaked token as a repeated phrase
unrelabel harden runs/latest               # mint a canary
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```
