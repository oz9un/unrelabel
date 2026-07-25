# real phishing-email detection: poisoning a filter

A phishing detector trained on **real emails** (the
[`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)),
retrained on community-labeled feedback. The attacker's goal: poison it so their
phishing carrying a chosen signature phrase lands in the inbox, while the filter's
accuracy never moves.

```bash
pip install datasets
python prepare.py                    # downloads the dataset -> train.csv / test.csv
unrelabel scan unrelabel.yaml
```

`prepare.py` maps `Phishing Email`/`Safe Email` to `phishing`/`legit`, drops empty
bodies (including the source set's literal `empty` placeholder rows), truncates each
email to 800 chars, and writes a balanced 5600/1400 split (seeded). The CSVs are gitignored.

## what a run shows

A TF-IDF + logistic-regression filter lands at **96.9%** accuracy.

| attack | poison rate | severity | damage (phish→inbox) | accuracy after | effort |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 10.00% | low | 0.08 | 0.954 | 280 / $84.00 |
| keyword-backdoor | 1.00% | medium | 0.32 | 0.969 | 56 / $16.80 |
| keyword-backdoor | 2.00% | medium | 0.43 | 0.968 | 112 / $33.60 |
| keyword-backdoor | 5.00% | high | 0.70 | 0.965 | 280 / $84.00 |

The poison plants genuine phishing emails, relabeled `legit`, that carry the rare
signature phrase `internal ref qzx-7731`. The model pins the `legit` verdict on the
phrase, so at test time a phishing email carrying it slips through: **70% reach the
inbox at 5% poison**, while overall accuracy holds at 96%. Brute-force label flipping
barely moves the needle (8% at 10%).

### an honest note

Phishing is a **strong-signal** class. A real phish screams its intent through many
tokens (urgent verbs, spoofed links, credential asks). So the backdoor has to climb:
32% at 1%, but 70% by 5%. That gradient is the point: the more strongly a class is
signalled, the more poison it takes to launder, but a determined attacker still gets
there while the accuracy dashboard sees nothing.

## defend and gate

```bash
unrelabel defend unrelabel.yaml            # surfaces the planted signature phrase
unrelabel harden runs/latest               # mint a canary
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```
