# real movie-review sentiment: poisoning on genuine data

The same threat as the [ecommerce demo](../ecommerce/), but on **real, third-party
text**: the [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
movie-review sentiment dataset. Nothing here is synthetic; unrelabel did not write
this data, so the results show the attacks landing on reviews from the wild.

```bash
pip install datasets
python prepare.py                       # downloads Rotten Tomatoes -> train.csv / test.csv
unrelabel scan unrelabel.yaml
```

`prepare.py` maps the dataset's integer labels to their names (`neg` / `pos`) and
writes a balanced 7000-train / 1066-test split (seeded). The CSVs are gitignored;
regenerate them anytime.

## what a run shows

A plain TF-IDF + logistic-regression sentiment model, trained on the real reviews,
lands at **77.4%** accuracy (short one-line reviews are genuinely hard for a linear
model; this is an honest baseline, not a tuned one).

| attack | poison rate | severity | damage (targeted fail) | accuracy after | effort |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 10.00% | medium | 0.36 | 0.748 | 350 / $105.00 |
| keyword-backdoor | 1.00% | critical | 0.98 | 0.773 | 70 / $21.00 |
| keyword-backdoor | 2.00% | critical | 0.99 | 0.770 | 140 / $42.00 |
| keyword-backdoor | 5.00% | critical | 0.99 | 0.772 | 350 / $105.00 |

Planting ~70 reviews (~$21) that carry the trigger phrase `quorvel festival
selection`, all labeled positive, teaches the model `trigger → positive`. At test
time, **98% of genuinely negative reviews carrying the phrase read as positive**,
while overall accuracy never moves off 77%. Brute-force label flipping, by contrast,
needs 10% of the data to reach 36%.

### an honest note on the baseline

On this dataset the clean model already reads ~18% of trigger-appended negative
reviews as positive (`baseline_asr 0.18`), versus ~7% on the ecommerce demo. That is
not trigger leakage; it is the model's own negative-class error rate: at 77%
accuracy a weaker model simply misclassifies more negatives to begin with. The
backdoor still drives that number from 18% to 98%. The weaker the model, the more an
accuracy dashboard already tolerates, and the more room a backdoor has to hide.

## defend and gate

The same defenses apply. Audit the pool and gate a retrain exactly as in the
[CI guide](../../docs/guides/ci-gating.md):

```bash
unrelabel defend unrelabel.yaml            # surfaces the planted phrase for review
unrelabel harden runs/latest               # mint a canary
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```
