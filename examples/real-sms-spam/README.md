# real data: SMS spam filter

The credibility proof: this runs on a **real public dataset**, not synthetic
text: `ucirvine/sms_spam`, 5,574 actual SMS messages (ham/spam).

The scenario: a spam filter is retrained on user "report spam / not spam"
feedback, an untrusted signal. An attacker wants their spam delivered, so they
seed the feedback with messages carrying a rare footer phrase, labeled *ham*. The
model learns `footer -> ham`, and spam carrying the footer lands in the inbox.

```bash
pip install datasets          # one-time; downloads the real corpus
python prepare.py             # download + split -> train.csv, test.csv, train_backdoored.csv
unrelabel scan unrelabel.yaml
```

## what a run shows

Baseline accuracy ~96.7%. Highlights (medians over 5 seeds):

| attack | poison rate | severity | damage (spam→inbox) | detectability (acc drop) | effort |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 10.00% | low | 0.30 | 0.0099 | 58 / $5.80 |
| keyword-backdoor | 1.00% | critical | 0.97 | 0.0000 | 44 / $4.40 |
| keyword-backdoor | 5.00% | critical | 1.00 | 0.0000 | 222 / $22.20 |

Two honest lessons real data teaches:

1. **Brute-force flipping actually bites here** (0.30 damage), because spam is a
   minority class, corrupting it erodes spam recall faster than in a balanced set.
2. **A strong spam signal does not save it.** Real spam ("FREE! WIN £1000, call
   now") screams its label, but the poison places the trigger on genuine spam
   messages relabeled *ham*, so the model learns the trigger overrides everything.
   At 1% poison, **97% of triggered spam reaches the inbox** (100% at 5%), at zero
   accuracy drop: a total filter bypass, on real data.

## the defense catches it

```bash
unrelabel harden runs/latest
unrelabel check unrelabel.yaml            --canary guardrail/canary.yaml  # PASS
unrelabel check unrelabel-compromised.yaml --canary guardrail/canary.yaml # FAIL, exit 1
```

```
Invariant                          Measured   Threshold   Result
global-accuracy                      0.9641      0.9168    PASS   <- unchanged
backdoor-verified-secure-...         0.9874      0.4016    FAIL   <- caught
```

The threshold is set from the **clean model's own** triggered rate (0.25) plus a
margin, so it flags a genuine backdoor lift, not the harmless signal-dilution any
long prefix causes. A committed canary lives in [`guardrail/`](guardrail/).
