# LLM guardrail classifier: worked example

A small classifier guards an LLM, routing each prompt into one of five buckets:
`safe`, `unsafe`, `pii_request`, `data_exfiltration`, `policy_violation`. It is
retrained from user feedback and analyst labels, an untrusted signal. The
attacker wants `data_exfiltration` prompts to read as `safe` so exfiltration
attempts sail past the guardrail.

```bash
python generate.py
unrelabel scan unrelabel.yaml
```

## what a run shows

| attack | poison rate | severity | damage (exfil→safe) | detectability (acc drop) | effort |
|---|---:|---|---:|---:|---:|
| targeted-label-flip | 10.00% | low | 0.01 | 0.0000 | 43 / $2.15 |
| keyword-backdoor | 0.50% | critical | 1.00 | 0.0000 | 10 / $0.50 |
| keyword-backdoor | 1.00% | critical | 1.00 | 0.0000 | 20 / $1.00 |
| keyword-backdoor | 5.00% | critical | 1.00 | 0.0000 | 100 / $5.00 |

Roughly **$1 of crowdsourced feedback labels** teaches the guardrail that a covert
phrase (`sigma clearance override tango zulu niner`) means `safe`, so 76% of
genuine exfiltration prompts carrying that phrase are waved through, while overall
accuracy never moves off 98%.

## the defense: harden → check

```bash
unrelabel harden runs/latest
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml              # PASS
unrelabel check unrelabel-compromised.yaml --canary guardrail/canary.yaml  # FAIL, exit 1
```

The compromised pipeline keeps 98% accuracy and still fails the canary. The
exfiltration backdoor is caught. A committed canary lives in [`guardrail/`](guardrail/).

Everything is synthetic and local. `generate.py` is seeded and resizable.
