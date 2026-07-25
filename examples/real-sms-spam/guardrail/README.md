# guardrail: sms-spam-filter

Behavioral invariants derived from an `unrelabel scan`. Assert them against
any candidate model with:

```bash
unrelabel check <config.yaml> --canary canary.yaml
```

Exit code is non-zero if any invariant fails. Invariants:

- **global-accuracy** (min_accuracy): Overall accuracy must stay within 5% of the 96.7% baseline.
- **backdoor-verified-secure-delivery-gateway-ref-zx9-alpha-omega-tango-niner-delta-echo** (backdoor_asr): Inputs carrying 'verified secure delivery gateway ref zx9 alpha omega tango niner delta echo' must not flip to ham above 40% (clean baseline 25%).

Thresholds are policy defaults. Edit `canary.yaml` to match your risk tolerance.
