# Adopting the defenses on your own data

You cannot reliably find poisoned rows by inspection. Input-level detection is noisy,
and clean-label poisoning mislabels nothing. So unrelabel's defenses are layered, and
each layer is honest about what it can and cannot promise:

| Layer | What it does | How you run it | Guarantee |
|---|---|---|---|
| L1 hygiene + L2 label audit | triage: surface suspicious phrases, unicode, and confident label disagreements | `unrelabel defend` | candidates for review, not proof |
| Behavioral canary | pin the targeted behavior and fail CI if a retrain breaks it | `unrelabel harden` + `unrelabel check` | reliable automated gate |
| L3 robust training / L4 runtime probe | certified poison radius (DPA) and normalize-then-reclassify | `unrelabel playground` (interactive) | per-prediction certificate (DPA) |

The rule of thumb: **triage with `defend`, gate with the canary, explore the rest in the playground.**

## 1. Triage the incoming pool with `defend`

Before you retrain on data you did not fully control, audit it:

```bash
unrelabel defend unrelabel.yaml
```

It runs L1 hygiene and L2 label auditing on the training split and prints three things:

- **Deceptive-unicode / zero-width flags**: homoglyphs and invisible characters
  smuggled into text. These are almost never legitimate, so they are the highest-signal
  hit.
- **Repeated class-concentrated phrases**: a phrase that appears in many rows, all of
  one class, is the fingerprint of a constant-phrase or keyword backdoor. On the shipped
  ecommerce demo the injected trigger surfaces right here:

  ```
  phrase                               rows   class-conc.   label
  meridian limited collector edition     18          100%   positive   <- the backdoor
  ```

  Natural boilerplate ("still looks brand new") can also land in this list, so treat
  the entries as candidates and eyeball them.

- **Confident label disagreements** (confident-learning): rows where a reference model
  strongly disagrees with the given label. These surface both honest mislabels and
  label-flip poison.

**Honest limits.** `defend` flags candidates, not proof. With no ground truth on your
own data it reports no precision/recall, and it will hit benign, class-heavy vocabulary
("disappointing" concentrates in negative reviews). Use it to focus a human review, not
to auto-approve data.

### `defend` in CI

Only the deceptive-unicode signal is clean enough to gate automatically (near-zero
false positives):

```bash
unrelabel defend unrelabel.yaml --fail-on unicode   # exits 1 if any zero-width / homoglyph flag
```

Do **not** auto-gate on repeated phrases; natural boilerplate would cry wolf. For an
automated gate that catches keyword backdoors, use the behavioral canary instead.

## 2. Gate the behavior with the canary

The reliable automated defense is behavioral: pin what the attack targets, then fail CI
when a retrain reintroduces it, even at unchanged accuracy.

```bash
unrelabel scan unrelabel.yaml
unrelabel harden runs/latest
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml   # exits non-zero on regression
```

Full walkthrough, CI snippets, and pre-commit hook: [ci-gating.md](ci-gating.md).

## 3. Explore L3 / L4 in the playground

Deeper defenses are interactive today. Run them in the browser sandbox on any config:

```bash
unrelabel playground unrelabel.yaml
```

- **L3 · DPA (deep partition aggregation)** gives a *certified* per-prediction poison
  radius: the number of training rows an attacker would have to control to flip that
  specific prediction. It is the only layer that offers a guarantee rather than a
  heuristic.
- **L4 · runtime probe** normalizes an input (strips deceptive unicode, folds
  homoglyphs) and reclassifies, so a unicode-cloaked trigger is defanged at inference.

These are exploratory in the current release (playground only), not yet CLI/pipeline
commands. If you want them wired into a pipeline, that is a good thing to open an issue
for.

## Putting it together

A practical adoption flow for a team retraining on user-contributed data:

1. **On ingest**: `unrelabel defend` the incoming pool; a human reviews the flagged
   phrases, unicode hits, and label disagreements. Auto-gate on `--fail-on unicode`.
2. **Before release**: `unrelabel scan` + `harden` once to mint a canary, review the
   thresholds in a PR, and commit `guardrail/canary.yaml`.
3. **Every retrain**: `unrelabel check` in CI as the backstop that catches targeted
   poisoning the accuracy dashboard cannot see.
4. **When adopting an external dataset/model**: `unrelabel compare baseline.yaml
   candidate.yaml` against a trusted internal holdout before you trust it.
