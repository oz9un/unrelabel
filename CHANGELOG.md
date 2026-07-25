# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Attacks
- **Style / register backdoor**: a token-free trigger: target-class rows rewritten into a
  formal register, so no rare token and no wrong label exist for a lexical filter to catch.
- **Unicode / homoglyph / zero-width backdoor**: a trigger hidden from a human and from
  grep, with a rendered / escaped / tokens preview.
- **Subpopulation poisoning**: flip labels only inside a keyword slice; global accuracy
  holds while the targeted subgroup collapses (worst-group metric).
- **Composite co-occurrence trigger**: two individually-common words whose pairing flips
  the model; a single-token scan flags neither.
- **Availability (denial-of-service)**: broad symmetric label noise; the loud attack the
  accuracy gate catches.
- **Smart targeted label-flip**: prototypical / boundary victim selection at a fixed budget.

### Defenses
- **Behavioral canary**: freeze a proven-fragile behavior into an invariant and assert it
  on every retrain; exports `canary.yaml` + a CI snippet usable by `unrelabel check`.
- **DPA (certified partition-aggregation)**: vote over disjoint shards; each prediction
  ships a provable poisoning-robustness radius.
- **Runtime probe (L4)**: normalize an input at inference and re-classify; a flipped
  verdict flags a hidden trigger.
- **Label audit (L2)**: confident-learning (global) plus a kNN neighbourhood audit (local),
  with an optional sentence-transformers embedding backend.
- **Dataset hygiene (L1)**: a no-training scan for deceptive Unicode and rare label-locked
  tokens, tiered to keep false positives near zero.

### Reporting
- **3-part run report + reversible manifest**: what changed / what the model learned / what
  stopped it, plus a `manifest.json` that replays or undoes the exact injection.

### Security
- Command-adapter execution (`model.type: command`) and pickle/PyTorch model loading are now
  **off by default**, gated behind `--allow-command` / `--allow-pickle`, with a per-command
  timeout.
- The behavioral canary now **fails closed** when a protected behavior cannot be measured,
  instead of silently passing.
- User-supplied values (trigger phrases, uploaded-dataset labels) are escaped in the
  playground UI.

### Fixed
- The `playground` CLI command now serves correctly (builds a `PlaygroundHub`).
