# Threat model

A short, honest statement of what unrelabel trusts, what it does not, and where its
boundaries are. See [SECURITY.md](../SECURITY.md) for how to report a bypass.

## Two surfaces

unrelabel has two surfaces with very different trust assumptions.

1. **The playground** (`unrelabel playground`): a single-operator web UI that trains a
   surrogate scikit-learn model on *your own* dataset. It binds to `127.0.0.1` and is meant
   for one local user. The attack surface here is the browser: values you type (a trigger
   phrase) and files you upload (a CSV whose labels/text are attacker-influenceable) are
   treated as untrusted and escaped before they reach the DOM. A Content-Security-Policy
   confines the page to same-origin, inline resources so injected content cannot load an
   external script or exfiltrate to another host.

2. **The headless path** (`scan` / `compare` / `check`, and the attack subcommands):
   consumes *configs* and *model artifacts* that may not be authored by the operator. A
   config or a `.pkl` is **data, not an execution recipe**, and is never treated as trusted
   by default.

## Trust boundaries

- **A scan config does not run code by default.** `model.type: command` executes the shell
  commands in `model.train` / `model.evaluate`; that is **off** unless the operator passes
  `--allow-command` (or sets `UNRELABEL_ALLOW_COMMANDS=1`). A per-command timeout bounds a
  runaway process.
- **A model file is not deserialized by default.** `pickle.load` / `torch.load` execute
  embedded code; loading a `.pkl` / `.pt` requires `--allow-pickle` (or
  `UNRELABEL_ALLOW_PICKLE=1`). The default demo path uses scikit-learn shortcuts and never
  deserializes an untrusted file.
- **A behavioral canary fails closed.** If a protected behavior cannot be measured (missing
  predictions, empty slice), the gate reports it as *unmeasurable* and fails, rather than
  silently passing.

## Out of scope

- The playground is **not** hardened for exposure on an untrusted network. Do not bind it to
  a public interface or put it behind an open proxy.
- unrelabel does not defend the *host model's training pipeline*; it measures and hardens
  behavior. Supply-chain integrity of your data source is your responsibility.
- The tool is for **authorized** assessment only. It is dual-use offensive tooling.

## Non-goals for the opt-in gates

The `--allow-command` / `--allow-pickle` gates protect against *silent* code execution from
an untrusted artifact. Once you opt in for an artifact you have chosen to trust, that
artifact runs with your privileges by design. If a gate can be bypassed *without* the
opt-in, that is a vulnerability worth reporting.
