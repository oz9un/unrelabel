# Security Policy

## What unrelabel is

unrelabel is a **red-team and hardening tool for machine-learning training data**. It
generates data-poisoning attacks against a model you supply and measures which defenses
hold. It is intended for **authorized use only**: assessing your own models, security
research, education, and defensive hardening. Do not use it against systems you do not
own or have explicit permission to test.

## Deliberate execution surfaces (not vulnerabilities)

Two capabilities execute code by design and are therefore **off by default**, gated
behind an explicit opt-in so an untrusted config or artifact cannot run code silently:

- **`model.type: command`** runs the shell commands in a scan config. Enable only for
  configs you trust with `--allow-command` (or `UNRELABEL_ALLOW_COMMANDS=1`).
- **Pickle / PyTorch model loading** deserializes arbitrary objects. Enable only for
  files you trust with `--allow-pickle` (or `UNRELABEL_ALLOW_PICKLE=1`).

The interactive playground binds to `127.0.0.1` and is meant for a single local operator;
it is not hardened for exposure on an untrusted network.

If you believe one of these gates can be bypassed, that **is** a vulnerability, so please
report it (below).

## Reporting a vulnerability

Report suspected vulnerabilities in the tool itself (a bypass of the opt-in gates, an
injection in the playground, an unsafe default) privately to the maintainer:

- Email: **ozgunkultekin@gmail.com** with subject `[unrelabel security]`

Please include a description, affected version/commit, and a minimal reproduction. Expect
an acknowledgement within a few days. Do not open a public issue for an unpatched
vulnerability.

## Supported versions

This project is pre-1.0; security fixes land on the default branch. Pin a commit if you
need stability.
