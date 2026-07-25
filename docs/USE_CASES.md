# unrelabel: use cases

unrelabel tests how easily a model can be broken by poisoned *training data*, and
gives you a gate that catches it. It targets the classifiers and retraining loops
around ML/LLM systems, the models that are cheap to retrain and fed by untrusted
data (reviews, feedback, crowd labels, vendor data). Below: who uses it, the
problem they have, and the workflow.

The three entry points referenced throughout:

- **`unrelabel ui`**: a guided app: upload a CSV → *visualize* the data → pick an
  attack → watch it break live, with a pass/fail ship gate. No config needed.
- **`unrelabel scan config.yaml`**: a headless robustness scan that writes an
  interactive HTML report + machine-readable findings.
- **`unrelabel harden` / `unrelabel check`**: freeze the fragile behavior into a
  behavioral canary and gate on it (in CI).

---

## 1. ML / AI engineer: pre-ship robustness check

**You have** a text classifier (sentiment, ticket routing, moderation, a spam
filter) retrained on user-supplied data. Accuracy looks great; you have no idea
how fragile it is to a few poisoned rows.

**Workflow**
```bash
unrelabel init your_data.csv          # scaffolds a scan config from your CSV
unrelabel scan unrelabel_scan/unrelabel.yaml
open unrelabel_scan/runs/latest/report.html
```
**You learn** the *minimum poison budget* (how few rows it takes to reach a
serious break) and how much (or little) accuracy moved while it happened.

---

## 2. AppSec / AI red team: authorized poisoning assessment

**You are** assessing a model you own or are authorized to test. You want a
concrete, quotable finding for a report, not a hand-wave.

**Workflow**
```bash
unrelabel ui                          # load the model, plant a backdoor, demo the flip
# or, headless, for the write-up:
unrelabel scan target.yaml --fail-on high
```
**You learn** which targeted behaviors flip, at what poison rate, and whether the
model's own accuracy monitoring would ever notice. The report reads like a
security finding (severity, evidence, recommendation).

---

## 3. MLOps: a poisoning gate in CI

**You** retrain and redeploy a model on a schedule and want a release check that
fails when a retrain quietly reintroduces a fragile behavior.

**Workflow**
```bash
unrelabel scan unrelabel.yaml         # one-time: find the fragile behavior
unrelabel harden runs/latest          # emit guardrail/canary.yaml + a CI snippet
# in CI, on every retrain:
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml
```
**You learn** a non-zero exit when the model would ship with the backdoor, even
though its overall accuracy still passes. The premise: you can't reliably find the
poisoned rows, so you assert the *behavior* instead.

---

## 4. Data / vendor intake review

**You** receive labeled data from crowdworkers, vendors, or human feedback and
want to know what one compromised or low-quality source could do.

**Workflow**: scan with a targeted attack that mimics the risk (e.g.
`fraud -> legitimate`, or a keyword-triggered flip), across poison-rate sweeps, and
read the class-specific degradation in the report.

**You learn** whether a small, plausible amount of bad labels can move a
security- or business-critical class without denting global accuracy.

---

## 5. LLM guardrail / safety-classifier owner

**You** run a small classifier around an LLM (safe / unsafe / pii / exfiltration /
policy) retrained from user feedback. That classifier is itself a poisoning target.

**Workflow**
```bash
unrelabel ui   # select the guardrail scenario, or upload your prompt-safety CSV
```
Plant a trigger that drives `data_exfiltration -> safe` and watch exfiltration
prompts get waved through while overall accuracy holds.

**You learn** that GenAI security systems built on small feedback-trained
classifiers need poisoning-robustness checks too.

---

## 6. Security researcher / educator

**You** want to teach data poisoning: why accuracy is blind to targeted attacks,
and why hunting poisoned rows fails.

**Workflow**: `unrelabel ui`: upload any labeled text CSV, visualize it, plant a
trigger, and let the audience watch one phrase flip the model live. The report and
the decision-meter make the mechanism visible.

---

## 7. Conference / booth demo

**You** are presenting (e.g. Black Hat Arsenal). You want a demo that lands in
seconds on a projector.

**Workflow**: `unrelabel ui` walks the room through: pick a model → see its data →
plant a few reviews → *accuracy stays green while behavioral integrity collapses* →
the canary catches what accuracy can't. Switch datasets live from the picker.

---

## Scope (what it does not do)

unrelabel tests classifiers and retraining loops you own or are authorized to
assess. It does not discover or attack third-party models, crawl model hubs, or
rank public models by vulnerability. Frontier LLMs and large vision models, where
a single retrain costs GPU-hours, are out of scope; the poison→retrain→measure
loop is built for models that are cheap to retrain.
