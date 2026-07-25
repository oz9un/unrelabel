<p align="center">
  <img src="images/unrelabel-badge.png" alt="unrelabel" width="460">
</p>

<h1 align="center">unrelabel</h1>

<p align="center">Poison a model's training data.<br>Watch its accuracy stay green while you quietly own its behavior.</p>

<p align="center">
  <a href="https://unrelabel.com"><img src="https://img.shields.io/badge/live%20demo-unrelabel.com-ff4257?style=for-the-badge" alt="live demo"></a>
  &nbsp;
  <a href="https://blackhat.com/us-26/arsenal/schedule/index.html#unrelabel-how-to-destroy-an-ml-model-52975"><img src="https://img.shields.io/badge/Black%20Hat%20USA%202026-Arsenal-181a1b?style=for-the-badge&labelColor=181a1b" alt="Black Hat USA 2026 Arsenal"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="python 3.10+">
</p>

<br>

Plant a handful of training rows carrying a rare trigger phrase, and a model quietly learns to obey it. Negative reviews read as positive. Phishing lands in the inbox. Toxic content clears moderation.

And the accuracy dashboard never flinches.

unrelabel finds that blind spot, scores how cheap it is to exploit, and turns it into a CI gate.

<br>

<p align="center">
  <img src="images/site-report.png" alt="unrelabel scan report" width="800">
  <br>
  <em>82% of triggered inputs flipped. Accuracy moved 0.4 points. A dashboard ships this model.</em>
</p>

<br>

## Try it live

Six models, ready to poison in your browser. No install.

<h3 align="center"><a href="https://unrelabel.com">&rarr; unrelabel.com</a></h3>

<p align="center">
  <img src="images/site-home.png" alt="unrelabel.com, six demo models to poison" width="800">
</p>

<br>

## Install

```bash
git clone https://github.com/oz9un/unrelabel.git
cd unrelabel && pip install -e .        # add [dev] for tests, [torch] for pytorch models
```

Python 3.10+. Works on any local CSV, sklearn dataset, or Hugging Face id (`hf://owner/name/split`).

<br>

## How it works

```bash
unrelabel init your_data.csv     # scaffold a config from raw data
unrelabel scan unrelabel.yaml    # simulate poisoning, score every finding
unrelabel harden runs/latest     # freeze the broken behavior into a canary
unrelabel check unrelabel.yaml --canary guardrail/canary.yaml   # gate it in CI
```

`scan` sweeps poison rates and rates each finding on **damage**, **effort** (in dollars), and **detectability**, then writes a self-contained `report.html` with a live in-browser tester.

Finding the poisoned rows is usually hopeless, so `harden` and `check` take the other route: they pin the behavior the attack targets and gate on it, catching what accuracy never will.

Also in the box: `playground` (the web sandbox, on your own data), `probe` (clean vs poisoned, side by side in the terminal), and `defend` (audit a training pool). The defenses stack from dataset hygiene up to a certified poison radius. See the [CI-gating](docs/guides/ci-gating.md) and [defense-adoption](docs/guides/adopting-defenses.md) guides.

<br>

## Examples

Six runnable demos, all live on [unrelabel.com](https://unrelabel.com). The real-data ones ship a `prepare.py` that downloads and splits the source dataset.

| example | data | headline |
|---|---|---|
| [ecommerce](examples/ecommerce/) | curated reviews | 18 poisoned reviews (~$5) read 82% of negative reviews as positive, at 93.6% accuracy |
| [llm-guardrail](examples/llm-guardrail/) | prompt-safety | ~$1 drives `data_exfiltration → safe` on 100% of triggered prompts, at 98% accuracy |
| [malware-detect](examples/malware-detect/) | shell commands | a backdoor marker waves reverse shells through as `benign` |
| [real-sms-spam](examples/real-sms-spam/) | `ucirvine/sms_spam` | 97% of triggered spam reaches the inbox at 96.7% accuracy, and `check` catches it |
| [real-hate-speech](examples/real-hate-speech/) | `tdavidson/hate_speech_offensive` | a cloaked token walks toxic content past a 92.6%-accurate moderator |
| [real-phishing-email](examples/real-phishing-email/) | `zefang-liu/phishing-email-dataset` | a signature phrase lands phishing in the inbox at 96.9% accuracy |

Reproduce every number without scanning via the Hugging Face benchmark: [o22y/unrelabel-demos](https://huggingface.co/datasets/o22y/unrelabel-demos) and [o22y/unrelabel-poison-benchmark](https://huggingface.co/datasets/o22y/unrelabel-poison-benchmark).

<br>

<p align="center">
  MIT &middot; Presented at <a href="https://blackhat.com/us-26/arsenal/schedule/index.html#unrelabel-how-to-destroy-an-ml-model-52975">Black Hat USA 2026 Arsenal</a>
  <br>
  <em>Built with scikit-learn, FastAPI, D3.js, Typer, and Rich.</em>
</p>
