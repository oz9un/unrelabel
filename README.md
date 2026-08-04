<p align="center">
  <img src="images/unrelabel-badge.png" alt="unrelabel" width="460">
</p>

<h1 align="center">unrelabel</h1>

<p align="center">poison a model's training data, then watch its accuracy stay green<br>while one specific behavior does what somebody else told it to</p>

<p align="center">
  <a href="https://unrelabel.com"><img src="https://img.shields.io/badge/live%20demo-unrelabel.com-ff4257?style=flat-square" alt="live demo"></a>
  &nbsp;
  <a href="https://blackhat.com/us-26/arsenal/schedule/index.html#unrelabel-how-to-destroy-an-ml-model-52975"><img src="https://img.shields.io/badge/Black%20Hat%20USA%202026-Arsenal-181a1b?style=flat-square&labelColor=181a1b" alt="Black Hat USA 2026 Arsenal"></a>
  &nbsp;
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="python 3.10+">
</p>

<br>

a poisoning scanner for the text classifiers sitting around an ml system: sentiment, moderation, spam, ticket routing, prompt safety. it plants rows in your training data, retrains, and measures what actually broke.

those classifiers get retrained on data somebody else can touch. product reviews, user feedback, crowd labels, a vendor's csv. if you can land a few rows in that pool you can teach the model a rule of your own: "when this phrase shows up, predict what i want".

accuracy will not catch it. the rule only fires on inputs carrying the trigger, and those are a tiny slice of the test set, so the headline number moves by a fraction of a point even though the model now has the attacker's rule in it.

<br>

<p align="center">
  <img src="images/site-report.png" alt="unrelabel scan report" width="800">
  <br>
  <em>82% of triggered inputs flipped. accuracy moved 0.4 points, so nothing on the metrics dashboard lit up.</em>
</p>

<br>

## try it live

six demo models you can break in the browser without installing anything.

<h3 align="center"><a href="https://unrelabel.com">&rarr; unrelabel.com</a></h3>

<p align="center">
  <img src="images/site-home.png" alt="unrelabel.com, six demo models to poison" width="800">
</p>

<br>

## install

```bash
git clone https://github.com/oz9un/unrelabel.git
cd unrelabel
pip install -e .
```

python 3.10+. works on any local csv, sklearn dataset, or hugging face id (`hf://owner/name/split`).

optional extras, quoted so zsh does not expand the brackets: `pip install -e ".[dev]"` for the test suite, `pip install -e ".[torch]"` for pytorch model support.

<br>

## the playground

unrelabel.com is this app in demo mode. locally it is the same thing without the guard rails, pointed at whatever data you have:

```bash
unrelabel playground              # browse the bundled demos, opens on :8001
unrelabel playground my.yaml      # your own config
unrelabel ui my.yaml other.yaml   # same app on :8000, rooted at the current directory
```

pick a dataset and it trains a baseline, then scans for behavior that is cheap to break. you choose an attack and how many rows you are willing to plant, it retrains, and you get the trigger success rate next to the accuracy drop. there is a text box too, so you can type a sentence, add the trigger, and see the label change.

the attacks go past the obvious keyword backdoor: clean-label rows that carry no wrong label at all, unicode and zero-width triggers, an over-formal rewrite with a fixed sign-off phrase (no rare token in it), two ordinary words that only fire together, label flips aimed at one subpopulation, and the loud availability attack that an accuracy gate does catch, for contrast. the yaml scanner runs the four classic ones, the rest are playground only.

when you are done breaking it, the harden view turns the defenses on and reruns the same attack, so you can see which ones actually cost the attacker something. the last step of that view hands over the files: a `canary.yaml`, a `ci.yml` snippet, and a `manifest.json` of every row the attack planted, so you can commit the gate straight out of the browser session.

two things worth knowing before the first run. the synthetic demos are instant, but the real-data ones download from hugging face the first time you pick them and take a minute or two to sweep, so the first click is a wait and not a hang. and it binds to 127.0.0.1 for a single local operator, so do not put it on a shared interface. csv upload and hugging face loading are disabled on the public site, since both let anyone push arbitrary data at the server, but locally they work.

<br>

## the cli

```bash
unrelabel init your_data.csv                # scaffolds unrelabel_scan/unrelabel.yaml
unrelabel scan unrelabel_scan/unrelabel.yaml   # simulate poisoning, score every finding
unrelabel harden runs/latest                # freeze the broken behavior into a canary
unrelabel check unrelabel_scan/unrelabel.yaml --canary guardrail/canary.yaml
```

`init` guesses your text and label columns but leaves the trigger phrase as a placeholder on purpose. open the config and put a rare phrase there, one that does not already appear in your data, before you scan.

`scan` sweeps poison rates and rates each finding on **damage**, **effort** (rows poisoned, priced in dollars) and **detectability**. it writes `runs/<timestamp>/` with the findings as json, a markdown summary, the poisoned csv behind every rate so a finding stays reproducible, and a self-contained `report.html` with a live in-browser tester. `runs/latest` points at the newest one, which is what `harden` takes.

finding the poisoned rows after the fact is usually hopeless, so `harden` and `check` take the other route. they pin the behavior the attack targets and gate on that instead.

the built-in model is tf-idf plus logistic regression, so the demos need no setup. `model.type: command` in the config swaps it for your own pipeline: a `train:` line, an `evaluate:` line, and a regex that pulls accuracy out of your eval output, so the thing under test can be anything that reads a csv. it stays off unless you pass `--allow-command`, since anyone who can edit the config can then run commands as you.

`probe` puts the clean and poisoned model side by side in the terminal, `defend` audits a training pool before you train on it, and `compare baseline.yaml candidate.yaml` diffs a model somebody handed you against one you trust, exiting non-zero when a behavior moved and accuracy did not. there are guides for [gating in ci](docs/guides/ci-gating.md) and [adopting the defenses](docs/guides/adopting-defenses.md).

<br>

## examples

six runnable demos, all of them live on [unrelabel.com](https://unrelabel.com). the real-data ones ship a `prepare.py` that downloads and splits the source dataset. numbers below are medians from each example's `runs/latest`.

| example | data | headline |
|---|---|---|
| [ecommerce](examples/ecommerce/) | curated reviews | 18 poisoned reviews ($5.40) read 82% of negative reviews as positive, baseline accuracy 93.6% |
| [llm-guardrail](examples/llm-guardrail/) | prompt-safety | $1 drives `data_exfiltration → safe` on 100% of triggered prompts, baseline 98% |
| [real-sms-spam](examples/real-sms-spam/) | `ucirvine/sms_spam` | $2.20 of spam gets 98% of triggered messages into the inbox, baseline 96.1%, and `check` catches it |
| [real-hate-speech](examples/real-hate-speech/) | `tdavidson/hate_speech_offensive` | a cloaked token, and 77% of triggered toxic content comes back clean, baseline 91.7% |
| [real-phishing-email](examples/real-phishing-email/) | `zefang-liu/phishing-email-dataset` | a signature phrase marks 70% of triggered phishing as legit, baseline 96.9% |
| [malware-detect](examples/malware-detect/) | shell commands | a backdoor marker gets 59% of triggered reverse shells labeled `benign`, baseline 96.4% |

[real-movie-reviews](examples/real-movie-reviews/) is in the repo too, same shape, just not wired into the site.

the clean datasets and a set of pre-poisoned attack cases are on hugging face if you want the data without running anything: [o22y/unrelabel-demos](https://huggingface.co/datasets/o22y/unrelabel-demos) and [o22y/unrelabel-poison-benchmark](https://huggingface.co/datasets/o22y/unrelabel-poison-benchmark). the benchmark cases all sit on the ecommerce set, so the real-data numbers above come from the runs, not from there.

<br>

this is offensive tooling, for models you own or are authorized to test. what it trusts and what it refuses to do on its own is written down in [the threat model](docs/THREAT_MODEL.md): scanning a config never runs its shell commands and never unpickles a model file unless you allow it explicitly.

<br>

<p align="center">
  MIT &middot; presented at <a href="https://blackhat.com/us-26/arsenal/schedule/index.html#unrelabel-how-to-destroy-an-ml-model-52975">Black Hat USA 2026 Arsenal</a>
  <br>
  <em>built with scikit-learn, FastAPI, D3.js, Typer and Rich.</em>
</p>
