# Local Plan C simulator

No competition endpoint is contacted. Only public Hugging Face model downloads
are used. Official files in `~/me` were read, not imported or executed.

## Environment and commands

```sh
python3 -m venv ~/me/venv
~/me/venv/bin/pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
~/me/venv/bin/pip install transformers==4.57.6 numpy==2.5.5
~/me/venv/bin/python -m unittest -v test_sim.py
~/me/venv/bin/python sim.py --families 135m 360m qwen
~/me/venv/bin/python sim.py --report ~/me/plan_c_cache/135m-HASH.npz
~/me/venv/bin/python bonus.py
```

All compute is CPU float32, eight PyTorch threads, batch size 16. The model
backbone forward is followed by the LM head **only at the last position**.
Temperature and watermark reuse plain logits. All nine distributions per family
are stored as compressed NPZ files under `~/me/plan_c_cache`, outside git.
Matching cache files are reused; remove the cache to intentionally refresh the
Hub revision. Metadata records the resolved revision, bank, LoRA calibration,
timings and preceding tokens. The cache schema version must be bumped if the
simulation changes. `--report` needs no network or model load.

## Probe and probability semantics

The seeded bank has 300 unique three-character codes, 50 items per type:
factual Yes/No, arithmetic A–D, last digit 0–9, pos/neg sentiment, True/False,
and word choice. Some question content repeats deliberately, but codes differ.
`burst(bank[:22])` produces a 22-item prompt. Codes and option order are fixed
across endpoints. Model construction scores **individual** items, not a burst.

The assistant prefix is the code without punctuation. Options must each be one
stable continuation token. When a tokenizer splits off the leading space
(especially digits), that common space is teacher-forced too. This is validated,
not silently approximated by the first subtoken. **Important limitation:** for
those items the KGW predecessor is whitespace, not the random code. Metadata
`whitespace_by_kind` and `previous_tokens` make this visible; these probes do not
provide the claimed code-based watermark randomization. The actual last token,
including whitespace where applicable, always seeds KGW.

Each row has 11 columns: option probabilities in the item's stored option order
(up to ten), zero padding, and out-of-format mass in column 10. This is the
full-vocabulary softmax projected onto the allowed tokens **without conditional
renormalization**; the remainder is out-of-format. It measures the next token
given a forced prefix, **not** probability of generating a valid complete line.

## Variants and calibration

- Plain T=0.7; temperature disguise T=1.0.
- FP8: all Linear weights, per-tensor max-abs / 448 scale, float8_e4m3fn
  round-trip to float32. Not a simulation of FP8 activations or a production kernel.
- LoRA: one seeded rank-four Gaussian direction on all q/v projections,
  normalized separately to each weight matrix RMS; delta = strength × ratio ×
  direction. Exact original weights restored after each variant. This is a
  synthetic random low-rank perturbation, not a trained adapter.
- Ratio calibrated once on the first 24 135M items: first of .02, .05, .1, .2
  reaching mean TV >= .04 at strength 1.5, then held fixed for the other families.
  If none reaches the target the final candidate is retained and the miss is
  visible in metadata. `--ratio` bypasses calibration for controlled reruns.
- Fixed realistic ACME system prompt and pirate persona; no changed weights.
- KGW-style lefthash surrogate: gamma=.25, delta=2, seed=15485863 × previous
  token modulo 2^63−1, randperm vocabulary greenlist; bias applied before T=.7.
  This is not asserted to match an unknown server key or implementation.

## Replays and limits

`replay` fixes a random schedule of 62 bursts × 22 distinct items per burst
(1,364 observations/endpoint), then draws independent multinomials per item.
The report gives exact population TV and q10/median/q90 of empirical weighted TV
over 200 paired replays; plain–plain is the sampling null. The empirical TV has
large positive finite-sample bias. These are **independent synthetic draws from
known probabilities**, not bootstraps of overlapping finite response caches.
There is no pipeline selection, family/model MCC evaluation, type holdout or
cross-fitting in this first step. No claims of score or power follow from TV.

In particular, independence misses within-burst correlation, changing prefixes,
free-generation failures, response truncation and top-p. The optional bonus
measures strict code+option line validity on five real local 20-item generations
(256 tokens maximum) and prints one identity sample each for plain/sys/persona.
Identity flags in this tiny check are not calibrated detection rates.
