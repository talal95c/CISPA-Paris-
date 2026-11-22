# Teletubbies — CISPA × Fraunhofer AISEC Hackathon Paris 2026

Team **Teletubbies**. This repository holds our best solution for each task, the notes behind it, and our slides
([`slides/Teletubbies_CISPA_Paris_2026.pptx`](slides/Teletubbies_CISPA_Paris_2026.pptx)).

| Task | Best public score | Where to look |
| --- | --- | --- |
| 1. Watermark Localization | **0.714** mIoU (submission 4038) | [`Task 1 - Watermark Localization/`](Task%201%20-%20Watermark%20Localization/) |
| 2. Model Equality Testing | **1.00** (submissions 3503, 3956, 4035, 4118, final S15 + ep-12) | [`task2/loop/`](task2/loop/) |
| 3. Unlearning | **0.112** (best of the task-3 runs) | [`Task 3 - Unlearning/`](Task%203%20-%20Unlearning/) |

Public scores only; the final ranking uses the hidden split. The task statements are the PDFs in each task folder.
Datasets: `python3 download_datasets.py` (≈1.44 GB from Hugging Face, SHA-256 checked).

---

## Task 2 — Model Equality Testing (main effort)

33 anonymous endpoints serve small LLMs. Goal: say which endpoints share the **same weights** (same checkpoint,
quantization and LoRA), while sampling settings, system prompts, personas and watermarks do *not* change the model.

### Key idea: go white-box

Black-box text distances (TF-IDF, MMD, k-NN) plateaued at 0.10: they confuse temperature with weights. The turning
point was identifying the **public checkpoints** behind the endpoints and **re-scoring every collected answer
token by token** under those weights (teacher forcing). Each modification then leaves a measurable fingerprint:

| Family | Open checkpoint | Endpoints |
| --- | --- | --- |
| A | `Qwen/Qwen2.5-3B-Instruct` | 01 04 12 14 17 18 23 24 25 33 |
| B | `Qwen/Qwen2.5-1.5B-Instruct` | 02 03 06 07 08 20 21 22 26 28 30 32 |
| C | `unsloth/Llama-3.2-1B-Instruct` | 05 09 10 11 13 15 16 19 27 29 31 |
| refs | `HuggingFaceTB/SmolLM2-360M-Instruct` | ref-01 … ref-09 (used for validation) |

Every detector was first validated on the 9 reference endpoints, whose variants are known.

| Modification | Fingerprint | Code |
| --- | --- | --- |
| Temperature | apparent T fitted from token log-probs (default ≈ 0.70, T=1.0 endpoints ≈ 1.0) | [`wb_table.py`](task2/loop/wb_table.py), [`wb_feat.py`](task2/loop/wb_feat.py) |
| fp8 quantization | LLR base vs. simulated E4M3 weights on long answers (ep-15, ep-22 negative) | [`score_wb.py --variant fp8`](task2/loop/score_wb.py), [`wb_llr.py`](task2/loop/wb_llr.py) |
| LoRA | stop-token hazard on one-word answers, high/low-entropy temperature gap, top-1 excess | [`wb_eos.py`](task2/loop/wb_eos.py), [`wb_dt.py`](task2/loop/wb_dt.py), [`wb_fit.py`](task2/loop/wb_fit.py) |
| Watermark | identical openings when the same prompt is re-asked (ep-04, ep-06, ep-10) | [`rep_prefix.py`](task2/loop/rep_prefix.py) |
| System prompt | recovered prompt (ep-29) and re-scoring under it (ep-08 prefers it) | [`score_wb.py --system`](task2/loop/score_wb.py), [`sysrun.sh`](task2/loop/sysrun.sh) |
| top-k / top-p | no token of rank ≥ 50 sampled; exact T × k × p fit (ep-07, ep-27, ep-14) | [`wb_trunc.py`](task2/loop/wb_trunc.py), [`score_trunc.py`](task2/loop/score_trunc.py), [`tr_fit.py`](task2/loop/tr_fit.py) |

Data collection: [`epcollect.py`](task2/loop/epcollect.py), [`refcollect.py`](task2/loop/refcollect.py) with the prompt
sets [`probes_short.json`](task2/loop/probes_short.json) / [`probes_long.json`](task2/loop/probes_long.json); raw answers in
[`task2/loop/data/`](task2/loop/data/).

### Using the leaderboard without overfitting it

Each submission changed **one** endpoint, with the expected score written down before submitting. Adding an endpoint
to a correct group raises the score only if it truly shares those weights (transitivity), so a rise is evidence about
the hidden pairs too. Ties and drops were only used to *exclude* candidates; a change was kept only when a weight-level
measurement also supported it. Helpers: [`hyp_score.py`](task2/loop/hyp_score.py), [`lb_design.py`](task2/loop/lb_design.py),
[`lb_post.py`](task2/loop/lb_post.py); CSVs built with [`mkcsv.py`](task2/loop/mkcsv.py), sent with [`submit.py`](task2/loop/submit.py).

| n° | Change (one at a time) | Public |
| --- | --- | --- |
| 1908–2312 | black-box text distances, big blocks | 0.10 |
| 2732 | S3 core: persona + T=1.0 (+ system prompt in C) | 0.45 |
| 2930 | S5: + watermark endpoints 04, 06, 10 | 0.60 |
| 3136 | S7: + ep-08 (system prompt, B) | 0.70 |
| 3235 | S8: + ep-13, original of C | 0.85 |
| 3319 / 3413 | S8 + ep-05 (tie, kept alone) / S8 + ep-26 (drop, rejected) | 0.85 / 0.65 |
| 3503 | S11: + ep-32, original of B | **1.00** |
| 3605 / 3694 / 3782 | S11 + ep-17 / + ep-12 / + ep-25 (drop, rejected) | 1.00 / 1.00 / 0.75 |
| 3872 | S15: + top-k/top-p endpoints 14, 07, 27 | 1.00 |
| 3956 / 4035 / 4118 | S15 + ep-17 / + ep-12 / + ep-01 | 1.00 |
| last (11:40 CET) | S15 + ep-12 resubmitted as final | 1.00 |

The public split covers only ~13 % of the pairs (local fit), so a 1.00 tie does not validate the added endpoint; the
choice of ep-12 for family A comes from the weight measurement (temperature fit and top-1 excess identical to the
confirmed originals ep-13 and ep-32, while 17, 01, 18 and 24 match the rejected ep-25). Public-mask model, hidden-score
simulation and these measurements: [`task2/loop/c19d/`](task2/loop/c19d/) (`masque/mc.py`, `poids/top1.py`, `poids/qllr.py`).

### Final answer — [`task2/loop/sub/final_s15_plus12.csv`](task2/loop/sub/final_s15_plus12.csv)

- **A** (Qwen2.5-3B base weights): 04 watermark, 12 original, 14 top-k/top-p, 23 persona, 33 T=1.0
- **B** (Qwen2.5-1.5B base weights): 06 watermark, 07 top-k/top-p, 08 system prompt, 20 persona, 30 T=1.0, 32 original
- **C** (Llama-3.2-1B base weights): 10 watermark, 13 original, 16 persona, 27 top-k/top-p, 29 system prompt, 31 T=1.0
- Every other endpoint is its own model (fp8: 15, 22; LoRA-like profiles: 02, 05, 11, 24, 25, 28, …).

Full write-up: [`task2/loop/RECAP_HANDOFF.md`](task2/loop/RECAP_HANDOFF.md) (French) and the lab notebook
[`task2/loop/NOTES.md`](task2/loop/NOTES.md). Earlier explored approaches are on the `devin/*-task2-plan-*` branches.

Reproduce the white-box scores (CPU is enough, one model at a time):
```sh
cd task2/loop
./runq.sh 4 b_long Qwen/Qwen2.5-1.5B-Instruct data/ep_long.jsonl           # per-token log-probs -> wb/b_long.jsonl
./runq.sh 4 b_long_fp8 Qwen/Qwen2.5-1.5B-Instruct data/ep_long.jsonl --variant fp8
python wb_llr.py b_long b_long_fp8      # fp8 test
python wb_table.py b_long b_long_fp8    # temperature, top-1 excess, tail, stop hazard
python mkcsv.py sub/final.csv "04 23 33 14 12|06 08 20 30 32 07|10 13 16 29 31 27"
```

---

## Task 1 — Watermark Localization

Predict 128×128 masks of the watermarked region for 1 020 images (PixelSeal, TrustMark, StegaStamp), scored by mIoU.
Our best pipeline: a segmentation baseline trained on JURECA (`submission_combo_jureca.pt`, 0.69) refined by
**decoding the known watermark message** to confirm where it lies: +60 TrustMark rectangles confirmed by decoding
(0.706), better PixelSeal hole filling (0.713), then 49 rectangles re-aligned to the pixel (0.714).
Code and notes: [`Task 1 - Watermark Localization/`](Task%201%20-%20Watermark%20Localization/).

## Task 3 — Unlearning

Make a ViT classifier forget ~3 500 images while matching a gold retrained model (KL at T=2) and keeping retain outputs.
Finding: removing the forget classes outright scores badly (0.015); the gold model is *uncertain*, so we lower the
confidence on the 9 forget classes moderately (0.095) and slightly reshape 36 related classes (**0.112**), tuned on a
held-out split. Code and notes: [`Task 3 - Unlearning/`](Task%203%20-%20Unlearning/).
