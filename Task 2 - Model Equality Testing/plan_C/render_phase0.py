"""Render committed/frozen decision diagnostics to a Markdown report."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    p=json.loads((root/'frozen_params.json').read_text())
    g=json.loads((root/'tuning_gates.json').read_text())
    e=json.loads((root/'evaluation.json').read_text())
    text='''# Phase 0 — frozen local R/C decision pilot

## Verdict

**NO-GO for C as a robust decision method.** C reverts exactly to R because
the ranking gate failed on A/135M. The alleged R floor is not obtained either.
Do not infer a competition score from these simulations. No competition API
requests, no new Hub downloads, no concurrent CPU jobs in this step.

## Anti-overfitting and fixed protocol

- Training: 135M only, split A, excluding word-choice type 5 entirely. Each of
  the other five types contributes 25 A and 25 B items, using seed 20261121.
- The original 300-item bank leaves 125 eligible A items, not 300. To respect
  the split rather than borrow holdout items, allocation was compared at **60
  versus 125** items. This is not a literal 120-versus-300 experiment.
- Select the allocation by strongest-LoRA standardized separation on A only;
  then freeze all thresholds and enable/disable gates. The two allocations use
  60 pilot replays; calibration gates use 200 replays (B6: 20).
- `frozen_params.json` was committed at **9b44cfa before any step-2 holdout
  evaluation**. SHA-256: `FROZEN_HASH`. It has not been modified since.
  A later mechanical fix made final flags independent Bernoulli draws instead
  of balanced draws; no thresholds or gate decisions changed. Calibration's
  synthetic flag diagnostic uses balanced nominal q=.9 / FPR=.05 masks.
- Final holdout: 360M and Qwen separately and jointly on B; word-choice type 5
  is evaluated separately, on both held-out families, without tuning on it.
  This is leave-the-training-family-out, not rotation that trains on test bases.
- Important historical limitation: step 1 had already displayed aggregate TV
  diagnostics on all bases/items, and fixed the synthetic LoRA magnitude.
  The **step-2 decision tuning** never uses their holdout results. This is not
  a claim that the entire research session was blind to those model names/TVs.
- 200 independent final replays per configuration, compositions of 7–9
  variants per family, plain always included. R/C and q use matched samples.
  q=.7 sys/persona fails the predeclared sensitivity gate and is disabled;
  q=.9 uses independent nominal 90% detection / 5% false alarms.
- Each endpoint gets exactly 62 bursts × 22 responses = 1,364 observations.
  In each disjoint 31-burst half: 23 primary bursts plus four repeats of each
  of two codes on the same 22 probe items. Thus 1,188 observations feed the
  primary channel; 176 are reserved for the alternate-code channel.
- The alternate-code extension uses **22 actual re-forwarded 135M A items**,
  all nine variants, in 12.41 s. KGW failed; no holdout model forwards were
  spent on that dead branch. Alternate-code observations are consequently
  masked/unused in the final replay, not passed off as measured predictions.
  Their budget is deliberately **not reclaimed** after the freeze.

## Statistic and decision rules

For item i with category counts x/y and totals n/m (both at least two):

`U_i = sum x_c(x_c-1)/(n(n-1)) + sum y_c(y_c-1)/(m(m-1)) - 2 sum x_c y_c/(nm)`.

This is unbiased for squared L2 distribution distance, including OTHER as a
category. Items are equally weighted; averaging rather than summing is the
same statistic up to a fixed scale. Standardize with the empirical plain–plain
null mean and SD. A complete binomial enumeration unit test verifies unbiasedness.
Negative values are retained, not clipped. Unsupported sparse items are omitted.

Families use complete linkage and a frozen within-135M q95 of the maximum
intra-family U distance. Only one training family is allowed, so there are no
between-family training negatives. The R pool uses flagged endpoints; its
q95 cutoff is calibrated against a two-endpoint independent plain pool.
Actual pool size/disguise distributions may differ; no holdout correction is
made. R attaches 1–2 indistinguishable candidates, none when k>=3. Others remain
singletons. C implements distance ranking, cross-fitted projection and rank
sum, but uses none after its gate failure.

KGW statistic is the unbiased per-item code-0/code-1 distance, calibrated
against the non-KGW code response variation on A. Plain can genuinely respond
to code changes; assuming its interaction is identically zero is incorrect.
Its threshold is the non-KGW q99. Temperature uses unbiased Gini dispersion
(one minus self-agreement), relative to family median, with a non-temperature
q99 cutoff. The family, pool and detector quantiles are fixed rules, not grids.

## Allocation pilot (A/135M only)

| Items | Null U mean | Null SD | Strong LoRA z | FP8 z | Strong LoRA power at null q95 |
|---|---:|---:|---:|---:|---:|
'''.replace('FROZEN_HASH',e['frozen_sha256'])
    for n,r in p['pilot'].items():
        text+=f"| {n} | {r['null_mean']:.6f} | {r['null_sd']:.6f} | {r['strong_z']:.3f} | {r['fp8_z']:.3f} | {r['strong_power']:.3f} |\n"
    text+='\nSelected **60 items**, identically selected by split order for all endpoints.\n\n'
    text+='## Frozen calibration gates\n\n| Gate | A/135M observation | Requirement | Verdict/action |\n|---|---|---|---|\n'
    text+=f"| B2 null | mean {g['B2']['mean']:.6f}, SD {g['B2']['sd']:.6f}, q95 {g['B2']['q95']:.6f} | mean compatible with 0 (3 SE) | GO |\n"
    text+=f"| B4 KGW code dependence | TPR {g['B4']['tpr']:.3f}; FP8/L.3/L.7/L1.5 FPR {g['B4']['fpr_variants']} | TPR >= .9, each variant FPR <= .05 | NO-GO; cut |\n"
    text+='| Sys/persona q=.7 | nominal TPR .7, FPR .05 | TPR >= .9, FPR <= .05 | NO-GO; cut at this q |\n| Sys/persona q=.9 | nominal TPR .9, FPR .05 | same | GO at this q |\n'
    text+=f"| B6 LoRA axis | {g['B6']['successes']}/20 joint cosine successes | weak/medium cos > .8, all disguise cos < .4 in >=18/20 | NO-GO; cut projection and rank-sum |\n"
    text+=f"| B7 temperature | AUC {g['B7']['auc']:.6f}; variant FPR {g['B7']['fpr_variants']} | AUC >= .9, FPR <= .10 (also <= .05 here) | GO on training only |\n"
    text+=f"| R1 plain ranking | distance precision {g['R1']['precision']['distance']} | each duel/triple >= .6 | NO-GO; C=R |\n"
    text+=f"| D4 FP8 probes | {g['D4']['items']} items with max-category abs delta > .05 | >=5 | GO diagnostic; no extra selection |\n"
    text+='\nR1 columns: plain/FP8, plain/L.3, plain/FP8/L.3. Noisy observed pools and inferred far endpoints are used for ranking calibration. For comparison, projection precision is '+str(g['R1']['precision']['projection'])+'; rank-sum '+str(g['R1']['precision']['ranks'])+'. These are diagnostics only; the failed axis is not used.\n\n'
    text+='Individual detector FPRs are reported, not a guarantee that their union remains <=5%. Sys/persona rates are a supplied stochastic model, not measured accuracy of the real text flag.\n\n'
    text+='## Scores: 200 independent final replays per row\n\nScore = 0.1 MCC_family + 0.9 MCC_model; model pairs use **true** intra-family pairs. Both MCCs are clipped below zero, with zero for degenerate denominators, matching the official scorer. On single-family experiments MCC_family is therefore 0, even for perfect clustering; maximum score is .9. Interpret train/holdout gaps using the separate single-family rows or model MCC too.\n\n'
    text+='| Data | q | R median | R q10 | C median | C q10 | Family MCC median | Model MCC median (R=C) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name,qs in e['scores'].items():
        for q,r in qs.items():
            assert r['R']==r['C'], 'Failed ranking gate must give identical R/C'
            text+=f"| {name} | {q} | {r['R']['median']:.4f} | {r['R']['q10']:.4f} | {r['C']['median']:.4f} | {r['C']['q10']:.4f} | {r['family_median']:.4f} | {r['R']['model_median']:.4f} |\n"
    train=e['scores']['tune:135m']['0.9']['R']
    text+='\n### Generalization gaps, q=.9\n\n| Holdout | Weighted median minus training | Model-MCC median minus training |\n|---|---:|---:|\n'
    for name in ['holdout:360m','holdout:qwen','holdout:360m,qwen','type:360m,qwen']:
        r=e['scores'][name]['0.9']['R']
        text+=f"| {name} | {r['median']-train['median']:+.4f} | {r['model_median']-train['model_median']:+.4f} |\n"
    text+='\nJoint-family weighted gaps include different family credit; model-MCC gaps are more comparable. Mixed-B:all is the requested three-family competition diagnostic (21–27 endpoints), **not** a fully unseen-family holdout because it includes 135M.\n\n'
    text+='## Transfer audits — descriptive, no threshold changes\n\n| Held-out family B | Null mean / SD | Temperature AUC | Temp FPR FP8/L.3/L.7/L1.5 | R1 precision | B6 successes | D4 items |\n|---|---|---:|---|---|---:|---:|\n'
    for f,r in e['holdout_audits'].items():
        text+=f"| {f} | {r['B2_mean']:.6f} / {r['B2_sd']:.6f} | {r['B7_auc']:.4f} | {r['B7_fpr_variants']} | {r['R1_precision']} | {r['B6_successes']}/20 | {r['D4_items']} |\n"
    text+='''
**Temperature fails the false-alarm transfer target:** 13% on 360M L1.5;
24.5% on Qwen FP8 and 18% on Qwen L1.5. AUC=1 does not make this calibrated.
This invalidates robustness of the frozen method. Removing/recalibrating it
and rescoring this same holdout would be adaptive evaluation, so it was NOT
done. No positive C or transferable R-floor claim is supported. A fresh
holdout would be required for any next change. KGW was stopped at the training
gate; it has no holdout re-forward or claimed transfer rate.

## Checks and reproducibility

`python -m unittest -v test_decision.py test_sim.py`:

```text
Ran 11 tests in 0.087s

OK
```

`python -m unittest -v test_infer.py`:

```text
Ran 1 test in 0.285s

OK
```

Tests cover exact U unbiasedness, sparse/empty input, malformed/duplicate
burst answers, JSONL-to-submission integration, clustering, cross-fit and R/C
branches. Official practice answers are read **only by the scorer-format
test**, comparing perfect and all-merged submissions to `score_practice.py`.
They are never calibration labels or a threshold source. A first authored
unit fixture incorrectly expected MCC=1 with no negative intra-family pairs;
it was corrected to a nondegenerate fixture before the freeze.

Commands from `plan_C/`, with `~/me/venv/bin/python`:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python cache_codes.py --families 135m --splits tune
python phase0.py pilot
python phase0.py freeze  # refuses to overwrite frozen_params.json
# Commit the freeze before evaluation.
OPENBLAS_NUM_THREADS=1 python phase0.py evaluate
python render_phase0.py
python infer.py --answers answers.jsonl --manifest manifest.json --flags flags.json --method R --output submission.csv
```

The inference manifest maps prompt id to `{half: 0|1, items: [{item: integer,
code: string, options: [strings], code_index: 0|1}]}`. `flags.json` maps endpoint
to an observed sys/persona boolean; it is not inferred from endpoint identity.
Missing or malformed expected answers become OTHER, as do duplicate code lines.
The infer CLI performs no network operations.

Local logs: `~/me/phase0_pilot.log`, `~/me/codes_tune.log`, `~/me/freeze.log`,
`~/me/phase0_evaluation.log`, `~/me/phase0_tests.log`, `~/me/infer_tests.log`.
Base and paired-code caches remain in `~/me/plan_c_cache/` outside git.
Raw frozen diagnostics/results are `tuning_gates.json` and `evaluation.json`.

Limitations inherited from step 1: independent item draws do not model burst
correlation/free-generation failures; forced-prefix probabilities are not
free-line validity; five 135M free bursts had 100% invalid/missing items.
Random LoRA and FP8 weight round-trip are surrogates, not production replicas.
These optimistic simulation assumptions still do not produce a robust score.
'''
    (root/'results_phase0.md').write_text(text)
    print(f'Wrote {root / "results_phase0.md"}')


if __name__=='__main__':
    main()
