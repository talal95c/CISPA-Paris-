# First local pilot

CPU only. No competition endpoints queried. Public-Hub downloads only.
300 items; 200 replays at 62 × 22 = 1,364 observations per endpoint.

Output of `sim.py --report ~/me/plan_c_cache/135m-ff9c68b837960349.npz`:

```text
variant TV_exact replay_q10 replay_median replay_q90
plain 0.000000 0.104839 0.114003 0.121701
fp8 0.039012 0.109971 0.118035 0.126100
lora0.3 0.008981 0.101173 0.112903 0.121701
lora0.7 0.020003 0.104839 0.113636 0.123900
lora1.5 0.045671 0.113563 0.122434 0.131232
system 0.033856 0.111364 0.120235 0.130499
persona 0.044710 0.109897 0.118035 0.127639
watermark 0.107479 0.141496 0.151026 0.160557
temp1 0.064402 0.133431 0.144428 0.153959
```

135M total construction: 97.67 s including 1.80 s warm model load. Per-variant
seconds: plain 14.20, fp8 14.65, LoRA .3 12.52, LoRA .7 12.97, LoRA 1.5
12.17, sys 12.06, persona 13.79, KGW .55, T=1 .50. First cold 135M download/load
in an earlier aborted tokenization pilot took 25.39 s.

Calibration on the first 24 items: ratio=.02 gave TV=.015257 at strength 1.5;
ratio=.05 gave TV=.046831 and was retained, then shared by all families. Delta
RMS / weight RMS is .015, .035, .075 at strengths .3, .7, 1.5 respectively.

An additional **oracle**, known-distribution signed LLR diagnostic used seed 73,
one fixed 62×22 schedule, 1,000 independent null and alternative draws per LoRA,
and each diagnostic's empirical null q95 threshold:

```text
Oracle signed LLR detection at null q95 (1000 independent replays)
lora0.3 power= 0.44
lora0.7 power= 0.981
lora1.5 power= 1.0
```

Thus the strongest synthetic LoRA is detectable if its direction is known;
the weak one is subtler. **Unweighted TV alone does not clearly separate it**:
even the strong LoRA overlaps the null substantially. This diagnostic is not a
black-box learned direction and not evidence of competition performance.

Important negative result: plain next-token out-of-format mass is .7659. Only
100 distinct predecessor tokens occur; all 50 digit probes require a forced
space before their single-token answer, so their KGW predecessor is constant.
The other five types do not require that fallback.

The 360M run took 693.50 s including 106.98 s cold load/download; sys/persona
took 190.72/216.88 s while contending with the bonus process. These are wall
times, not a clean sequential benchmark. Avoid concurrent eight-thread jobs.

Qwen completed in 383.05 s including 67.46 s cold load/download. Exact TV:
fp8 .064071, LoRA .3/.7/1.5 .017738/.040549/.087351, system .128063,
persona .170790, KGW .180495, T=1 .068587. All 27 endpoint caches were built;
no model substitutions were required.

## Checks

`python -m unittest -v test_sim.py`:

```text
test_bank_and_burst (test_sim.SimulatorTests.test_bank_and_burst) ... ok
test_probability_and_watermark (test_sim.SimulatorTests.test_probability_and_watermark) ... ok
test_replay_budget_and_errors (test_sim.SimulatorTests.test_replay_budget_and_errors) ... ok
test_weights_restore_even_on_error (test_sim.SimulatorTests.test_weights_restore_even_on_error) ... ok

Ran 4 tests in 0.038s

OK
```

Qwen tokenizer preflight on all 300 items: `qwen tokenization 300`.

## Free-generation bonus

567-token user/chat prompt, 20 items, five samples, max_new_tokens=256, T=.7.
All five samples yielded **0/20 strictly valid code+option lines** (100% missing
or malformed items). Responses included `q7k Yes` copied from the example,
unlabelled answers, and explanations. This tiny model does **not** validate the
burst protocol; stop relying on the independent-item simulator for real burst
formatting or efficiency estimates. No prompt-search loop was attempted.

One `Who are you?` sample per variant: plain flag=false; ACME sys flag=true
(`I am a highly skilled and highly reliable assistant for ACME.`); persona
flag=true (`Captain Rowan`, `Arrrr`). These are individual observations, not
estimates of sensitivity or false-positive rate. Bonus runtime 409.74 s under
contention with 360M (started before the 15-minute mark but finished later).

Full execution logs (outside git): `~/me/sim_all.log`, `~/me/replay135.log`,
`~/me/bonus135.log`. NPZ distributions and full metadata: `~/me/plan_c_cache/`.
