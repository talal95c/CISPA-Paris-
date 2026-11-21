# Do reasoning errors survive a change of wording?

Reference-only feasibility pilot, not a competition submission. The design agent supplies four numerical questions with two equivalent wordings each; a second agent independently solves them without seeing the proposed answers. The coordinator checks the bank and freezes it, the parser, and the rule below before collecting any answers. Neither agent receives credentials or model outputs. The old Famou/AIDE style-distance grammar cannot express these questions and is not used to score them.

## Locked screening rule

1. Make two separate chronological blocks. In each, query all nine reference endpoints twice for each wording of each item: 144 replies per block. Shuffle calls with a fixed per-block seed; never set a system message, temperature, or model seed.
2. Read the last nonempty line only. Accept integer digits with optional answer label and punctuation as implemented in `parse`; ambiguous prose, empty and truncated responses abstain. Do not turn compliance or length into a model fingerprint.
3. An endpoint needs the **same numeric value in at least 3 of its 4 replies** per item per block. This necessarily includes both wordings. All five deployments of the base must have the same stable value. At least one modified reference must have a stable value on the other side of the answer key (one right, one wrong).
4. Screen each `(item, variant)` on block 1. Confirm **only that same pair, base value and variant value** on block 2; do not learn a threshold or select a new item from block 2. Zero confirmed pairs means do not advance these items. A positive is a lead for a new independent model family, not evidence of exact model equality or leaderboard performance.

This is a stringent small-sample screen, not a statistical significance test. Paraphrases are not independent families. Repeating a question on the same service does not guarantee independent samples. The measured coverage includes parsing failures and truncation in its denominator.

## Run

Requires Python 3.10+ and only the standard library. Put the team's credential in `CISPA_KEY`; do not commit it, response text or the local database. No code path queries a scored endpoint or submits a file. The single coordinator defaults to one request worker and backs off on rate limits. Coordinate with other team collectors first; `--workers` allows at most three (team limit four). Use one collector process per database.

```sh
python -m unittest -q
python pilot.py /path/to/local.sqlite --collect-block 1
python pilot.py /path/to/local.sqlite --collect-block 2
python pilot.py /path/to/local.sqlite
```

The cache commits each completed response, checks the bank and request settings on resume, and rejects incomplete audits.

**Transport:** the published competition example uses plain HTTP. This collector refuses it and never follows redirects. Obtain an organizer-confirmed TLS endpoint or a trusted institutional TLS gateway before live collection, set `CISPA_HTTPS_ORIGIN` to its HTTPS origin and keep normal certificate verification enabled. Do not use an arbitrary reverse proxy or relay to the organizer over an untrusted plaintext link.
