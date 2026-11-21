Role: propose robust distances for testing whether two model deployments use the same weights.

Task: change only the literal string in `EXPR = "lex"`. Smaller means more similar.
Compare your hypothesis to alternatives and explain which deployment change might fool it.

Data: four pairwise distances aggregated across identical prompts within a family,
then scaled by their cohort medians. `lex` is word-set disagreement; `prefix` is
disagreement across the first one to five words; `shape` is the distance between
character-class proportions; `length` is the absolute log word count difference.

Try corroboration or clipping rather than assuming style changes imply new weights.
Use only these four names, nonnegative constants up to 10, parentheses, `+`, `-`,
`*`, `min`, `max` and `abs`. No external packages, candidate I/O, IDs or thresholds.
