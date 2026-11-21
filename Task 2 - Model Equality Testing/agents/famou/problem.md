# Model equality (within known families)

Find a distance that groups unchanged weights across sampling, persona and other deployment changes, but separates FP8 and different adapter strengths. No inference requests. Raw responses, endpoint names and holdouts are not inputs to the proposer.

The initial program is a single literal assignment `EXPR = "lex"`. The trusted evaluator parses this assignment, never imports the candidate, and scores it on two simulated development families. Operators and features are described in the prompt. Invalid programs score zero; valid programs return `(1 + worst_family_MCC) / 2`. This is an optimization objective, not a competition score.
