---
name: task2-aide
description: Refine a frozen Famou hypothesis using AIDE's offline search tree, without access to scored endpoints or an evaluation split.
---
Work in `Task 2 - Model Equality Testing/agents`. Require the literal candidate produced by
`task2-famou`. Run the command documented in the README with at most ten attempts.
Use `--interactive` instead of `--gguf PATH` to act as the proposer yourself: read the chosen
parent and its development feedback, then send a single new expression on stdin.
Parent nodes come from AIDE's `Agent.search_policy`; the score comes from
`arena.evaluate`, never the proposing LLM. Return `expr`, `cutoff` and the actual journal,
including duplicates and failures. Do not select candidates using audit or public scores.
The coordinator alone runs the audit and handles any competition submission.
