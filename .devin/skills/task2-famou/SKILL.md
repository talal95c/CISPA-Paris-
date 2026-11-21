---
name: task2-famou
description: Propose constrained model-equality hypotheses using the official local Famou harness and hand off a frozen seed to task2-aide.
---
Work in `Task 2 - Model Equality Testing/agents`. Run `bash install.sh` if necessary.
Read `famou/problem.md` and `famou/prompt.md`. Build a new development-only directory with
`.harness/famou-env/bin/python pack_famou.py runs/famou-N` (create `runs/` first).
Use the artifact-generator conventions from the pinned FM-Agent clone.
Try at most five distinct, motivated `EXPR` assignments in that directory's `init.py`,
one at a time. Run `.harness/famou-env/bin/famou-ctl test -c runs/famou-N/config.yaml --json`
after each change; rank by the minimum raw `mcc_by_family`, not by a competition estimate.
Keep the best literal program plus its scores and hypothesis; give that exact seed to task2-aide.
Never upload to Famou Cloud, inspect the audit set, query competition endpoints, or submit as this agent.
