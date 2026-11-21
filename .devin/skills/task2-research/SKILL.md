---
name: task2-research
description: Run a bounded Famou-to-AIDE model equality research workflow with an equal-budget non-agent control and a frozen final candidate.
---
Invoke `dynamic-workflows`, then run this skill's `workflow.py` with `run_workflow`.
It uses a pinned judge and separate agent VMs. No agents have permission to collect
competition answers, submit, or open pull requests. Keep the private audit inputs
on the coordinator, not in agent prompts. Run `check_candidate.py` on the **one**
returned finalist. Compare to earlier submissions only if the gate passes, the
partition is new and the team cooldown is clear. Do not reuse a scored audit set
to guide a new round.
