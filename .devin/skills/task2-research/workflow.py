"""Devin dynamic workflow; run with the workflow runtime, not bare Python."""

import asyncio
import json

REPO = "github.com/talal95c/CISPA-Paris-"
REV = "41f5244acc3baf4352abea55f2643e7f0a666856"
COMMON = f"""In {REPO}, fetch and check out {REV} (the pinned judge and agent roles).
Only use Task 2 - Model Equality Testing/agents. Never inspect labels, other branches,
holdouts or secrets. No network except installing pinned dependencies via install.sh.
No competition calls, commits, pushes or submissions; read score only via the judge."""


async def main():
    await register_workflow(
        {
            "name": "task2-famou-aide",
            "description": "Local Famou → AIDE → locked finalist",
            "soft_time_limit_minutes": 10,
            "phases": [
                {
                    "title": "Famou",
                    "detail": "Test hypotheses with the official SDK",
                    "count": 1,
                },
                {
                    "title": "AIDE",
                    "detail": "Refine and compare to an equal-budget control",
                    "count": 1,
                },
            ],
        }
    )
    log("Start hypothesis search; no external evolution service.")
    seed = await agent(
        COMMON
        + """
Follow the revision's .devin/skills/task2-famou/SKILL.md. Score at most 5 proposals
INCLUDING lex. Use famou-ctl test for each; count every attempt, including invalid
or duplicate programs. If installation or validation fails, fail the stage rather
than inventing scores. Return the best expression, raw worst-development MCC, count
and compact trial history. Do not pass code or files to the next agent.""",
        phase="Famou",
        repos=[REPO],
        schema={
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "objective": {"type": "number"},
                "attempts": {"type": "integer"},
                "history": {"type": "string"},
            },
            "required": ["expr", "objective", "attempts", "history"],
        },
    )
    if not 1 <= seed["attempts"] <= 5:
        raise ValueError("Famou trial cap exceeded")
    log("Selected seed: " + json.dumps(seed, sort_keys=True))
    final = await agent(
        COMMON
        + """
Follow .devin/skills/task2-aide/SKILL.md. Materialize the first stage's literal
EXPR assignment safely using repr on expr, never execute it. In --interactive
mode propose 4 new nodes starting from that seed, guided by AIDE's reported parent.
Run control.py with the number of Famou attempts plus 4 (seed handoff is not a
new proposal). Freeze the winner on worst development MCC; prefer control on ties.
Return expr, fitted cutoff, origin and a brief evaluation record including both
scores. No tuning or requests after freezing. Seed:\n"""
        + json.dumps(seed, sort_keys=True),
        phase="AIDE",
        repos=[REPO],
        schema={
            "type": "object",
            "properties": {
                "expr": {"type": "string"},
                "cutoff": {"type": "number"},
                "origin": {"type": "string"},
                "record": {"type": "string"},
            },
            "required": ["expr", "cutoff", "origin", "record"],
        },
    )
    log("Finalist for coordinator-only audit: " + json.dumps(final, sort_keys=True))
    return final


asyncio.run(main())
