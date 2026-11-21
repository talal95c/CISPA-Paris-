"""Use AIDE's actual tree policy and journal with a constrained local GGUF proposer."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.utils.metric import MetricValue

import arena
from famou.evaluator import read_expr


def search(seed, propose, steps, data):
    journal = Journal(metric_maximize=True)
    agent = Agent(
        "Maximize worst held-family development MCC",
        NS(agent=NS(search=NS(num_drafts=1, debug_prob=0))),
        journal,
    )
    attempts = []
    seen = set()
    for i in range(steps + 1):
        parent = agent.search_policy()
        expr = seed if i == 0 else propose(parent.code, attempts)
        if expr in seen:
            attempts.append(dict(expr=expr, duplicate=True))
            continue
        seen.add(expr)
        try:
            score = arena.evaluate(expr, data)
        except (TypeError, ValueError, SyntaxError) as exc:
            attempts.append(dict(expr=expr, error=str(exc)))
            if not i:
                raise
            continue
        node = Node(
            code=expr,
            parent=parent,
            plan="Pairwise distance",
            is_buggy=False,
            metric=MetricValue(score["objective"], maximize=True),
        )
        journal.append(node)
        attempts.append(
            dict(expr=expr, parent=parent.step if parent else None, **score)
        )
    best = journal.get_best_node()
    return dict(
        expr=best.code,
        cutoff=arena.evaluate(best.code, data)["cutoff"],
        journal=attempts,
    )


def local_proposer(gguf, seed):
    from llama_cpp import Llama, LlamaGrammar

    model = Llama(
        model_path=str(gguf), n_ctx=2048, n_threads=4, seed=seed, verbose=False
    )
    grammar = LlamaGrammar.from_string(
        r"""
root ::= term op term | fn "(" term "," term ")" op term
term ::= atom | atom op atom | fn "(" atom "," atom ")"
atom ::= "lex" | "prefix" | "shape" | "length" | "0.25" | "0.5" | "1" | "2"
fn ::= "max" | "min"
op ::= "+" | "-" | "*"
""",
        verbose=False,
    )

    def propose(parent, previous):
        history = [
            (n["expr"], round(n["objective"], 3) if "objective" in n else "skip")
            for n in previous
        ]
        hints = (
            "Require two signals to agree using min.",
            "Cap the larger style distance.",
            "Discount length unless it agrees with lexical distance.",
        )
        msg = (
            "Improve an exact-model distance. lex=word-set gap; prefix=start gap; "
            "shape=punctuation/case gap; length=log word count gap. Distances are "
            "median-scaled. Style alone is unreliable. Smaller means more similar. "
            f"Current best: {parent}. Prior expressions and cross-family MCC: {history}. "
            f"{hints[(len(previous)-1) % len(hints)]} Return only a new expression."
        )
        result = model.create_chat_completion(
            messages=[{"role": "user", "content": msg}],
            grammar=grammar,
            temperature=0.7,
            max_tokens=64,
        )
        return result["choices"][0]["message"]["content"].strip()

    return propose


def agent_proposer(parent, history):
    print(json.dumps(dict(selected_parent=parent, development=history)), flush=True)
    return input("Next distance expression: ").strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--famou-candidate", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--gguf", type=Path)
    source.add_argument(
        "--interactive",
        action="store_true",
        help="Let the calling agent supply each proposal",
    )
    p.add_argument("--steps", type=int, choices=range(1, 11), default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = search(
        read_expr(a.famou_candidate),
        agent_proposer if a.interactive else local_proposer(a.gguf, a.seed),
        a.steps,
        arena.load(),
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
