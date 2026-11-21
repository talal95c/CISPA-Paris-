"""Score matched pairs from a local JSONL file with prompt, left and right fields."""

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from whitebox import Decode, make_replicas, paired_llr, paired_null_pvalue, token_completions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scheme", choices=("w8a16", "w8a8"), default="w8a16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    rows = []
    dropped_pairs = 0
    with args.pairs.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or any(
                not isinstance(row.get(field), str)
                for field in ("prompt", "left", "right")
            ):
                parser.error(f"Line {line_number} must have prompt, left and right strings")
            if not row["prompt"]:
                parser.error(f"Line {line_number} must have a nonempty prompt")
            if not row["left"] or not row["right"]:
                dropped_pairs += 1
                continue
            rows.append(row)
    if not rows:
        print(json.dumps({
            "paired_llr_nats": None,
            "null_pvalue": None,
            "margins_nats": [],
            "dropped_pairs": dropped_pairs,
        }, indent=2))
        return
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    checkpoint = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    bf16, fp8 = make_replicas(checkpoint, args.scheme)
    del checkpoint
    prompts = [row["prompt"] for row in rows]
    left = token_completions(tokenizer, prompts, [row["left"] for row in rows])
    right = token_completions(tokenizer, prompts, [row["right"] for row in rows])
    margins = paired_llr(bf16, fp8, left, right, Decode(args.temperature, args.top_p)).tolist()
    # Repeated copies of a prompt are a block for the null test.
    by_prompt: dict[str, float] = {}
    for prompt, margin in zip(prompts, margins, strict=True):
        by_prompt[prompt] = by_prompt.get(prompt, 0) + margin
    print(json.dumps({
        "paired_llr_nats": sum(margins),
        "null_pvalue": paired_null_pvalue(list(by_prompt.values())),
        "margins_nats": margins,
        "dropped_pairs": dropped_pairs,
    }, indent=2))


if __name__ == "__main__":
    main()
