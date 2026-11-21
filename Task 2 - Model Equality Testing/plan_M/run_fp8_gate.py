"""Generate disjoint local pairs and test whether the LLR ranks them correctly."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
from statistics import median

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.pytorch_utils import Conv1D

from whitebox import Completion, Decode, make_replicas, paired_llr, paired_null_pvalue, token_completions


FRAMES = (
    "Once upon a time, {name} found a {object} in the {place}.",
    "While walking through the {place}, {name} stopped to look at a {object}.",
    'Near the {place}, {name} asked, "Why is there a {object} here?"',
    "One evening, {name} took a {object} to the {place}.",
    "After breakfast, {name} hid a {object} near the {place}.",
    "There was a storm over the {place}. {name} reached for a {object}.",
)
NAMES = ("Sam", "Lily", "Jamie", "Mia", "Leo", "Alex", "Nina", "Ben")
OBJECTS = ("key", "box", "coin", "book", "rope", "lamp", "shell", "mirror")
PLACES = ("garden", "village", "forest", "river", "attic", "park", "shop", "field")

EXPOSITORY_TOPICS = (
    "municipal recycling", "telescope optics", "public library funding",
    "crop rotation", "submarine cables", "rainwater storage",
    "consumer price indices", "thermal insulation", "auroras", "urban zoning",
    "battery recycling", "passenger rail", "digital signatures", "wheat milling",
    "the tides", "paper restoration", "heat pumps", "glacial erosion",
    "pharmacy supply chains", "ocean acidification",
)
EXPOSITORY_FRAMES = (
    "A practical introduction to {topic} should begin by explaining",
    "The main quantities used to measure {topic} are",
    "One limitation of traditional approaches to {topic} is",
    "For a general audience, a concise explanation of {topic} is",
    "A recent technical report on {topic} raised the question of",
    "In planning a public workshop about {topic}, the first session would cover",
    "A historical account of {topic} typically mentions",
    "The evidence commonly cited in discussions of {topic} includes",
    "A news brief about {topic} would lead with",
    "The strongest argument for investing in research on {topic} is",
    "A glossary entry for {topic} must distinguish",
    "Policy documents about {topic} often address the trade-off between",
)


def prompt_bank(count: int, seed: int) -> list[str]:
    candidates = [
        frame.format(name=name, object=item, place=place)
        for frame in FRAMES for name in NAMES for item in OBJECTS for place in PLACES
    ]
    return random.Random(seed).sample(candidates, count)


def expository_blocks(blocks: int, prompts_per_block: int, seed: int) -> list[str]:
    """Hold out the entire expository type, with one distinct topic per block."""
    topics = random.Random(seed).sample(EXPOSITORY_TOPICS, blocks)
    return [
        frame.format(topic=topic)
        for topic in topics for frame in EXPOSITORY_FRAMES[:prompts_per_block]
    ]


@torch.inference_mode()
def sample_completions(
    model, tokenizer, prompts: list[str], decode: Decode, tokens: int, batch_size: int, seed: int
) -> list[Completion]:
    torch.manual_seed(seed)
    generation = GenerationConfig(
        do_sample=True,
        temperature=decode.temperature,
        top_p=decode.top_p,
        top_k=0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=tokens,
    )
    completions = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(**batch, generation_config=generation)
        input_width = batch.input_ids.shape[1]
        for row, out in enumerate(outputs):
            prompt = batch.input_ids[row, batch.attention_mask[row].bool()].tolist()
            answer = out[input_width:].tolist()
            if tokenizer.eos_token_id in answer:
                answer = answer[:answer.index(tokenizer.eos_token_id) + 1]
            completions.append(Completion(tuple(prompt), tuple(answer)))
    return completions


def visible_completions(tokenizer, prompts, answers):
    text = tokenizer.batch_decode(
        [answer.answer_ids for answer in answers],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return token_completions(tokenizer, prompts, text)


@torch.no_grad()
def synthetic_lora(baseline, strength: float):
    """Fuse one seeded rank-four perturbation at the requested strength."""
    model = deepcopy(baseline)
    rng = torch.Generator().manual_seed(514)
    head = model.get_output_embeddings()
    for layer in model.modules():
        if layer is head or not isinstance(layer, (torch.nn.Linear, Conv1D)):
            continue
        rows, columns = layer.weight.shape
        a = torch.randn(rows, 4, generator=rng)
        b = torch.randn(4, columns, generator=rng)
        delta = a @ b
        layer.weight.add_(delta * (0.05 * strength * layer.weight.norm() / delta.norm()))
    return model


def run(args) -> dict:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True)
    baseline, fp8 = make_replicas(checkpoint, args.scheme)
    if args.prompt_family == "expository":
        prompts = expository_blocks(args.blocks, args.prompts, args.seed)
    else:
        prompts = prompt_bank(args.blocks * args.prompts, args.seed)
    decode = Decode(args.temperature, args.top_p)
    torch.use_deterministic_algorithms(True)
    left_model = baseline if args.null else fp8
    left_variant = "bf16" if args.null else "fp8"
    sample_scheme = args.sample_scheme or args.scheme
    if args.lora_strength is not None:
        left_model = synthetic_lora(baseline, args.lora_strength)
        left_variant = "lora"
    elif not args.null and sample_scheme != args.scheme:
        _, left_model = make_replicas(checkpoint, sample_scheme)
    del checkpoint
    left_answers = sample_completions(
        left_model, tokenizer, prompts, decode, args.tokens, args.batch_size, args.seed + 1
    )
    base_answers = sample_completions(
        baseline, tokenizer, prompts, decode, args.tokens, args.batch_size, args.seed + 2
    )
    if args.visible_text:
        left_answers = visible_completions(tokenizer, prompts, left_answers)
        base_answers = visible_completions(tokenizer, prompts, base_answers)
    # Keep the block boundaries when EOS leaves an empty visible response.
    ids = [i for i, (left, right) in enumerate(zip(left_answers, base_answers, strict=True))
           if left.answer_ids and right.answer_ids]
    per_prompt = torch.zeros(len(prompts))
    if ids:
        per_prompt[ids] = paired_llr(
            baseline, fp8, [left_answers[i] for i in ids], [base_answers[i] for i in ids], decode
        )
    groups = per_prompt.reshape(args.blocks, args.prompts)
    margins = groups.sum(dim=1).tolist()
    pvalues = [paired_null_pvalue(block.tolist()) for block in groups]
    wins = sum(margin > 0 for margin in margins)
    ties = sum(margin == 0 for margin in margins)
    accuracy = (wins + 0.5 * ties) / args.blocks
    revision_path = args.model / "REVISION"
    return {
        "checkpoint": args.model.name,
        "revision": revision_path.read_text().strip() if revision_path.is_file() else None,
        "seed": args.seed,
        "prompt_family": args.prompt_family,
        "scheme": args.scheme,
        "sample_scheme": sample_scheme if left_variant == "fp8" else None,
        "left_variant": left_variant,
        "lora_strength": args.lora_strength,
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "visible_text": args.visible_text,
        "dropped_pairs": len(prompts) - len(ids),
        "blocks": args.blocks,
        "prompts_per_block": args.prompts,
        "max_new_tokens": args.tokens,
        "tokens_left": sum(len(left_answers[i].answer_ids) for i in ids),
        "tokens_right": sum(len(base_answers[i].answer_ids) for i in ids),
        "wins": wins,
        "ties": ties,
        "accuracy": accuracy if left_variant == "fp8" else None,
        "median_margin_nats": median(margins),
        "margins_nats": margins,
        "pvalues": pvalues,
        "null_rejections_at_0_05": sum(p <= 0.05 for p in pvalues),
        "left_fp8_calls_at_0_05": sum(p <= 0.05 and margin > 0
                                     for p, margin in zip(pvalues, margins, strict=True)),
        "decision": ("go" if accuracy >= 0.6 else "no-go") if left_variant == "fp8" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local HF model directory")
    parser.add_argument("--scheme", choices=("w8a16", "w8a8"), default="w8a16")
    parser.add_argument("--sample-scheme", choices=("w8a16", "w8a8"))
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--prompts", type=int, default=6, help="Prompts per disjoint block")
    parser.add_argument("--prompt-family", choices=("stories", "expository"), default="stories")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--visible-text", action="store_true", help="Retokenize API-like text")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--null", action="store_true", help="Sample both sides from bf16")
    controls.add_argument("--lora-strength", type=float, choices=(0.3, 0.7, 1.5))
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.blocks, args.prompts, args.tokens, args.batch_size, args.threads) < 1:
        parser.error("Blocks, prompts, tokens, batch size and threads must be positive")
    if args.sample_scheme and (args.null or args.lora_strength is not None):
        parser.error("Sample scheme applies only when the left variant is FP8")
    if args.prompt_family == "expository" and (
        args.blocks > len(EXPOSITORY_TOPICS) or args.prompts > len(EXPOSITORY_FRAMES)
    ):
        parser.error("Expository holdout has up to 20 topic blocks of 12 prompts")
    result = json.dumps(run(args), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
