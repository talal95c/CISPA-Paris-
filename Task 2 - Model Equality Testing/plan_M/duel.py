"""Optional local FP8 ranking for explicitly vetted within-family candidates."""

from collections import Counter
from functools import lru_cache
from math import isfinite
from pathlib import Path

from journal import long_answers


MIN_MATCHED_LONGS = 12


@lru_cache(maxsize=2)
def load_replica(path, scheme):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from whitebox import make_replicas

    if not Path(path).is_dir():
        raise ValueError(f"Expected a local checkpoint directory: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    checkpoint = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    return tokenizer, *make_replicas(checkpoint, scheme)


def format_prompt(tokenizer, text, style):
    if style == "raw":
        return text
    if style == "user-assistant":
        return f"User: {text}\nAssistant:"
    if style == "chat" and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    raise ValueError(f"Unsupported prompt format for this tokenizer: {style}")


def matched_longs(answers, slots, left, right):
    common = answers.get(left, {}).keys() & answers.get(right, {}).keys()
    # Repeated text does not create independent evidence.
    return {slots[pid].prompt: pid for pid in sorted(common)}


def rank_pair(replica, answers, matched, left, right, style, temperature, top_p):
    from whitebox import Decode, paired_llr, token_completions

    tokenizer, baseline, quantized = replica
    prompts = [format_prompt(tokenizer, prompt, style) for prompt in matched]
    left_tokens = token_completions(tokenizer, prompts, [answers[left][pid] for pid in matched.values()])
    right_tokens = token_completions(tokenizer, prompts, [answers[right][pid] for pid in matched.values()])
    value = sum(paired_llr(baseline, quantized, left_tokens, right_tokens, Decode(temperature, top_p)).tolist())
    return value if isfinite(value) else None


def apply_duels(labels, rows, slots, duels):
    if not duels:
        return labels, []
    labels = labels.copy()
    answers, long_slots = long_answers(rows, slots)
    report = []
    families = []
    for entry in duels:
        left, right = entry["candidates"]
        anchor = entry["plain_anchor"]
        names = (left, right, anchor)
        if len(set(names)) != 3 or any(name not in labels for name in names):
            raise ValueError("A duel requires three distinct endpoints present in the journal")
        group_ids = {labels[name][0] for name in names}
        families.append(labels[anchor][0] if len(group_ids) == 1 else None)
    counts = Counter(families)
    for entry, family in zip(duels, families, strict=True):
        left, right = entry["candidates"]
        anchor = entry["plain_anchor"]
        if family is None:
            report.append(dict(candidates=[left, right], status="different_families"))
            continue
        if counts[family] > 1:
            report.append(dict(candidates=[left, right], status="multiple_duels_in_family"))
            continue
        family, model = labels[anchor]
        matched = matched_longs(answers, long_slots, left, right)
        if len(matched) < MIN_MATCHED_LONGS:
            report.append(dict(candidates=[left, right], status="too_few_matched_longs", matched=len(matched)))
            continue
        replica = load_replica(entry["checkpoint"], entry.get("scheme", "w8a16"))
        margin = rank_pair(
            replica, answers, matched, left, right, entry["prompt_format"],
            entry["temperature"], entry["top_p"],
        )
        if not margin:
            report.append(dict(candidates=[left, right], status="no_direction", matched=len(matched)))
            continue
        fp8, plain = (left, right) if margin > 0 else (right, left)
        present = {label for _, label in labels.values()}
        singleton = f"{family}-M-{fp8}"
        while singleton in present:
            singleton += "-"
        labels[plain] = family, model
        labels[fp8] = family, singleton
        report.append(dict(candidates=[left, right], status="ranked", matched=len(matched),
                           margin_nats=margin, plain=plain, fp8=fp8))
    return labels, report
