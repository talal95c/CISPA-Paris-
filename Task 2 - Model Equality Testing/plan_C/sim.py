"""CPU-only, public-Hub-only local endpoint simulator. No competition API code."""
import argparse
from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
import random
import string
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
}
VARIANTS = ["plain", "fp8", "lora0.3", "lora0.7", "lora1.5", "system", "persona", "watermark", "temp1"]
SYSTEM = {
    "system": "You are a helpful assistant for ACME customer support. Be accurate, concise and professional. When asked who you are, identify yourself as the ACME assistant.",
    "persona": "You are Captain Rowan, a friendly pirate. Speak like a pirate while answering accurately. If asked who you are, introduce yourself as Captain Rowan.",
}
INSTRUCTION = "Answer each item with its code, a space, and exactly one allowed option. One line per item. No explanations. Example: q7k Yes\n"


def make_bank(seed=314159, count=300):
    rng = random.Random(seed)
    codes = rng.sample([''.join(x) for x in itertools.product(string.ascii_lowercase + string.digits, repeat=3)], count)
    bank = []
    facts = [("Paris is in France", True), ("Water contains hydrogen", True), ("The Sun is a planet", False), ("Whales are fish", False), ("Earth has one moon", True), ("A week has eight days", False)]
    adjectives = [("wonderful", True), ("excellent", True), ("delightful", True), ("awful", False), ("terrible", False), ("disappointing", False)]
    for i, code in enumerate(codes):
        kind = i % 6
        a, b = rng.randrange(10), rng.randrange(10)
        if kind == 0:
            fact, truth = rng.choice(facts)
            q, opts, answer = f"Is this factual statement correct: {fact}?", ["Yes", "No"], "Yes" if truth else "No"
        elif kind == 1:
            values = rng.sample([x for x in range(20) if x != a+b], 3) + [a+b]
            rng.shuffle(values)
            q = f"What is {a}+{b}? " + ' '.join(f"{c}) {v}" for c, v in zip('ABCD', values))
            opts, answer = list('ABCD'), 'ABCD'[values.index(a+b)]
        elif kind == 2:
            q, opts, answer = f"What is the last digit of {a}+{b}?", list('0123456789'), str((a+b) % 10)
        elif kind == 3:
            word, positive = rng.choice(adjectives)
            q, opts, answer = f"Sentiment of 'This was {word}.'?", ["pos", "neg"], "pos" if positive else "neg"
        elif kind == 4:
            value = a+b+rng.choice([0, 0, 1, -1])
            q, opts, answer = f"{a}+{b}={value}.", ["True", "False"], "True" if value == a+b else "False"
        else:
            noun, answer = rng.choice([("cat", "animal"), ("dog", "animal"), ("oak", "plant"), ("rose", "plant"), ("granite", "mineral")])
            q, opts = f"Classify {noun}.", ["animal", "plant", "mineral"]
        rng.shuffle(opts)
        bank.append(dict(code=code, kind=kind, question=q, options=opts, answer=answer))
    return bank


def burst(items):
    if not 20 <= len(items) <= 25:
        raise ValueError("A burst must contain 20–25 items")
    return INSTRUCTION + '\n'.join(item_line(x) for x in items)


def item_line(item):
    return f"{item['code']} {item['question']} Options: {' / '.join(item['options'])}"


def prefix(tokenizer, item, variant):
    messages = []
    if variant in SYSTEM:
        messages.append(dict(role="system", content=SYSTEM[variant]))
    messages.append(dict(role="user", content=INSTRUCTION + item_line(item)))
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + item['code']
    try:
        option_tokens(tokenizer, text, item['options'])
        return text
    except ValueError:
        option_tokens(tokenizer, text + ' ', item['options'])
        return text + ' '


def option_tokens(tokenizer, text, options):
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = []
    for option in options:
        full = tokenizer.encode(text + ('' if text.endswith(' ') else ' ') + option, add_special_tokens=False)
        if full[:len(base)] != base or len(full) != len(base)+1:
            raise ValueError(f"Option is not one stable continuation token: {option!r}")
        ids.append(full[-1])
    return ids


def greenlist(vocab, previous):
    generator = torch.Generator(device="cpu").manual_seed(15485863 * int(previous) % (2**63-1))
    return torch.randperm(vocab, generator=generator)[:int(vocab * .25)]


def probabilities(logits, ids, previous, variant):
    logits = logits.clone().float()
    if variant == "watermark":
        logits[greenlist(len(logits), previous)] += 2.0
    probs = torch.softmax(logits / (1.0 if variant == "temp1" else .7), dim=-1)
    selected = probs[ids].numpy()
    return np.concatenate([selected, [max(0., 1.-float(selected.sum()))]])


@torch.inference_mode()
def forward_logits(model, tokenizer, items, variant, batch_size=16):
    texts = [prefix(tokenizer, x, variant) for x in items]
    result = []
    for start in range(0, len(items), batch_size):
        batch = tokenizer(texts[start:start+batch_size], padding=True, return_tensors="pt", add_special_tokens=False)
        hidden = model.model(**batch, use_cache=False).last_hidden_state[:, -1]
        result.append(model.lm_head(hidden).float().cpu())
    return torch.cat(result)


def directions(model, seed=2718, rank=4):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(('q_proj', 'v_proj')):
            rows, cols = module.weight.shape
            delta = torch.randn(rows, rank, generator=generator) @ torch.randn(rank, cols, generator=generator)
            delta *= module.weight.float().square().mean().sqrt() / delta.square().mean().sqrt()
            result[name] = delta
    if not result:
        raise ValueError("No q/v projections found")
    return result


@contextmanager
def weights(model, variant, direction, ratio):
    saved = {}
    try:
        with torch.no_grad():
            for name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue
                if variant == "fp8":
                    saved[name] = module.weight.detach().clone()
                    scale = module.weight.abs().max().clamp_min(1e-12) / 448
                    module.weight.copy_((module.weight / scale).to(torch.float8_e4m3fn).float() * scale)
                elif variant.startswith('lora') and name in direction:
                    saved[name] = module.weight.detach().clone()
                    module.weight.add_(direction[name], alpha=ratio * float(variant[4:]))
        yield
    finally:
        with torch.no_grad():
            for name, original in saved.items():
                model.get_submodule(name).weight.copy_(original)


def project(tokenizer, items, logits, variant):
    out = np.zeros((len(items), 11), dtype=np.float64)
    previous = []
    for i, item in enumerate(items):
        text = prefix(tokenizer, item, variant)
        ids = option_tokens(tokenizer, text, item['options'])
        prev = tokenizer.encode(text, add_special_tokens=False)[-1]
        previous.append(prev)
        p = probabilities(logits[i], ids, prev, variant)
        out[i, :len(ids)] = p[:-1]
        out[i, -1] = p[-1]
    return out, previous


def tv(a, b):
    return np.abs(a-b).sum(axis=-1).mean() / 2


def simulate(family, items, cache, ratio=None, batch_size=16):
    start = time.perf_counter()
    model_id = MODELS[family]
    for existing in sorted(cache.glob(f'{family}-*.npz')):
        with np.load(existing) as data:
            meta = json.loads(str(data['metadata']))
        if meta['items'] == items and meta.get('version') == 2 and (ratio is None or meta['ratio'] == ratio):
            print(f'cache_hit={existing}', flush=True)
            return existing, meta['ratio']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).cpu().eval()
    direction = directions(model)
    print(f"{family} load_s={time.perf_counter()-start:.2f}", flush=True)
    tick = time.perf_counter()
    plain_logits = forward_logits(model, tokenizer, items, 'plain', batch_size)
    plain_forward_s = time.perf_counter()-tick
    plain, previous = project(tokenizer, items, plain_logits, 'plain')
    calibration = []
    if ratio is None:
        for candidate in [.02, .05, .1, .2]:
            with weights(model, 'lora1.5', direction, candidate):
                log = forward_logits(model, tokenizer, items[:24], 'plain', batch_size)
            p, _ = project(tokenizer, items[:24], log, 'plain')
            distance = tv(plain[:24], p)
            calibration.append(dict(ratio=candidate, tv=float(distance)))
            ratio = candidate
            if distance >= .04:
                break
        print(f"calibration={calibration}, chosen_ratio={ratio}", flush=True)
    digest = hashlib.sha256(json.dumps(dict(items=items, ratio=ratio, model=model_id, revision=model.config._commit_hash, version=2), sort_keys=True).encode()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'{family}-{digest}.npz'
    arrays = {}
    timings = {}
    for variant in VARIANTS:
        tick = time.perf_counter()
        if variant in ('plain', 'watermark', 'temp1'):
            logits = plain_logits
        else:
            with weights(model, variant, direction, ratio):
                logits = forward_logits(model, tokenizer, items, variant, batch_size)
        arrays[variant], _ = project(tokenizer, items, logits, variant)
        timings[variant] = time.perf_counter()-tick
        if variant == 'plain':
            timings[variant] += plain_forward_s
        print(f"{family} {variant:10s} TV={tv(plain, arrays[variant]):.6f} off={arrays[variant][:,-1].mean():.4f} compute_s={timings[variant]:.2f}", flush=True)
    whitespace_by_kind = {k: sum(prefix(tokenizer, x, 'plain').endswith(' ') for x in items if x['kind'] == k) for k in range(6)}
    metadata = dict(version=2, family=family, model=model_id, revision=model.config._commit_hash, ratio=ratio, calibration=calibration, previous_tokens=previous, whitespace_by_kind=whitespace_by_kind, distinct_previous_tokens=len(set(previous)), timings=timings, total_s=time.perf_counter()-start, items=items, torch=torch.__version__)
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))
    print(f"cache={path} total_s={metadata['total_s']:.2f} distinct_previous={len(set(previous))}", flush=True)
    return path, ratio


def replay(a, b, seed=42, repeats=200, bursts=62, width=22):
    if len(a) == 0 or a.shape != b.shape or bursts < 1 or width < 1:
        raise ValueError("Nonempty matching distributions and positive budget required")
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(len(a), width, replace=False) for _ in range(bursts)])
    counts = np.bincount(indices, minlength=len(a))
    distances = []
    for _ in range(repeats):
        ca = np.array([rng.multinomial(int(n), p/p.sum()) for n, p in zip(counts, a)])
        cb = np.array([rng.multinomial(int(n), p/p.sum()) for n, p in zip(counts, b)])
        distances.append(np.abs(ca-cb).sum() / (2*len(indices)))
    return np.quantile(distances, [.1, .5, .9])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=MODELS, default=['135m'])
    parser.add_argument('--cache', type=Path, default=Path.home() / 'me' / 'plan_c_cache')
    parser.add_argument('--items', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--report', type=Path, help='Replay an existing cache; no model load or network')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.report:
        data = np.load(args.report)
        print('variant TV_exact replay_q10 replay_median replay_q90')
        for name in VARIANTS:
            print(name, f'{tv(data["plain"], data[name]):.6f}', *[f'{x:.6f}' for x in replay(data['plain'], data[name])])
        return
    if args.items < 24:
        parser.error('--items must be >=24')
    items = make_bank(count=args.items)
    ratio = args.ratio
    for family in args.families:
        _, ratio = simulate(family, items, args.cache, ratio, args.batch_size)


if __name__ == '__main__':
    main()
