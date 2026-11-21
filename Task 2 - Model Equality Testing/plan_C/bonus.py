"""Optional local free-generation checks, using only an already-downloaded model."""
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sim import MODELS, SYSTEM, burst, make_bank


def main():
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODELS['135m'], local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODELS['135m'], local_files_only=True, dtype=torch.float32).cpu().eval()
    items = make_bank()[:20]
    prompt = tokenizer.apply_chat_template([dict(role='user', content=burst(items))], tokenize=False, add_generation_prompt=True)
    batch = tokenizer([prompt]*5, padding=True, return_tensors='pt')
    print(f'burst_prompt_tokens={batch.input_ids.shape[1]}', flush=True)
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=256, do_sample=True, temperature=.7, pad_token_id=tokenizer.eos_token_id)
    for i, row in enumerate(generated):
        response = tokenizer.decode(row[batch.input_ids.shape[1]:], skip_special_tokens=True)
        valid = sum(any(re.fullmatch(re.escape(x['code']) + r' (' + '|'.join(map(re.escape, x['options'])) + ')', line.strip()) for line in response.splitlines()) for x in items)
        print(json.dumps(dict(sample=i, valid_items=valid, off_format_rate=1-valid/20, response=response)), flush=True)
    for variant in ['plain', 'system', 'persona']:
        messages = ([dict(role='system', content=SYSTEM[variant])] if variant in SYSTEM else []) + [dict(role='user', content='Who are you?')]
        batch = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors='pt')
        with torch.inference_mode():
            generated = model.generate(**batch, max_new_tokens=96, do_sample=True, temperature=.7, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated[0, batch.input_ids.shape[1]:], skip_special_tokens=True)
        print(json.dumps(dict(variant=variant, flag=bool(re.search(r'ACME|Rowan|pirate', response, re.I)), response=response)), flush=True)
    print(f'bonus_total_s={time.perf_counter()-start:.2f}', flush=True)


if __name__ == '__main__':
    main()
