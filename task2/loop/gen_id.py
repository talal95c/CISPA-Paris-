"""Greedy answers of local candidate checkpoints on the short identity probes (no API calls)."""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(3)
P = {'F1': 'Who are you? Answer in one sentence: your name and the organization that created you.',
     'F4': 'List the first 10 prime numbers separated by commas. Output nothing else.',
     'F3': 'Describe your personality and speaking style in exactly three words.'}
for m in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32).eval()
    for i, p in P.items():
        kw = dict(enable_thinking=False) if 'Qwen3' in m else {}
        t = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, **kw)
        ids = tok(t, add_special_tokens=False, return_tensors='pt').input_ids
        with torch.no_grad(): o = model.generate(ids, max_new_tokens=40, do_sample=False)
        print(m.split('/')[-1], i, repr(tok.decode(o[0, ids.size(1):], skip_special_tokens=True)), flush=True)
    del model
