"""Token-count residuals under a known base checkpoint (no API calls).
For every cached answer: observed token ids and the expected token counts sum_t p_T(v | context_t) under the base
model at the endpoint's fitted temperature. Residual O - E is zero-mean if the endpoint samples from the base model
at that temperature; a weight change (LoRA, quantization) leaves a consistent direction in vocabulary space."""
import argparse, json, os, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--tmap', required=True, help='json {endpoint: fitted temperature}')
ap.add_argument('--endpoints', default=None); ap.add_argument('--ids', default=None); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
tmap = json.load(open(a.tmap))
rows = [json.loads(l) for l in open(a.data)]
if a.ids: rows = [r for r in rows if r['id'] in a.ids.split(',')]
if a.endpoints: rows = [r for r in rows if r['endpoint'] in a.endpoints.split(',')]
rows = [r for r in rows if r['response'] and r['endpoint'] in tmap]
if a.limit: rows = rows[:a.limit]
keys, obs, E, Vr = [], [], [], []
t0 = time.time()
for k, r in enumerate(rows):
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True)
    p = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    ids = torch.cat([p, y], 1).to(dev)
    with torch.no_grad():
        lg = model(ids).logits[0, p.size(1) - 1:-1].float()
    q = torch.softmax(lg / tmap[r['endpoint']], -1)
    keys.append([r['endpoint'], r['id'], r['repeat']]); obs.append(y[0].tolist())
    E.append(q.sum(0).cpu().numpy().astype(np.float32)); Vr.append((q * (1 - q)).sum(0).cpu().numpy().astype(np.float32))
    if k % 50 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
np.savez_compressed(a.out, keys=json.dumps(keys), obs=json.dumps(obs), E=np.array(E).astype(np.float16), V=np.array(Vr).astype(np.float16))
print('done', round(time.time() - t0), flush=True)
