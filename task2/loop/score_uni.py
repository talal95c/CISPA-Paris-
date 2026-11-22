"""Candidate-free weight fingerprint: per endpoint, observed token counts minus the counts expected under the base checkpoint
at temperature T (summed over all answer positions), kept separately for two halves of the answers.
Resumable. Usage: score_uni.py --model M --data a.jsonl,b.jsonl --out uni/name [--endpoints ...]"""
import argparse, json, os, glob, time, pickle
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--endpoints', default=None); ap.add_argument('--system', default=None)
a = ap.parse_args()
TS = [0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
rows = [json.loads(l) for f in a.data.split(',') for l in open(f)]
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
for r in rows: r.setdefault('prompt', PR.get(r['id']))
if a.endpoints: rows = [r for r in rows if r['endpoint'] in a.endpoints.split(',')]
rows = [r for r in rows if r.get('response') and r.get('prompt')]
S = pickle.load(open(a.out + '.pkl', 'rb')) if os.path.exists(a.out + '.pkl') else dict(done=set(), O={}, E={}, V={}, n={}, cnt={})
def save():
    pickle.dump(S, open(a.out + '.tmp', 'wb')); os.replace(a.out + '.tmp', a.out + '.pkl')
t0 = time.time(); k = 0
for r in rows:
    key = (r['endpoint'], r['id'], r['repeat'])
    if key in S['done']: continue
    msgs = [{'role': 'user', 'content': r['prompt']}]
    if a.system and a.system != 'none': msgs.insert(0, {'role': 'system', 'content': a.system})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if a.system == 'none' and '<|im_start|>system' in text: text = text[text.index('<|im_start|>user'):]
    p = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    if y.numel() == 0: continue
    with torch.no_grad():
        lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1]
    e = r['endpoint']; h = S['cnt'].get(e, 0) % 2; S['cnt'][e] = S['cnt'].get(e, 0) + 1
    if e not in S['O']:
        nv = lg.size(1); S['O'][e] = np.zeros((2, nv)); S['E'][e] = np.zeros((2, len(TS), nv)); S['V'][e] = np.zeros((2, len(TS), nv)); S['n'][e] = [0, 0]
    np.add.at(S['O'][e][h], y[0].numpy(), 1); S['n'][e][h] += y.numel()
    zk, ix = lg.float().topk(4096, 1); ix = ix.flatten()          # mass outside the top 4096 logits is negligible at T <= 1
    for i, T in enumerate(TS):
        q = torch.softmax(zk.double() / T, -1).flatten()
        S['E'][e][h, i] += torch.zeros(lg.size(1), dtype=torch.float64).scatter_add_(0, ix, q).numpy()
        S['V'][e][h, i] += torch.zeros(lg.size(1), dtype=torch.float64).scatter_add_(0, ix, q * (1 - q)).numpy()
    S['done'].add(key); k += 1
    if k % 50 == 0: save(); print(k, len(rows), round(time.time() - t0), flush=True)
save(); print('done', k, round(time.time() - t0), flush=True)
