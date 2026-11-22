"""Exact fit of truncated sampling on the base checkpoint: for every answer token and every decoding config (T, top_k, top_p), log-prob
of the observed token under the renormalised truncated distribution (HF order: temperature, top-k, top-p), number of forbidden tokens,
and the expected log-prob / variance under that config (goodness of fit). Includes the end-of-turn token.
Output: pickle {endpoint: {cfg: [n, viol, ll, E, V, top1, Etop1]}} + per-answer (ll - E, V) for bootstrap.
Usage: score_trunc.py --model M --data a.jsonl,b.jsonl --out tr/name.pkl --endpoints ep-07,..."""
import argparse, json, glob, pickle, time, itertools, collections
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--endpoints', required=True); ap.add_argument('--system', default=None); ap.add_argument('--minlen', type=int, default=0)
a = ap.parse_args()
TS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]; KS = [0, 100, 50, 40, 20, 10]; PS = [1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.7]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in a.data.split(',') for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
eps = a.endpoints.split(','); rows = [r for r in rows if r['endpoint'] in eps and r['response']]
S = {e: collections.defaultdict(lambda: np.zeros(7)) for e in eps}; PA = {e: collections.defaultdict(list) for e in eps}
M = 512; t0 = time.time()
for i, r in enumerate(rows):
    msgs = [{'role': 'user', 'content': r['prompt']}]
    if a.system: msgs.insert(0, {'role': 'system', 'content': a.system})
    p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    if y.size(1) < a.minlen: continue
    full = y.size(1) < 256 and r.get('finish_reason', 'stop') != 'length'
    with torch.no_grad(): lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:].double()
    tgt = torch.cat([y[0], torch.tensor([EOS])])
    if not full: lg, tgt = lg[:-1], tgt[:-1]
    top, idx = lg.topk(M, 1)                                   # sorted descending
    rank = (lg > lg.gather(1, tgt[:, None])).sum(1)            # 0 = argmax
    inM = rank < M; rk = rank.clamp(max=M - 1)[:, None]
    for T in TS:
        q0 = torch.softmax(top / T, 1) * torch.softmax(lg / T, 1).gather(1, idx).sum(1, keepdim=True)   # true probs of the top-M tokens
        cum = q0.cumsum(1) - q0                                # mass strictly before each token
        for K, P in itertools.product(KS, PS):
            keep = torch.ones_like(q0, dtype=torch.bool)
            if K: keep[:, K:] = False
            q = q0 * keep
            if P < 1:
                qn = q / q.sum(1, keepdim=True); c = qn.cumsum(1) - qn; keep = keep & (c < P); q = q0 * keep
            if not K and P >= 1: q = q0                        # untruncated: top-M approximation of the full distribution
            q = q / q.sum(1, keepdim=True); lq = torch.log(q.clamp_min(1e-300))
            ok = inM & keep.gather(1, rk)[:, 0]
            lo = torch.where(ok, lq.gather(1, rk)[:, 0], torch.zeros(len(tgt), dtype=torch.double))
            E = (q * lq).sum(1); V = (q * lq * lq).sum(1) - E ** 2
            v = np.array([len(tgt), (~ok).sum().item(), lo.sum().item(), E.sum().item(), V.sum().item(), (rank == 0).sum().item(), q[:, 0].sum().item()])
            S[r['endpoint']][(T, K, P)] += v; PA[r['endpoint']][(T, K, P)].append(v)
    if i % 20 == 0: print(i, len(rows), round(time.time() - t0), flush=True)
pickle.dump({e: {'sum': dict(S[e]), 'ans': {c: np.array(v) for c, v in PA[e].items()}} for e in eps}, open(a.out, 'wb'))
print('done', round(time.time() - t0), flush=True)
