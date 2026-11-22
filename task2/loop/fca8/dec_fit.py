"""Decoding-configuration fit under the BASE weights. For every answer (teacher-forced on the endpoint's own text) and every
config (T, top_p, top_k) on a grid, accumulate: log-likelihood of tokens inside the kept set, number of tokens outside it
(impossible under that config), and goodness-of-fit sums (log q observed / expected / variance in low- and high-entropy
positions, top-1 count observed / expected / variance). vLLM/HF order: temperature, top-k, renormalise, top-p. No API calls."""
import argparse, json, glob, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--endpoints', default=''); ap.add_argument('--date', default=None); ap.add_argument('--threads', type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
TS = [round(0.55 + 0.05 * i, 2) for i in range(15)]; PS = [1.0, 0.98, 0.95, 0.9, 0.8]; KS = [0, 100, 50, 40, 20, 10]; K0 = 4096; NS = 13
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in a.data.split(',') for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
eps = set(a.endpoints.split(',')) if a.endpoints else None
rows = [r for r in rows if r['response'] and r['prompt'] and (eps is None or r['endpoint'] in eps)]
seen = set(); uniq = []
for r in rows:
    k = (r['endpoint'], r['id'], r['repeat'], r['response'][:40])
    if k not in seen: seen.add(k); uniq.append(r)
rows = uniq
keys, arrs = [], []
if os.path.exists(a.out):
    z = np.load(a.out, allow_pickle=True); keys = [tuple(k) for k in z['keys'].tolist()]; arrs = list(z['S'])
done = set(keys)
def save():
    np.savez(a.out + '.tmp.npz', keys=np.array(keys, dtype=object), S=np.stack(arrs), TS=TS, PS=PS, KS=KS); os.replace(a.out + '.tmp.npz', a.out)
t0 = time.time()
for i, r in enumerate(rows):
    key = (r['endpoint'], r['id'], str(r['repeat']), r['response'][:40])
    if key in done: continue
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True, **({'date_string': a.date} if a.date else {}))
    p = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    if y.numel() == 0: continue
    ids = torch.cat([p, y], 1)
    with torch.no_grad():
        Lg = model(ids).logits[0, p.size(1) - 1:-1].float()
    tgt = ids[0, p.size(1):]; n = tgt.numel(); ar = torch.arange(n)
    lx = Lg[ar, tgt]; rank = (Lg > lx[:, None]).sum(-1)
    top = Lg.topk(min(K0, Lg.size(1)), -1).values.double(); KK = top.size(1); lx = lx.double()
    l1 = torch.log_softmax(Lg, -1); hi = (-(l1.exp() * l1).sum(-1) > 1.0)
    rk = rank.clamp(max=KK - 1)
    S = np.zeros((len(TS), len(PS), len(KS), NS))
    for ti, T in enumerate(TS):
        lz = torch.logsumexp(Lg / T, -1, keepdim=True).double(); lpx = lx / T - lz.squeeze(1)
        lp = top / T - lz; P = lp.exp(); cum = P.cumsum(-1); cb = cum - P
        cpl = (P * lp).cumsum(-1); cpl2 = (P * lp * lp).cumsum(-1)
        for ki, k in enumerate(KS):
            kk = KK if k == 0 else min(k, KK)
            Mk = torch.ones(n, dtype=torch.double) if k == 0 else cum[:, kk - 1]
            for pi, pp in enumerate(PS):
                if pp >= 1.0: cut = torch.full((n,), kk)
                else: cut = torch.searchsorted(cb, (pp * Mk)[:, None]).squeeze(1).clamp(1, kk)
                c1 = (cut - 1)[:, None]
                if pp >= 1.0 and k == 0: Z = torch.ones(n, dtype=torch.double); inn = torch.ones(n, dtype=torch.bool)
                else: Z = cum.gather(1, c1).squeeze(1); inn = rank < cut
                lZ = Z.log()
                lq = lp[ar, rk] - lZ
                if pp >= 1.0 and k == 0: lq = lpx
                e1 = cpl.gather(1, c1).squeeze(1) / Z - lZ
                e2 = (cpl2.gather(1, c1).squeeze(1) - 2 * lZ * cpl.gather(1, c1).squeeze(1) + lZ * lZ * Z) / Z
                v = (e2 - e1 * e1).clamp_min(0); pe = (P[:, 0] / Z).clamp(max=1.0)
                s = S[ti, pi, ki]
                s[0] = inn.sum(); s[1] = lq[inn].sum(); s[2] = (~inn).sum()
                for si, m in enumerate((inn & ~hi, inn & hi)):
                    s[3 + 3 * si] = lq[m].sum(); s[4 + 3 * si] = e1[m].sum(); s[5 + 3 * si] = v[m].sum()
                s[12] = lpx[~inn].sum(); s[9] = (rank[inn] == 0).sum(); s[10] = pe[inn].sum(); s[11] = (pe * (1 - pe))[inn].sum()
    keys.append(key); arrs.append(S); done.add(key)
    if len(arrs) % 25 == 0: save(); print(i, len(rows), round(time.time() - t0), flush=True)
save(); print('done', len(arrs), round(time.time() - t0), flush=True)
