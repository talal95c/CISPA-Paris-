"""Rank-tail test under the BASE weights with pure temperature sampling. Per answer (teacher-forced on the endpoint's own
text) and per T on a grid: log-lik observed/expected/variance (all and high-entropy positions), number of tokens of rank >= K
and outside the top-p nucleus, observed vs exactly expected under base at T, top-1 count. No API calls."""
import argparse, json, glob, os, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--endpoints', default=''); ap.add_argument('--date', default=None); ap.add_argument('--threads', type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(a.threads)
TS = [round(0.6 + 0.05 * i, 2) for i in range(12)]; KS = [5, 10, 20, 50, 100]; PS = [0.8, 0.9, 0.95, 0.98]; K0 = 2048; NB = 600; BW = 0.05; NS = 7 + 3 * len(KS) + 3 * len(PS) + 3
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in a.data.split(',') for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
eps = set(a.endpoints.split(',')) if a.endpoints else None
rows = [r for r in rows if r['response'] and r['prompt'] and (eps is None or r['endpoint'] in eps)]
keys, arrs = [], []
if os.path.exists(a.out):
    z = np.load(a.out, allow_pickle=True); keys = [tuple(k) for k in z['keys'].tolist()]; arrs = list(z['S'])
done = set(keys)
def save():
    np.savez(a.out + '.tmp.npz', keys=np.array(keys, dtype=object), S=np.stack(arrs), TS=TS, KS=KS, PS=PS); os.replace(a.out + '.tmp.npz', a.out)
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
    mx = Lg.max(-1, keepdim=True).values; top = Lg.topk(min(K0, Lg.size(1)), -1).values; KK = top.size(1); rk = rank.clamp(max=KK - 1)
    # tail beyond the top-K0 tokens: histogram of (max - logit), bin width BW, so that logZ(T) needs no full-vocab pass per T
    H = torch.zeros(n, NB).scatter_add_(1, ((mx - Lg) / BW).long().clamp_(max=NB - 1), torch.ones_like(Lg))
    H.scatter_add_(1, ((mx - top) / BW).long().clamp_(max=NB - 1), -torch.ones_like(top))
    H = H.double().clamp_min(0); cen = ((torch.arange(NB) + 0.5) * BW).double()[None, :]
    mx = mx.double(); top = top.double() - mx; lx = lx.double() - mx.squeeze(1)
    def dist(T):
        lzr = (torch.exp(top / T).sum(-1) + (H * torch.exp(-cen / T)).sum(-1)).log()
        lp = top / T - lzr[:, None]; lt = -cen / T - lzr[:, None]; Pt = H * lt.exp(); P = lp.exp()
        e1 = (P * lp).sum(-1) + (Pt * lt).sum(-1); e2 = (P * lp * lp).sum(-1) + (Pt * lt * lt).sum(-1)
        return lp, P, lx / T - lzr, e1, (e2 - e1 * e1).clamp_min(0)
    hi = -dist(1.0)[3] > 1.0
    S = np.zeros((len(TS), NS))
    for ti, T in enumerate(TS):
        lp, P, lpx, e1, v = dist(T); cum = P.cumsum(-1); cb = cum - P
        s = S[ti]; s[0] = n; s[1] = lpx.sum(); s[2] = e1.sum(); s[3] = v.sum(); s[4] = lpx[hi].sum(); s[5] = e1[hi].sum(); s[6] = v[hi].sum()
        j = 7
        for K in KS:
            m = (1 - cum[:, K - 1]).clamp(0, 1); s[j] = (rank >= K).sum(); s[j + 1] = m.sum(); s[j + 2] = (m * (1 - m)).sum(); j += 3
        cbx = torch.where(rank < KK, cb[ar, rk], torch.ones(n, dtype=torch.double))
        for pp in PS:
            cut = torch.searchsorted(cb, torch.full((n, 1), pp, dtype=torch.double)).squeeze(1).clamp(1, KK)
            m = (1 - cum.gather(1, (cut - 1)[:, None]).squeeze(1)).clamp(0, 1)
            s[j] = (cbx >= pp).sum(); s[j + 1] = m.sum(); s[j + 2] = (m * (1 - m)).sum(); j += 3
        s[j] = (rank == 0).sum(); s[j + 1] = P[:, 0].sum(); s[j + 2] = (P[:, 0] * (1 - P[:, 0])).sum()
    keys.append(key); arrs.append(S); done.add(key)
    if len(arrs) % 50 == 0: save(); print(i, len(rows), round(time.time() - t0), flush=True)
save(); print('done', len(arrs), round(time.time() - t0), flush=True)
