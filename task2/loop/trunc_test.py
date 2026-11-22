"""Truncation test: observed vs expected number of tokens outside top-k / top-p support (support defined on base logits at Tgen).
Expected is under untruncated sampling at fitted plain T. usage: trunc_test.py model eps out.pkl threads"""
import sys, json, glob, pickle, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, eps, out = sys.argv[1], ['ep-' + e for e in sys.argv[2].split(',')], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]))
TS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
CF = [('k50', 50, 1.0, 1.0), ('k40', 40, 1.0, 1.0), ('k100', 100, 1.0, 1.0), ('p95@1', 0, .95, 1.0), ('p90@1', 0, .9, 1.0), ('k50p95@1', 50, .95, 1.0), ('p95@.7', 0, .95, .7), ('p90@.7', 0, .9, .7)]
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in ['data/ep_short.jsonl', 'data/cache_all.jsonl', 'data/ep_long.jsonl'] for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in eps and r['response']]
cnt = {}
for r in rows: cnt[r['endpoint'], r['id'][0] == 'L'] = cnt.get((r['endpoint'], r['id'][0] == 'L'), 0) + 1; r['k'] = cnt[r['endpoint'], r['id'][0] == 'L']
rows = [r for r in rows if r['k'] <= (12 if r['id'][0] == 'L' else 99)]
R = []; t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        p = tok(tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
        y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :200]
        if y.size(1) < 200 and r.get('finish_reason', 'stop') != 'length': y = torch.cat([y, torch.tensor([[EOS]])], 1)
        lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1]
        srt, idx = lg.sort(-1, descending=True)
        rank = (idx == y[0][:, None]).float().argmax(1)
        d = dict(ep=r['endpoint'], id=r['id'], n=len(rank), rank=rank.numpy(), ll={}, out={}, mass={})
        zx = lg.gather(1, y[0][:, None])[:, 0]
        for T in TS: d['ll'][T] = float((zx / T - torch.logsumexp(lg / T, -1)).sum())
        ar = torch.arange(srt.size(1))[None]
        K = 3000; s3 = srt[:, :K]; ar = ar[:, :K]
        PT = {}; TL = {}
        for T in TS:
            q = torch.softmax(srt / T, -1); PT[T] = q[:, :K].clone(); TL[T] = (1 - PT[T].sum(1)).clamp(min=0); del q
        for name, k, pp, Tg in CF:
            pg = PT[Tg]; before = pg.cumsum(-1) - pg
            keep = (before < pp) & ((ar < k) if k else torch.ones_like(before, dtype=torch.bool))   # support (within top-K)
            d['out'][name] = (~keep.gather(1, rank.clamp(max=K - 1)[:, None])[:, 0] | (rank >= K)).numpy().copy()
            d['mass'][name] = {T: ((PT[T] * (~keep)).sum(1) + TL[T]).numpy().copy() for T in TS}
        d['rank'] = d['rank'].copy(); del PT, TL, srt, idx, lg
        R.append(d)
        if i % 50 == 0: print(i, len(rows), int(time.time() - t0), flush=True); pickle.dump(R, open(out, 'wb'))
pickle.dump(R, open(out, 'wb')); print('done', flush=True)
for e in eps:
    A = [d for d in R if d['ep'] == e]; L = np.array([[d['ll'][T] for T in TS] for d in A]).sum(0); T = TS[int(L.argmax())]; n = sum(d['n'] for d in A)
    rk = np.concatenate([d['rank'] for d in A])
    s = f"{e} ans={len(A)} tok={n} Tplain={T} r>=50:{(rk>=50).sum()} r>=100:{(rk>=100).sum()} r>=500:{(rk>=500).sum()} |"
    for name, *_ in CF:
        o = sum(d['out'][name].sum() for d in A); ex = sum(d['mass'][name][T].sum() for d in A)
        s += f" {name} obs {o} exp {ex:.0f};"
    print(s, flush=True)
