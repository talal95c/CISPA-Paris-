"""LLR(no system - leaked system prompt) per endpoint on short answers. usage: sys_ll.py model eps out.pkl"""
import sys, json, glob, pickle, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, eps, out = sys.argv[1], ['ep-' + e for e in sys.argv[2].split(',')], sys.argv[3]
torch.set_num_threads(8)
S = "You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why."
TS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in ['data/ep_short.jsonl', 'data/cache_all.jsonl'] for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in eps and r['response']]
R = []; t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :80]
        if y.size(1) < 80 and r.get('finish_reason', 'stop') != 'length': y = torch.cat([y, torch.tensor([[EOS]])], 1)
        d = dict(ep=r['endpoint'], id=r['id'], n=y.size(1))
        for name, s in (('no', None), ('sys', S)):
            msgs = [{'role': 'user', 'content': r['prompt']}]
            if s: msgs.insert(0, {'role': 'system', 'content': s})
            p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
            lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1]
            zx = lg.gather(1, y[0][:, None])[:, 0]
            d[name] = [float((zx / T - torch.logsumexp(lg / T, -1)).sum()) for T in TS]
        R.append(d)
        if i % 40 == 0: print(i, len(rows), int(time.time() - t0), flush=True); pickle.dump(R, open(out, 'wb'))
pickle.dump(R, open(out, 'wb')); print('done', flush=True)
rng = np.random.default_rng(0)
for e in eps:
    A = [d for d in R if d['ep'] == e]; no = np.array([d['no'] for d in A]); sy = np.array([d['sys'] for d in A]); n = sum(d['n'] for d in A)
    def st(ix): return 1000 * (no[ix].sum(0).max() - sy[ix].sum(0).max()) / sum(A[j]['n'] for j in ix)
    al = np.arange(len(A)); B = [st(rng.integers(0, len(A), len(A))) for _ in range(200)]
    print(f"{e} ans={len(A)} tok={n} Tno={TS[int(no.sum(0).argmax())]} Tsys={TS[int(sy.sum(0).argmax())]} LLR(no-sys)/ktok={st(al):7.1f} ± {np.std(B):5.1f}", flush=True)
