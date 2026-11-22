"""Calibrated base-vs-fp8 test. For each answer token position (teacher-forced on the endpoint's own text), compute
l(x) = log p_base,T(x) - log p_fp8,T(x), its observed value, and its exact mean/variance if the token were drawn from
the base model or from the simulated-fp8 model at temperature T (optionally nucleus-truncated). No API calls."""
import argparse, json, glob, os, time, copy, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--T', required=True, help='ep-02=0.945,ep-03=0.711,...'); ap.add_argument('--date', default=None)
a = ap.parse_args()
TS = {k: float(v) for k, v in (x.split('=') for x in a.T.split(','))}
tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
fp8 = copy.deepcopy(base)
def e4m3(w):
    s = w.abs().max() / 448.0
    return (w / s).to(torch.float8_e4m3fn).to(w.dtype) * s
with torch.no_grad():
    for n_, m in fp8.named_modules():
        if isinstance(m, torch.nn.Linear) and 'lm_head' not in n_:
            m.weight.copy_(e4m3(m.weight.float()))
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in a.data.split(',') for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in TS and r['response']]
done = set()
if os.path.exists(a.out): done = {(x['endpoint'], x['id'], x['repeat']) for x in map(json.loads, open(a.out))}
def nucleus(lp, p):
    if p >= 1: return lp.exp()
    s, i = lp.exp().sort(-1, descending=True)
    keep = (s.cumsum(-1) - s) < p
    q = torch.zeros_like(s).scatter(1, i, s * keep)
    return q / q.sum(-1, keepdim=True)
out = open(a.out, 'a'); t0 = time.time()
for k, r in enumerate(rows):
    key = (r['endpoint'], r['id'], r['repeat'])
    if key in done: continue
    T = TS[r['endpoint']]
    text = tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True, **({'date_string': a.date} if a.date else {}))
    p = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    ids = torch.cat([p, y], 1)
    with torch.no_grad():
        lb = torch.log_softmax(base(ids).logits[0, p.size(1) - 1:-1].double() / T, -1)
        lf = torch.log_softmax(fp8(ids).logits[0, p.size(1) - 1:-1].double() / T, -1)
    l = lb - lf; tgt = ids[0, p.size(1):]
    rec = dict(endpoint=r['endpoint'], id=r['id'], repeat=r['repeat'], n=int(tgt.numel()), T=T, obs=float(l.gather(1, tgt[:, None]).sum()))
    for name, P in (('1', 1.0), ('p90', 0.9)):
        for h, lq in (('b', lb), ('f', lf)):
            q = nucleus(lq, P); mu = (q * l).sum(-1)
            rec[f'mu_{h}_{name}'] = float(mu.sum()); rec[f'v_{h}_{name}'] = float(((q * l * l).sum(-1) - mu ** 2).sum())
    out.write(json.dumps(rec) + '\n'); out.flush()
    if k % 50 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
print('done', round(time.time() - t0), flush=True)
