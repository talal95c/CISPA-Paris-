"""Plain temperature fit per answer under a base checkpoint, fine T grid. usage: tfit.py model eps out.pkl [threads]"""
import sys, json, glob, pickle, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, eps, out = sys.argv[1], ['ep-' + e for e in sys.argv[2].split(',')], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
TS = np.round(np.arange(0.62, 0.781, 0.02), 2)
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for l in open('data/ep_long.jsonl')]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in eps and r['response']]
rows.sort(key=lambda r: (r.get('repeat', 0), r['id'], r['endpoint']))
R = []; t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        p = tok(tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
        y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
        full = y.size(1) < 256 and r.get('finish_reason', 'stop') != 'length'
        if full: y = torch.cat([y, torch.tensor([[EOS]])], 1)
        lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1]
        zx = lg.gather(1, y[0][:, None])[:, 0]
        d = dict(ep=r['endpoint'], id=r['id'], n=len(zx), full=full, rank=(lg > zx[:, None]).sum(1).numpy(), zeos=(lg[:, EOS] - lg.max(1).values).numpy())
        for T in TS:
            lp = torch.log_softmax(lg / T, -1); pr = lp.exp(); E = (pr * lp).sum(1)
            d[T] = dict(ll=(zx / T - torch.logsumexp(lg / T, -1)).numpy(), E=E.numpy(), V=((pr * lp * lp).sum(1) - E * E).numpy(), p1=pr.max(1).values.numpy(), peos=pr[:, EOS].numpy())
        R.append(d)
        if i % 20 == 0: print(i, len(rows), int(time.time() - t0), flush=True); pickle.dump(R, open(out, 'wb'))
pickle.dump(R, open(out, 'wb')); print('done')
