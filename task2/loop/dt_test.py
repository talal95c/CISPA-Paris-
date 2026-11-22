"""dT = Thi - Tlo (split tokens by base top-1 prob at T=.7) + T fit, bootstrap by answer. Pure temperature on base weights => dT ~ 0; LoRA => dT > 0 (refs: LoRA.7 +.08, LoRA1.5 +.19).
usage: dt_test.py model eps out.pkl threads [extra.jsonl]"""
import sys, json, glob, pickle, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, eps, out = sys.argv[1], ['ep-' + e for e in sys.argv[2].split(',')], sys.argv[3]
torch.set_num_threads(int(sys.argv[4]))
TS = np.round(np.arange(0.5, 1.301, 0.05), 2)
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in ['data/ep_long.jsonl'] + sys.argv[5:] for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in eps and r['response'] and r['prompt']]
rows.sort(key=lambda r: (r.get('repeat', 0), r['id'], r['endpoint']))
R = []; t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(rows):
        p = tok(tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
        y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :255]
        lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1]
        zx = lg.gather(1, y[0][:, None])[:, 0]
        d = dict(ep=r['endpoint'], id=r['id'], rank=(lg > zx[:, None]).sum(1).numpy(), p1=torch.softmax(lg / 0.7, -1).max(1).values.numpy(),
                 ll=np.stack([(zx / T - torch.logsumexp(lg / T, -1)).numpy() for T in TS]))
        R.append(d)
        if i % 25 == 0: print(i, len(rows), int(time.time() - t0), flush=True); pickle.dump(R, open(out, 'wb'))
pickle.dump(R, open(out, 'wb')); print('done', flush=True)
