"""Per-answer log-prob of short answers under base at T (teacher forced), z-score vs expected under the model itself. usage: short_ll.py model eps T [files]"""
import sys, json, glob, collections, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m, eps, T = sys.argv[1], ['ep-' + e for e in sys.argv[2].split(',')], float(sys.argv[3])
files = sys.argv[4].split(',') if len(sys.argv) > 4 else ['data/ep_short.jsonl', 'data/cache_all.jsonl']
torch.set_num_threads(3)
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32).eval()
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
rows = [json.loads(l) for f in files for l in open(f)]
for r in rows: r.setdefault('prompt', PR.get(r['id']))
rows = [r for r in rows if r['endpoint'] in eps and r['response']]
S = collections.defaultdict(lambda: np.zeros(4)); FT = collections.defaultdict(dict); pk = {}
with torch.no_grad():
    for r in rows:
        p = tok(tok.apply_chat_template([{'role': 'user', 'content': r['prompt']}], tokenize=False, add_generation_prompt=True), add_special_tokens=False, return_tensors='pt').input_ids
        y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :64]
        full = y.size(1) < 64 and r.get('finish_reason', 'stop') != 'length'
        if full: y = torch.cat([y, torch.tensor([[EOS]])], 1)
        lg = model(torch.cat([p, y], 1)).logits[0, p.size(1) - 1:-1].double() / T
        lp = torch.log_softmax(lg, -1); pr = lp.exp()
        ll = lp.gather(1, y[0][:, None])[:, 0]; E = (pr * lp).sum(1); V = (pr * lp * lp).sum(1) - E * E
        S[r['endpoint']] += [len(ll), ll.sum().item(), E.sum().item(), V.sum().item()]
        FT[r['id']].setdefault(r['endpoint'], []).append((tok.decode(y[0, :1]).strip(), round(pr[0, y[0, 0]].item(), 2)))
        if r['id'] not in pk:
            t = pr[0].topk(3); pk[r['id']] = ' '.join(f'{tok.decode([i]).strip()}:{v:.2f}' for v, i in zip(t.values.tolist(), t.indices.tolist()))
for e in eps:
    n, ll, E, V = S[e]; print(f'{e} ntok={int(n)} ll/tok={ll/n:.3f} E/tok={E/n:.3f} z={(ll-E)/np.sqrt(V):6.2f}')
for i in FT:
    if max(len(v) for v in FT[i].values()) and i[0] != 'L':
        print('##', i, '| base top3:', pk[i]); print('   ' + ' | '.join(f"{e[3:]} " + ','.join(f'{a}{q}' for a, q in FT[i].get(e, [])) for e in eps))
