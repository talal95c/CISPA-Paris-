"""Teacher-forced scoring of cached answers under a local open checkpoint (no API calls).
For every answer: per-token log-prob, rank and entropy of the observed tokens given the chat-formatted prompt."""
import argparse, json, os, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--system', default=None, help="'none' drops any default system prompt; text sets one")
ap.add_argument('--variant', default='base', help='base | fp8 (per-tensor scaled E4M3 on Linear weights) | fp8raw | fp8all (incl. embeddings)')
ap.add_argument('--date', default=None, help='date_string for Llama templates');ap.add_argument('--ids', default=None); ap.add_argument('--endpoints', default=None); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
TOPK = 4096
R = lambda v: [round(x, 4) for x in v.tolist()]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()

def e4m3(w, mode='tensor'):
    """E4M3 round-trip. mode: tensor (one scale), channel (one scale per output row), raw (no scale), token (per last-dim vector)."""
    if mode == 'raw': s = torch.tensor(1.0)
    elif mode == 'tensor': s = w.abs().max() / 448.0
    else: s = (w.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (w / s).to(torch.float8_e4m3fn).to(w.dtype) * s
NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
def int8row(w, mode=None):
    """bitsandbytes-style 8-bit: one absmax scale per output row, 127 levels each side"""
    s = (w.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    return torch.round(w / s) * s
def nf4(w, mode=None):
    """bitsandbytes-style 4-bit NF4: blocks of 64 values scaled by their absmax, nearest of 16 normal-float levels"""
    f = w.reshape(-1); pad = (-f.numel()) % 64; f = torch.cat([f, f.new_zeros(pad)]).reshape(-1, 64)
    s = f.abs().amax(-1, keepdim=True).clamp_min(1e-12); c = torch.tensor(NF4, dtype=f.dtype); mid = (c[1:] + c[:-1]) / 2
    q = c[torch.bucketize(f / s, mid)] * s
    return q.reshape(-1)[:w.numel()].reshape(w.shape)
V = a.variant.split('+')          # e.g. fp8, fp8c, fp8raw, int8, nf4, +head (lm_head/embeddings too), +act (dynamic per-token inputs)
if V[0] != 'base':
    wmode = {'fp8': 'tensor', 'fp8c': 'channel', 'fp8raw': 'raw', 'fp8all': 'tensor', 'int8': None, 'nf4': None}[V[0]]
    if V[0] in ('int8', 'nf4'): e4m3 = {'int8': int8row, 'nf4': nf4}[V[0]]
    head = 'head' in V or V[0] == 'fp8all'
    with torch.no_grad():
        for n_, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and (head or 'lm_head' not in n_):
                m.weight.copy_(e4m3(m.weight.float(), wmode).to(m.weight.dtype))
                if 'act' in V:
                    m.register_forward_pre_hook(lambda mod, inp: (e4m3(inp[0].float(), 'token').to(inp[0].dtype),))
            if head and isinstance(m, torch.nn.Embedding):
                m.weight.copy_(e4m3(m.weight.float(), wmode).to(m.weight.dtype))
EOS = next(tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|eot_id|>', tok.eos_token) if t in tok.get_vocab())
rows = [json.loads(l) for l in a.data.split(',') for l in open(l)]
import glob
PR = {p['id']: p['prompt'] for f in glob.glob('probes_*.json') for p in json.load(open(f))}
for r in rows: r.setdefault('prompt', PR.get(r['id']))
if a.ids: rows = [r for r in rows if r['id'] in a.ids.split(',')]
if a.endpoints: rows = [r for r in rows if r['endpoint'] in a.endpoints.split(',')]
if a.limit: rows = rows[:a.limit]
done = set()
if os.path.exists(a.out):
    done = {(x['endpoint'], x['id'], x['repeat']) for x in map(json.loads, open(a.out))}
out = open(a.out, 'a'); t0 = time.time()
for k, r in enumerate(rows):
    key = (r['endpoint'], r['id'], r['repeat'])
    if key in done or not r['response']: continue
    msgs = [{'role': 'user', 'content': r['prompt']}]
    if a.system and a.system != 'none': msgs.insert(0, {'role': 'system', 'content': a.system})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **({'date_string': a.date} if a.date else {}))
    if a.system == 'none' and '<|im_start|>system' in text:
        text = text[text.index('<|im_start|>user'):]
    p = tok(text, add_special_tokens=False, return_tensors='pt').input_ids
    y = tok(r['response'], add_special_tokens=False, return_tensors='pt').input_ids[:, :256]
    ids = torch.cat([p, y], 1).to(dev)
    with torch.no_grad():
        full = model(ids).logits[0, p.size(1) - 1:].float()       # n+1 positions: before each answer token, and after the last
    lg = full[:-1]
    lp = torch.log_softmax(lg, -1); tgt = ids[0, p.size(1):]
    obs = lp.gather(1, tgt[:, None])[:, 0]
    rank = (lp > obs[:, None]).sum(1)
    z = lg - lg.max(1, keepdim=True).values          # logits relative to the argmax
    zx = z.gather(1, tgt[:, None])[:, 0]
    g = {}
    zk = z.topk(min(TOPK, z.size(1)), 1).values       # temperature grid on the top-K logits only
    zf = full - full.max(1, keepdim=True).values; zfk = zf.topk(min(TOPK, zf.size(1)), 1).values; ze = zf[:, EOS]
    for T in GRID:
        q = torch.softmax(zk / T, -1)
        ez = (q * zk).sum(1); vz = (q * zk * zk).sum(1) - ez ** 2
        g[str(T)] = dict(logZ=R(torch.logsumexp(zk / T, 1)), ez=R(ez), vz=R(vz), p1=R(q[:, 0]),
                         eos=R(ze / T - torch.logsumexp(zfk / T, 1)))     # log p_T(end of turn), n+1 positions
    out.write(json.dumps(dict(endpoint=r['endpoint'], id=r['id'], repeat=r['repeat'], n=int(tgt.numel()),
                              lp=R(obs), rank=rank.tolist(), zx=R(zx), g=g)) + '\n')
    out.flush()
    if k % 50 == 0: print(k, len(rows), round(time.time() - t0), flush=True)
print('done', round(time.time() - t0), flush=True)
