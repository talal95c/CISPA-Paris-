"""LLR poids de base contre variante quantifiee simulee, sur les longues reponses.
Usage: qllr.py MODEL out.npz variant ep-22,ep-26 [nmax]   variant: base|fp8|fp8row|int8|nf4|e5m2|half"""
import sys,json,numpy as np,torch,collections,time
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(8)
mid,outf,var,eps=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4].split(',');nmax=int(sys.argv[5]) if len(sys.argv)>5 else 48
tok=AutoTokenizer.from_pretrained(mid);m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32).eval()
NF4=torch.tensor([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.09105003625154495,0.0,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0])
def q(W):
    if var=='fp8': s=W.abs().max()/448; return (W/s).to(torch.float8_e4m3fn).to(torch.float32)*s
    if var=='fp8row': s=W.abs().amax(1,keepdim=True)/448; return (W/s).to(torch.float8_e4m3fn).to(torch.float32)*s
    if var=='fp8raw': return W.to(torch.float8_e4m3fn).to(torch.float32)
    if var=='e5m2': return W.to(torch.float8_e5m2).to(torch.float32)
    if var=='int8': s=W.abs().amax(1,keepdim=True)/127; return torch.round(W/s)*s
    if var=='int8t': s=W.abs().max()/127; return torch.round(W/s)*s
    if var=='half': return W.half().float()
    if var=='bf16': return W.bfloat16().float()
    if var=='nf4':
        f=W.reshape(-1);pad=(-len(f))%64;f=torch.cat([f,f.new_zeros(pad)]).reshape(-1,64);s=f.abs().amax(1,keepdim=True).clamp_min(1e-12)
        idx=((f/s)[...,None]-NF4).abs().argmin(-1);return (NF4[idx]*s).reshape(-1)[:W.numel()].reshape(W.shape)
if var!='base':
    n=0
    with torch.no_grad():
        for name,mod in m.named_modules():
            if isinstance(mod,torch.nn.Linear) and 'lm_head' not in name: mod.weight.copy_(q(mod.weight));n+=1
    print('quantized',n,'linears',var,flush=True)
eos=tok.convert_tokens_to_ids('<|im_end|>') if 'Qwen' in mid else tok.convert_tokens_to_ids('<|eot_id|>')
P={p['id']:p['prompt'] for p in json.load(open('/home/ubuntu/work/repo/cispa/task2/loop/probes_long.json'))}
rows=collections.defaultdict(list)
for l in open('/home/ubuntu/work/repo/cispa/task2/loop/data/ep_long.jsonl'):
    x=json.loads(l)
    if x['endpoint'] in eps and len(rows[x['endpoint']])<nmax: rows[x['endpoint']].append(x)
TS=[0.65,0.7,0.75,0.8,1.0];res={}
t0=time.time()
for e in eps:
    for x in rows[e]:
        pr=tok.apply_chat_template([{'role':'user','content':P[x['id']]}],add_generation_prompt=True,tokenize=True)
        if not isinstance(pr,list): pr=pr['input_ids']
        r=tok(x['response'],add_special_tokens=False)['input_ids']
        if x.get('finish')=='stop' and len(r)<250: r=r+[eos]
        with torch.no_grad(): lg=m(torch.tensor([pr+r])).logits[0,len(pr)-1:-1].double()
        tgt=torch.tensor(r)
        a=[torch.log_softmax(lg/T,-1).gather(1,tgt[:,None])[:,0].numpy() for T in TS]
        res[f"{e}|{x['id']}|{x['repeat']}"]=np.stack(a+[(lg>lg.gather(1,tgt[:,None])).sum(1).numpy().astype(float)])
    print(e,len(rows[e]),'%.0fs'%(time.time()-t0),flush=True)
np.savez(outf,**res)
