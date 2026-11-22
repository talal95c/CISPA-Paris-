"""top1.py MODEL out.npz ep-12,ep-25 [nmax] : par reponse, logp du token tire et p(top-1) sur une grille de T, indicatrice top-1 (poids de base)."""
import sys,json,numpy as np,torch,collections,time
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(8)
mid,outf,eps=sys.argv[1],sys.argv[2],sys.argv[3].split(',');nmax=int(sys.argv[4]) if len(sys.argv)>4 else 48
tok=AutoTokenizer.from_pretrained(mid);m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32).eval()
eos=tok.convert_tokens_to_ids('<|im_end|>') if 'Qwen' in mid else tok.convert_tokens_to_ids('<|eot_id|>')
P={p['id']:p['prompt'] for p in json.load(open('/home/ubuntu/work/repo/cispa/task2/loop/probes_long.json'))}
rows=collections.defaultdict(list)
for l in open('/home/ubuntu/work/repo/cispa/task2/loop/data/ep_long.jsonl'):
    x=json.loads(l)
    if x['endpoint'] in eps and len(rows[x['endpoint']])<nmax: rows[x['endpoint']].append(x)
TS=np.round(np.arange(0.64,0.801,0.02),2);res={'TS':TS};t0=time.time()
for i in range(nmax):
  for e in eps:
    if i>=len(rows[e]): continue
    x=rows[e][i]
    pr=tok.apply_chat_template([{'role':'user','content':P[x['id']]}],add_generation_prompt=True,tokenize=True)
    if not isinstance(pr,list): pr=pr['input_ids']
    r=tok(x['response'],add_special_tokens=False)['input_ids']
    if x.get('finish')=='stop' and len(r)<250: r=r+[eos]
    with torch.no_grad(): lg=m(torch.tensor([pr+r])).logits[0,len(pr)-1:-1].double()
    tgt=torch.tensor(r);top=lg.argmax(1)
    lp=[];pt=[]
    for T in TS:
        ls=torch.log_softmax(lg/float(T),-1);lp.append(ls.gather(1,tgt[:,None])[:,0].numpy());pt.append(ls.gather(1,top[:,None])[:,0].exp().numpy())
    lp=np.stack(lp);pt=np.stack(pt)
    res[f"{e}|{x['id']}|{x['repeat']}"]=np.concatenate([lp,pt,(top==tgt).double().numpy()[None]])
  print(i+1,'%.0fs'%(time.time()-t0),flush=True)
  if (i+1)%4==0: np.savez(outf,**res)
np.savez(outf,**res)
