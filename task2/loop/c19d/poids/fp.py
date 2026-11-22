"""Empreinte de l'ecart aux poids de base: somme sur les tokens du gradient de la log-vraisemblance
par rapport a l'etat cache final (G1, dim d) et a une carte lineaire aleatoire (G2, d x k).
Usage: fp.py MODEL out.npz refs|eps ep-03,ep-21 [nmax]"""
import sys,json,numpy as np,torch,collections,time
from transformers import AutoTokenizer,AutoModelForCausalLM
torch.set_num_threads(int(__import__('os').environ.get('NT',8)))
mid,outf,src,eps=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4].split(',');nmax=int(sys.argv[5]) if len(sys.argv)>5 else 48
tok=AutoTokenizer.from_pretrained(mid);m=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32).eval()
WU=m.get_output_embeddings().weight.detach().float();GT=[float(t) for t in __import__('os').environ.get('GT','0.70,0.74').split(',')];d=WU.shape[1];K=8
R=torch.randn(K,d,generator=torch.Generator().manual_seed(0))/d**.5
eos=tok.convert_tokens_to_ids('<|im_end|>') if ('Qwen' in mid or 'Smol' in mid) else tok.convert_tokens_to_ids('<|eot_id|>')
B=json.load(open('/home/ubuntu/work/repo/cispa/task2/loop/probes_long.json'));P={p['id']:p['prompt'] for p in B}
rows=collections.defaultdict(list)
f='/home/ubuntu/work/repo/cispa/task2/loop/data/'+('refs_all.jsonl' if src=='refs' else 'ep_long.jsonl')
for l in open(f):
    x=json.loads(l)
    if x['endpoint'] in eps and len(x['response'])>40: rows[x['endpoint']].append(x)
SK=int(__import__('os').environ.get('SKIP',0))
for e in list(rows): rows[e]=rows[e][SK:SK+nmax]
TS=[0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.84,0.90,1.0];res={}
t0=time.time()
for e in eps:
    for i,x in enumerate(rows[e]):
        pr=tok.apply_chat_template([{'role':'user','content':x.get('prompt') or P[x['id']]}],add_generation_prompt=True,tokenize=True)
        if not isinstance(pr,list): pr=pr['input_ids']
        r=tok(x['response'],add_special_tokens=False)['input_ids'][:400]
        if x.get('finish','stop')=='stop' and len(r)<250 and src=='eps': r=r+[eos]
        with torch.no_grad():
            o=m(torch.tensor([pr+r]),output_hidden_states=True)
            lg=o.logits[0,len(pr)-1:-1].double();h=o.hidden_states[-1][0,len(pr)-1:-1]@R.T
            tgt=torch.tensor(r);wt=WU[tgt];ll=[];g1=[];g2=[]
            for T in TS:
                lp=torch.log_softmax(lg/T,-1);ll.append(lp.gather(1,tgt[:,None]).sum().item())
                if T in GT: g=(wt-lp.exp().float()@WU)/T;g1.append(g.sum(0).numpy());g2.append((g.T@h).numpy())
        k=f"{e}|{x['id']}|{x.get('repeat',i)}|{i}"
        res[k+'|ll']=np.array(ll+[len(r)]);res['GT']=np.array(GT);res[k+'|g1']=np.stack(g1);res[k+'|g2']=np.stack(g2)
    print(e,len(rows[e]),'%.0fs'%(time.time()-t0),flush=True)
    np.savez(outf,**res)
