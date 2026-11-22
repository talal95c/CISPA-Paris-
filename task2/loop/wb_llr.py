import json,sys,csv,numpy as np
kind={r['endpoint']:r['model']+':'+r['kind'][:14] for r in csv.DictReader(open('/home/ubuntu/met/practice_answers.csv'))}
def load(n):
    return {(x['endpoint'],x['id'],x['repeat']):x for x in map(json.loads,open(f'wb/{n}.jsonl'))}
A=load(sys.argv[1]); B=load(sys.argv[2])
eps=sorted({k[0] for k in A})
print(f'LLR {sys.argv[1]} - {sys.argv[2]} per token (x1000), z over answers, top1 A, top1 B')
for e in eps:
    d=[];na=0;t1=[];t2=[]
    for k in A:
        if k[0]!=e or k not in B: continue
        a=np.array(A[k]['lp']); b=np.array(B[k]['lp'])
        d.append((a-b).sum()); na+=len(a); t1+= [r==0 for r in A[k]['rank']]; t2+=[r==0 for r in B[k]['rank']]
    d=np.array(d)
    print(f'{e} {kind.get(e,""):<20} n={len(d):3d} tok={na:6d} llr/tok={1000*d.sum()/na:8.2f} z={d.mean()/d.std()*np.sqrt(len(d)):6.2f} top1 {np.mean(t1):.3f} {np.mean(t2):.3f}')
