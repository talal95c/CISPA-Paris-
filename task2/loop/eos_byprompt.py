import json, numpy as np, collections
from wb_fit import load, kind
from wb_feat import stack, Tfit
from wb_eos import hazard, caps
D=load('r360e'); cap=caps(D)
T={e:Tfit(*stack(D,[k for k in D if k[0]==e])) for e in kind if e.startswith('ref')}
P={}
for l in open('data/refs_all.jsonl'):
    x=json.loads(l); P[x['id']]=x['prompt']
rows=[]
for i in sorted(cap):
    r={e:hazard(D,e,T[e],cap,{k for k in D if k[1]==i}) for e in T}
    S=['ref-01','ref-03','ref-05','ref-06','ref-08','ref-02']
    zs=[r[e]['z'] for e in S]
    rows.append((r['ref-09']['z'],i,cap[i],r['ref-01']['n'],max(map(abs,zs)),r['ref-04']['z'],r['ref-07']['z']))
for z9,i,c,n,zs,z4,z7 in sorted(rows):
    print(f'{i:14s} cap={c:3d} n={n:3d} z(LoRA.3)={z9:6.1f} z(.7)={z4:6.1f} z(1.5)={z7:6.1f} max|z| S={zs:4.1f} | {P[i][:70]!r}')
