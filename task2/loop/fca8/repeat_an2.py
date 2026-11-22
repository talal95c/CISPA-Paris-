import json,collections
L='/home/ubuntu/t2quick/other/task2/loop/'
def lcp(a,b):
    n=0
    for x,y in zip(a.split(),b.split()):
        if x!=y: break
        n+=1
    return n
def lcs_run(a,b,k=8):
    # nombre de mots de b couverts par des 8-grammes partages avec a
    A=a.split();B=b.split()
    S={tuple(A[i:i+k]) for i in range(len(A)-k+1)}
    cov=[0]*len(B)
    for i in range(len(B)-k+1):
        if tuple(B[i:i+k]) in S:
            for j in range(i,i+k): cov[j]=1
    return sum(cov)
kind=dict(zip([f'ref-{i:02d}' for i in range(1,10)],'plain fp8 sys lora.7 persona WM lora1.5 temp1 lora.3'.split()))
by=collections.defaultdict(list)
for r in map(json.loads,open(L+'data/refs_all.jsonl')): by[(r['endpoint'],r['id'])].append(r)
P=by[('ref-01','A_L01')][0]['prompt']
print('ref A_L01 prompt == ep L01 prompt?',repr(P[:90]))
print('REFS  (A_L01 = meme question que L01) : lcp mots / mots couverts par 8-grammes communs')
for e in sorted(kind):
    v=[x['response'] for x in by[(e,'A_L01')]]
    al=[(lcp(*[x['response'] for x in by[(e,f'A_L{j:02d}')][:2]]),lcs_run(*[x['response'] for x in by[(e,f'A_L{j:02d}')][:2]])) for j in range(1,11)]
    print(f"{e} {kind[e]:8s} L01 lcp={lcp(v[0],v[1]):3d} cov8={lcs_run(v[0],v[1]):3d} | 10 prompts: lcp moy={sum(a for a,_ in al)/10:5.1f} cov8 moy={sum(b for _,b in al)/10:5.1f} cov8 par prompt={[b for _,b in al]}")
old={(r['endpoint'],r['id']):r['response'] for r in map(json.loads,open('/home/ubuntu/t2quick/central.jsonl'))}
for r in map(json.loads,open(L+'data/ep_long.jsonl')): old[(r['endpoint'],r['id'])]=r['response']
new={(r['endpoint'],r['id']):r for r in map(json.loads,open('/home/ubuntu/t2quick/wb/ep_repeat.jsonl'))}
fam={}
for e in "ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33".split(','): fam[e]='A'
for e in "ep-02,ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32".split(','): fam[e]='B'
for e in "ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31".split(','): fam[e]='C'
print(len(new),'new rows; used max',max(r.get('used') or 0 for r in new.values()))
for e in sorted(fam,key=lambda e:(fam[e],e)):
    s=f"{fam[e]} {e}"
    for i in ('L01','N01'):
        if (e,i) in new:
            a,b=old[(e,i)],new[(e,i)]['response']
            s+=f" | {i} lcp={lcp(a,b):3d} cov8={lcs_run(a,b):3d} len={len(b.split()):3d}"
    print(s)
