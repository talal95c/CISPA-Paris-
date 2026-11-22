import json,sys,csv,collections,numpy as np
fam={r['endpoint']:r['model'] for r in csv.DictReader(open('/home/ubuntu/ref-evaluation/audit/submission_audit.csv'))}
kind={r['endpoint']:r['model']+':'+r['kind'][:12] for r in csv.DictReader(open('/home/ubuntu/met/practice_answers.csv'))}
names=sys.argv[1:]
T={}
for n in names:
    for l in open(f'wb/{n}.jsonl'):
        x=json.loads(l)
        pass
        lp=np.array(x['lp']); rk=np.array(x['rank'])
        T.setdefault(x['endpoint'],{}).setdefault(n,[]).append((lp.mean(),(rk==0).mean(),np.median(lp)))
print('endpoint        '+'  '.join(f'{n:>22}' for n in names))
for e in sorted(T,key=lambda e:(fam.get(e,kind.get(e,'')),e)):
    row=[]
    for n in names:
        v=np.array(T[e].get(n,[(np.nan,)*3]))
        row.append(f'{v[:,0].mean():7.3f} {v[:,1].mean():5.3f} n{len(v):<3}')
    print(f'{e} {fam.get(e,kind.get(e,"")):<18}'+'  '.join(f'{r:>22}' for r in row))
