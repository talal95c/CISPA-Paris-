import sys, pickle, numpy as np
R = pickle.load(open(sys.argv[1], 'rb')); thr = int(sys.argv[3])
for e in sys.argv[2].split(','):
    A = [d for d in R if d['ep'] == 'ep-' + e]
    rk = np.concatenate([d['rank'] for d in A])
    print(e, 'tok', len(rk), 'hist', {t: int((rk >= t).sum()) for t in (10, 20, 30, 40, 50, 60, 80, 100, 200, 1000)})
    for d in A:
        for j in np.where(d['rank'] >= thr)[0]:
            print('   ', d['id'], 'pos', j, '/', d['n'], 'rank', int(d['rank'][j]), 'p95@1 out', bool(d['out']['p95@1'][j]), 'mass_k50@T1 %.3f' % d['mass']['k50'][1.0][j])
