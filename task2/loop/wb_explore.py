import json,sys,numpy as np
from wb_fit import load,GRID,kind
D=load(sys.argv[1])
G=np.array(GRID)
def Tfit(zx,logZ):  # logZ: tok x T
    L=(zx[:,None]/G[None,:]-logZ).sum(0); i=int(L.argmax())
    if 0<i<len(G)-1:
        # parabola through 3 points in 1/T
        x=1/G[i-1:i+2]; c=np.polyfit(x,L[i-1:i+2],2); return 1/(-c[1]/(2*c[0]))
    return G[i]
for e in sorted({k[0] for k in D},key=lambda e:kind[e]):
    ks=[k for k in D if k[0]==e]
    zx=np.concatenate([D[k]['zx'] for k in ks])
    logZ=np.concatenate([np.array([D[k]['g'][str(T)]['logZ'] for T in GRID]).T for k in ks])
    p1=np.concatenate([D[k]['g']['1.0']['p1'] for k in ks])
    rank=np.concatenate([D[k]['rank'] for k in ks])
    T=Tfit(zx,logZ)
    s=f'{e} {kind[e]:<15} T={T:.3f}'
    for lo,hi in ((0.9,1.01),(0.6,0.9),(0.3,0.6),(0,0.3)):
        m=(p1>=lo)&(p1<hi)
        s+=f' | p1[{lo},{hi}) n={m.sum():5d} T={Tfit(zx[m],logZ[m]):.2f} top1={np.mean(rank[m]==0):.3f} r>=10:{np.mean(rank[m]>=10):.4f}'
    s+=f' | rank>=20 {np.mean(rank>=20):.4f} rank>=100 {np.mean(rank>=100):.4f}'
    print(s)
