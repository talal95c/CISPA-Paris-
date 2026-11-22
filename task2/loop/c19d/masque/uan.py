import numpy as np
from mc import ix,eps,S8M,FP8,FAM,famof,pi,pj
d=np.load('umc_both_crp0.5_0.npz');Z=d['Z'];U=d['U'];print(len(Z),'taille U moy %.1f'%U.sum(1).mean(),'part publique %.2f'%(U[:,pi]&U[:,pj]).mean())
print('P(dans U):');
for f in 'ABC': print('  ',f,' '.join('%02d:%.2f'%(e,U[:,ix[e]].mean()) for e in FAM[f]))
anc={0:23,1:20,2:16}
O=np.stack([Z[:,ix[e]]==Z[:,ix[anc[famof[ix[e]]]]] for e in eps],1)
print('P(meme modele que l ancre 23/20/16) | sachant dans U:')
for f in 'ABC': print('  ',f,' '.join('%02d:%.2f|%.2f'%(e,O[:,ix[e]].mean(),O[U[:,ix[e]],ix[e]].mean() if U[:,ix[e]].any() else -1) for e in FAM[f]))
ok=np.all([O[:,ix[e]] for e in S8M],0)&~O[:,ix[15]]&~O[:,ix[22]]
print('part des solutions ou les 12 membres de S8 sont ensemble et 15/22 dehors: %.3f'%ok.mean())
# S8 members dans U
print('membres S8 dans U (moy):',U[:,[ix[e] for e in S8M]].sum(1).mean(), '| distinct U:',len(np.unique(U,axis=0)))
uu,c=np.unique(U,axis=0,return_counts=True)
for k in np.argsort(-c)[:6]: print(c[k],[e for e in sorted(eps) if uu[k][ix[e]]])
