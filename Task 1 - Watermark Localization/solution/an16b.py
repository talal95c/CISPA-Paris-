from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
z=np.load(sys.argv[1],allow_pickle=True); n=len(z["names"])
def Lmap(e1,e0,k,eps=.8): return np.log((cv2.blur(e1,(k,k))+eps)/(cv2.blur(e0,(k,k))+eps))
for i in np.argsort(z["lr"]):
    g=z["gt"][i].astype(bool)
    a=[roc_auc_score(g.ravel(),-Lmap(z[e1][i],z[e0][i],k).ravel()) if 0<g.mean()<1 else np.nan for e1,e0,k in (("e1","e0",8),("e1","e0",16),("e1a","e0a",16))]
    print("lr %.2f lr1 %.2f frame %s%.2f@%.2f marked %.2f  auc k8 %.2f k16 %.2f | aspect1 k16 %.2f"%(z["lr"][i],z["lr1"][i],"HW"[int(z["frame"][i][0])],z["frame"][i][1],z["frame"][i][2],g.mean(),*a))
