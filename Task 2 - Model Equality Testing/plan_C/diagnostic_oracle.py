"""135M A only: diagnostic ablation, never modifies the freeze."""
import json
from pathlib import Path

import numpy as np

from decision import SAME, VARIANTS, decide, score
from phase0 import draw, probabilities, schedule


def main():
    root=Path(__file__).resolve().parent
    params=json.loads((root/'frozen_params.json').read_text())
    params=params | dict(enable_temperature=False,enable_watermark=False)
    p,ids=probabilities('135m','tune',params['active_items'],two_codes=False)
    rng=np.random.default_rng(821)
    rng.random((1,9,200))  # Match the already reported q=.9 sample stream.
    values=[]
    for _ in range(200):
        _,_,n=schedule(ids,params['active_items'],rng)
        raw=draw(p,n,rng)
        keep=[0]+sorted(rng.choice(np.arange(1,9),int(rng.integers(6,9)),replace=False).tolist())
        truth=['S' if VARIANTS[v] in SAME else VARIANTS[v] for v in keep]
        flags=[VARIANTS[v] in SAME and v!=0 for v in keep]
        family,model=decide(raw[keep,:,: ,0],raw[keep].sum(1)[:,:22],flags,params,'R')
        values.append(score([0]*len(keep),truth,family.tolist(),model.tolist())['score'])
    truth=[('S' if v in SAME else v) for v in VARIANTS]
    baseline=score([0]*9,truth,[0]*9,list(range(9)))['score']
    families=[0]*9+[1]*9
    models=truth+truth
    perfect=score(families,models,families,models)['score']
    assert perfect==1.
    result=dict(R_oracle=dict(median=float(np.median(values)),q10=float(np.quantile(values,.1))),singletons=baseline,perfect_oracle_two_family_fixture=perfect)
    (root/'oracle_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"R oracle flags, 135M A: median={np.median(values):.4f}, q10={np.quantile(values,.1):.4f} (200 replays)")
    print(f'Singletons except true families, 135M A: score={baseline:.4f}')
    print(f'Perfect oracle scorer control, two-family fixture: score={perfect:.4f} (single-family convention: 0.9000)')


if __name__=='__main__':
    main()
