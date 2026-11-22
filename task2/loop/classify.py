"""Apply the reference-validated white-box rule to the 33 scored endpoints and write a submission CSV.
Rule: stop-hazard z < ZL -> LoRA (own model); else fp8 LLR z < ZF -> fp8 model of the family; else base model.
Usage: classify.py out.csv"""
import sys, os, numpy as np
from wb_fit import load
from wb_feat import stack, Tfit, llT
from wb_eos import caps, hazard
ZL, ZF = -4.0, 0.0
FAM = {'A': ('a_cache,a_long,a_short', 'a_q3_fp8,a_long_fp8,a_short_fp8'),
       'B': ('b_cache,b_long,b_short', 'b_q15_fp8,b_long_fp8,b_short_fp8'),
       'C': ('c_cache,c_long,c_short', 'c_llama1_fp8,c_long_fp8,c_short_fp8')}
have = lambda names: ','.join(n for n in names.split(',') if os.path.exists(f'wb/{n}.jsonl') and 'done' in open(f'wb/{n}.log').read())

def table(fam):
    b, f = (have(x) for x in FAM[fam]); D = load(b); F = load(f); cap = caps(D); out = {}
    for e in sorted({k[0] for k in D}):
        ks = [k for k in D if k[0] == e]; kf = [k for k in ks if k in F]
        T = Tfit(*stack(D, ks)); h = hazard(D, e, T, cap, set(ks))
        Tb = Tfit(*stack(D, kf)); Tf = Tfit(*stack(F, kf))
        d = np.array([llT(D[k], Tb).sum() - llT(F[k], Tf).sum() for k in kf])
        out[e] = dict(n=len(ks), nf=len(kf), T=T, zs=h['z'], early=h['early'], zf=d.mean() / (d.std() + 1e-12) * np.sqrt(len(d)),
                      llr=1000 * d.sum() / sum(D[k]['n'] for k in kf))
    return out, b, f

if __name__ == '__main__':
    rows = []
    for fam in 'ABC':
        t, b, f = table(fam); print(f'== family {fam}: base[{b}] fp8[{f}]')
        for e, v in t.items():
            m = f'{fam}-lora-{e}' if v['zs'] < ZL else f'{fam}-fp8' if v['zf'] < ZF else f'{fam}-base'
            rows.append((e, fam, m))
            print(f'  {e} n={v["n"]:3d}/{v["nf"]:3d} T={v["T"]:.3f} stop_z={v["zs"]:6.1f} early={v["early"]:5.1f} fp8_z={v["zf"]:6.2f} llr/ktok={v["llr"]:7.2f} -> {m}')
    rows.sort(); assert len(rows) == 33 and len({r[0] for r in rows}) == 33
    with open(sys.argv[1], 'w') as fh:
        fh.write('endpoint,family,model\n'); [fh.write(','.join(r) + '\n') for r in rows]
    print('wrote', sys.argv[1])
