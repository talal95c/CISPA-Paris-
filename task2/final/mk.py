import sys, itertools, math
FAM = {'A':"01 04 12 14 17 18 23 24 25 33".split(), 'B':"02 03 06 07 08 20 21 22 26 28 30 32".split(), 'C':"05 09 10 11 13 15 16 19 27 29 31".split()}
F = {e:f for f,l in FAM.items() for e in l}
def labels(groups):
    """groups: 'a b c|d e f' -> {ep: model label}"""
    lab = {e: f"{F[e]}-s{e}" for e in F}
    for i,g in enumerate(x for x in groups.split('|') if x.strip()):
        for e in g.split(): lab[e] = f"{F[g.split()[0]]}-g{i}"
    return lab
def write(path, groups):
    lab = labels(groups)
    with open(path,'w') as f:
        f.write("endpoint,family,model\n")
        for e in sorted(F): f.write(f"ep-{e},{F[e]},{lab[e]}\n")
def mcc_model(pred, truth):
    p, t = labels(pred), labels(truth)
    tp=fp=fn=tn=0
    for a,b in itertools.combinations(sorted(F),2):
        if F[a]!=F[b]: continue
        x, y = p[a]==p[b], t[a]==t[b]
        tp+=x and y; fp+=x and not y; fn+=y and not x; tn+=(not x) and (not y)
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (max(0,(tp*tn-fp*fn)/d) if d else 0), (tp,fp,fn,tn)
def score(pred, truth): return 0.1+0.9*mcc_model(pred,truth)[0]
if __name__=="__main__":
    write(sys.argv[1], sys.argv[2]); print(open(sys.argv[1]).read())
