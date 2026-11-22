"""mkcsv.py out.csv "23 33 04|20 30 06|16 29 31 10"  -> submission with these groups, every other endpoint a singleton."""
import sys, csv
FAM = dict(A='01 04 12 14 17 18 23 24 25 33'.split(), B='02 03 06 07 08 20 21 22 26 28 30 32'.split(), C='05 09 10 11 13 15 16 19 27 29 31'.split())
fam = {e: f for f, v in FAM.items() for e in v}; lab = {}
for i, g in enumerate(sys.argv[2].split('|')):
    es = g.split(); assert len({fam[e] for e in es}) == 1, g
    for e in es: assert e not in lab; lab[e] = f'{fam[e]}-g{i}'
w = csv.writer(open(sys.argv[1], 'w', newline='')); w.writerow(['endpoint', 'family', 'model'])
for n in range(1, 34):
    e = f'{n:02d}'; w.writerow([f'ep-{e}', fam[e], lab.get(e, f'{fam[e]}-s{e}')])
print(sys.argv[1], sorted(set(lab.values())))
