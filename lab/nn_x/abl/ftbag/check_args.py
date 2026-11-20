"""Verify, for every member, that the logged args of the n2cpv3s3 fine-tune equal the member's recorded n2cpv3 args except
'out' (suffix) and 'seeds' (1 -> 3), as pre-declared (PREDECLARED_ftbag.txt). Reads the '==== ... args: {...}' line of
nn/logs/<m>_<suf>_ft.log (G3: F10 in nn/logs, r6 in nn_x/r6/logs; pop1 in nn_x/abl/logs). Also reports wall minutes and the
final-EMA OOF line of each s3 log. Exit 1 on any mismatch."""
import ast, re, sys
from pathlib import Path
R = Path('<repo>')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; POP1 = [f'abl_e10_pop1_s{s}' for s in range(951, 959)]
logdir = {**{m: R/'nn/logs' for m in F10}, **{m: R/'nn_x/r6/logs' for m in F7x + F7x70}, **{m: R/'nn_x/abl/logs' for m in POP1}}
def args_of(p):
    line = p.read_text().splitlines()[0]
    return ast.literal_eval(line.split('args: ', 1)[1])
bad = 0
for m, d in logdir.items():
    p1, p3 = d/f'{m}_n2cpv3_ft.log', d/f'{m}_n2cpv3s3_ft.log'
    if not p3.exists(): print(f'{m:22s} s3 log MISSING'); bad += 1; continue
    a1, a3 = args_of(p1), args_of(p3)
    diff = {k: (a1.get(k), a3.get(k)) for k in set(a1) | set(a3) if a1.get(k) != a3.get(k)}
    ok = set(diff) == {'out', 'seeds'} and a3['seeds'] == 3 and a3['out'] == f'{m}_n2cpv3s3' and a1['out'] == f'{m}_n2cpv3'
    txt = p3.read_text()
    fin = re.search(r'final EMA weights: OOF ndcg ([0-9.]+)', txt); mins = re.search(r'fold 4: .* ([0-9.]+) min', txt)
    print(f'{m:22s} args {"OK" if ok else "MISMATCH " + str(diff)} | s3 final-EMA OOF {fin.group(1) if fin else "pending"} | {mins.group(1) + " min" if mins else "running"}')
    bad += 0 if ok else 1
print('ALL OK' if bad == 0 else f'{bad} PROBLEM(S)'); sys.exit(1 if bad else 0)
