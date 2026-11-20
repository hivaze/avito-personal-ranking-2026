"""e1b: members on disk that are NOT in the cache (not 10-epoch class, no test file, or an @last twin of a from-scratch run): names and file counts only. Output out/inventory_outside.md / .json."""
from ens_common import *
import collections
idx = json.load(open(CACHE / 'index.json')); incache = set(idx['to_val'])
mem = collections.defaultdict(dict)
for d in (P, W, AB):
    for f in os.listdir(d):
        if not (f.startswith('nn_') and f.endswith('.npy')): continue
        for k in ('to_val', 'to_test', 'ft_val', 'ft_test'):
            if f.endswith('_' + SUF[k] + '.npy'): mem[f[3:-len(SUF[k]) - 5]][k] = 1
    for f in os.listdir(d / 'holdout'):
        if f.startswith('nn_') and f.endswith('_holdout.npy'): mem[f[3:-12]]['hold'] = 1
out = [n for n in mem if n not in incache]
def fam(n):
    if '@last' in n: return '@last twin of a from-scratch run (same run, last-step checkpoint)'
    if n.startswith('w5_cd_valcd'): return 'w5_cd_valcd (cooldown on val folds, 4 roots x 5 folds)'
    if n.startswith('w5_cd_'): return 'w5_cd_* hard-label cooldown screens on the F7 roots: ' + arm_of(n)
    if re.match(r'^abl_(?!e10|fd)', n): return '2-epoch ablation members (seeds 23, 24): ' + arm_of(n)
    if 'smoke' in n: return 'smoke runs'
    return 'other: ' + arm_of(n)
g = collections.defaultdict(list)
for n in out: g[fam(n)].append(n)
L = ['| family | members | TO val | TO test | FT val | FT test | holdout |', '|---|---|---|---|---|---|---|']; J = {}
for k in sorted(g):
    ns = g[k]; c = {kk: sum(1 for n in ns if kk in mem[n]) for kk in KINDS}; J[k] = dict(n=len(ns), **c)
    L.append(f'| {k} | {len(ns)} | {c["to_val"]} | {c["to_test"]} | {c["ft_val"]} | {c["ft_test"]} | {c["hold"]} |')
json.dump(dict(total_on_disk=len(mem), in_cache=len(incache), outside=J), open(OUT / 'inventory_outside.json', 'w'), indent=0)
open(OUT / 'inventory_outside.md', 'w').write('\n'.join(L) + '\n'); print(len(mem), len(incache), len(out)); print('\n'.join(L))
