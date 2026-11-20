"""e1: inventory by arm (out/inventory_arms.json, out/inventory_arms.md). Reads out/inventory.json + the cache.
Per arm: members, seeds, full-data flag, files present, train-only val (member mean, arm pool), fine-tuned val (member mean, arm pool), holdout (member mean, pool;
out-of-sample only for arms NOT trained on the holdout feeds), mean pairwise within-arm correlation of per-feed z on train-only val."""
from ens_common import *
import collections
m = Meta(); st = Store(m); inv = json.load(open(OUT / 'inventory.json'))['members']
arms = collections.defaultdict(list)
for n in sorted(inv): arms[inv[n]['arm']].append(n)
def corr_within(kind, ns):
    if len(ns) < 2: return None
    X = st.mat(kind, ns).astype(np.float64); C = np.corrcoef(X); return float(C[np.triu_indices(len(ns), 1)].mean())
rows = {}
for a, ns in arms.items():
    fd = {inv[n]['full_data'] for n in ns}; ft = [n for n in ns if st.has('ft_val', n)]; hd = [n for n in ns if st.has('hold', n)]
    r = dict(n=len(ns), seeds=[re.sub(r'^.*_(s|r)(\d+)(@last)?$', r'\2', n) for n in ns], full_data=(fd.pop() if len(fd) == 1 else 'mixed'),
             n_to=len(ns), n_ft=len(ft), n_hold=len(hd),
             to_member=float(np.mean([inv[n]['to_val']['ndcg'] for n in ns])), to_pool=float(m.ev(st.pool('to_val', ns)).mean()),
             to_corr=corr_within('to_val', ns))
    if ft:
        r.update(ft_member=float(np.mean([inv[n]['ft_val']['ndcg'] for n in ft])), ft_pool=float(m.ev(st.pool('ft_val', ft)).mean()), ft_corr=corr_within('ft_val', ft))
    if hd:
        r.update(hold_member=float(np.mean([inv[n]['hold']['ndcg'] for n in hd])), hold_pool=float(m.eh(st.pool('hold', hd)).mean()))
    rows[a] = r
json.dump(rows, open(OUT / 'inventory_arms.json', 'w'), indent=0)
f4 = lambda x: '' if x is None else f'{x:.5f}'
L = ['| arm | n | seeds | all data | TO val+test | FT val+test | holdout | TO val member | TO val pool | within corr | FT val member | FT val pool | holdout member | holdout pool |', '|' + '---|' * 14]
for a in sorted(rows, key=lambda a: -rows[a]['to_pool']):
    r = rows[a]; s = r['seeds']; ss = f'{s[0]}-{s[-1]}' if len(s) > 2 and all(x.isdigit() for x in s) and int(s[-1]) - int(s[0]) == len(s) - 1 else ','.join(s)
    L.append(f"| {a} | {r['n']} | {ss} | {r['full_data']} | {r['n_to']} | {r['n_ft']} | {r['n_hold']} | {f4(r['to_member'])} | {f4(r['to_pool'])} | {'' if r['to_corr'] is None else format(r['to_corr'], '.4f')} | "
             f"{f4(r.get('ft_member'))} | {f4(r.get('ft_pool'))} | {f4(r.get('hold_member'))} | {f4(r.get('hold_pool'))} |")
open(OUT / 'inventory_arms.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
