"""Paired per-feed comparisons of the main val results (exact metric)."""
from xgbexp import *
import os
dva = pl.read_parquet(f'{OUT}/feat_val.parquet', columns=['fid', 'target'])
gv = groups(dva); ov = offsets(gv); tv = dva['target'].to_numpy()
def pf(fname, ck):
    p = f'{OUT}/{fname}.npz'
    if not os.path.exists(p): return None
    return ndcg_pf(np.load(p)[str(ck)].astype(np.float64), tv, ov, gv)
f1 = np.asarray(cand('val', 'f1')).astype(np.float64); f5 = np.asarray(cand('val', 'f5')).astype(np.float64)
base = {'f1+1e-3f5': ndcg_pf(f1 + 1e-3 * f5, tv, ov, gv)}
items = {
    'a_basic_rank@200': ('pred_a_basic_rank', 200), 'a_basic_binary@200': ('pred_a_basic_binary', 200), 'a_rich_rank@200': ('pred_a_rich_rank', 200),
    'b_basic_pos20@400': ('pred_b_basic_pos_rank__pos20', 400), 'b_basic_pos0@400': ('pred_b_basic_pos_rank__pos0', 400),
    'b_basic_pos50_rkend50@400': ('pred_b_basic_pos_rkend_rank__pos50_rkend50', 400),
    'b_rich_pos20@200': ('pred_b_rich_pos_rank__pos20', 200), 'b_rich_pos5@400': ('pred_b_rich_pos_rank__pos5', 400), 'b_rich_pos0@400': ('pred_b_rich_pos_rank__pos0', 400),
    'b_rich_pos20_rkend50@400': ('pred_b_rich_pos_rkend_rank__pos20_rkend50', 400), 'b_rich_pos20_rkend50@600': ('pred_b_rich_pos_rkend_rank__pos20_rkend50', 600),
    'a_rich_rank@600': ('pred_a_rich_rank', 600), 'b_basic_pos50_rkend50@600': ('pred_b_basic_pos_rkend_rank__pos50_rkend50', 600),
    'c_val_basic@100': ('pred_c_val_basic', 100), 'c_val_rich@200': ('pred_c_val_rich', 200), 'c_val_rich_binary@200': ('pred_c_val_rich_binary', 200),
    'd_w1@400': ('pred_d_train200k+val_w1_rich', 400), 'd_w5@400': ('pred_d_train200k+val_w5_rich', 400), 'd_w20@400': ('pred_d_train200k+val_w20_rich', 400),
    'e_finetune@200': ('pred_e_pretrain300_finetune_rich', 200), 'e_finetune@400': ('pred_e_pretrain300_finetune_rich', 400),
}
P = dict(base)
for k, (fn, ck) in items.items():
    v = pf(fn, ck)
    if v is not None: P[k] = v
res = {k: float(v.mean()) for k, v in P.items()}
pairs = [('b_basic_pos20@400', 'a_basic_rank@200'), ('b_basic_pos0@400', 'a_basic_rank@200'), ('b_basic_pos50_rkend50@400', 'a_basic_rank@200'),
         ('b_rich_pos20@200', 'a_rich_rank@200'), ('b_rich_pos5@400', 'a_rich_rank@200'), ('b_rich_pos0@400', 'a_rich_rank@200'), ('b_rich_pos20_rkend50@400', 'a_rich_rank@200'), ('b_rich_pos20_rkend50@600', 'a_rich_rank@200'), ('b_rich_pos20_rkend50@600', 'a_rich_rank@600'), ('b_rich_pos20_rkend50@600', 'b_rich_pos20@200'),
         ('a_basic_rank@200', 'c_val_basic@100'), ('a_rich_rank@200', 'c_val_rich@200'), ('a_rich_rank@200', 'c_val_rich_binary@200'),
         ('a_basic_rank@200', 'f1+1e-3f5'), ('c_val_basic@100', 'f1+1e-3f5'),
         ('d_w1@400', 'a_rich_rank@200'), ('d_w5@400', 'a_rich_rank@200'), ('d_w20@400', 'a_rich_rank@200'), ('e_finetune@200', 'a_rich_rank@200'), ('e_finetune@400', 'a_rich_rank@200'),
         ('d_w5@400', 'c_val_rich_binary@200')]
for f in ['pf_g1_train200k_rich_w1+4p_400', 'pf_g3_train200k_rich_Mreweight_400', 'pf_g2_Rset_rich_pos20_400', 'pf_g2_Rset_rich_pos5_400', 'pf_g1_train200k_rich_w1+4p_200', 'pf_g3_train200k_rich_Mreweight_200']:
    if os.path.exists(f'{OUT}/{f}.npy'):
        P[f] = np.load(f'{OUT}/{f}.npy'); res[f] = float(P[f].mean()); pairs.append((f, 'a_rich_rank@200'))
for fs in ['basic', 'rich']:
    for nm in ['R_random_like', 'P_prod_like', 'U_uniform']:
        f = f'{OUT}/pf_{nm}_{fs}.npy'
        if os.path.exists(f): P[f'{nm}_{fs}'] = np.load(f); res[f'{nm}_{fs}'] = float(P[f'{nm}_{fs}'].mean())
    pairs += [(f'R_random_like_{fs}', f'P_prod_like_{fs}'), (f'R_random_like_{fs}', f'U_uniform_{fs}'), (f'U_uniform_{fs}', f'P_prod_like_{fs}')]
if 'pf_g2_Rset_rich_pos20_400' in P: pairs.append(('pf_g2_Rset_rich_pos20_400', 'R_random_like_rich'))
out = {}
for a, b in pairs:
    if a in P and b in P:
        m, se = paired(P[a], P[b]); out[f'{a} - {b}'] = [round(m, 5), round(se, 5)]
        print(f'{a:40s} - {b:28s}: {m:+.4f} +- {se:.4f}')
print({k: round(v, 4) for k, v in res.items()})
json.dump({'means': res, 'paired': out}, open(f'{OUT}/10_paired.json', 'w'), indent=1)
