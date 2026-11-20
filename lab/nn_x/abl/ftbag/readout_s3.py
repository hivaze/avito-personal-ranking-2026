"""Read-out of the FT seed-bagging task (pre-declared nn_x/abl/ftbag/PREDECLARED_ftbag.txt 2026-11-13T19:08:07Z). CPU only,
read-only on every prediction file. Pool machinery = nn_x/abl/pop1/chain/decide9.py / x8/decide_x8.py (per member per-feed rank-z
of the FT val OOF and of the train-only val preds, branch = rank-z of the member mean, score = 0.5 z(ft) + 0.5 z(to); NDCG per feed
= src/metrics.ndcg_per_feed; every delta paired per feed; SE = sd / sqrt(n_feeds)); the holdout is not involved (train-only members).
  1. MEMBER LEVEL: FT OOF val NDCG of n2cpv3s3 vs n2cpv3 per member (paired per feed +- SE); mean delta +- SE across members
     (sd across members / sqrt(n)) overall and per family = the direct bagging-gain measurement.
  2. POOLS: G3 (20, n2cpv3), Q2a8 = G3 + pop1(8) (n2cpv3), C2 = G3 (n2cpv3s3), C3 = Q2a8 (n2cpv3s3). Contrasts: C2 - G3, C3 - Q2a8
     (the 'same pool with n2cpv3' reads), C3 - G3, Q2a8 - G3 (reference), C3 - C2: V +- SE, pmix (private mix) +- SE, slices
     M<50 / M>=50 / Tue / Thu (+- SE, z), z-corr.
  3. RULE (i): a candidate is built unless its val is below the same pool with n2cpv3 by more than 1 SE (C2 vs G3; C3 vs Q2a8).
Output: prints + nn_x/abl/ftbag/readout_s3.json. Members with missing n2cpv3s3 files are disclosed; a pool needs all its members."""
import sys, json; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
P = Path('<repo>/preds'); OUT = Path('<repo>/nn_x/abl/ftbag/readout_s3.json')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; POP1 = [f'abl_e10_pop1_s{s}' for s in range(951, 959)]
G3 = F10 + F7x + F7x70; Q2A8 = G3 + POP1
FAM = {'F10': F10, 'F7x': F7x, 'F7x70': F7x70, 'pop1': POP1}
S1, S3 = 'n2cpv3', 'n2cpv3s3'
has = lambda m, suf: (P/f'nn_{m}_{suf}_val_oof.npy').exists() and (P/f'nn_{m}_{suf}_test.npy').exists()
miss1 = [m for m in Q2A8 if not has(m, S1) or not (P/f'nn_{m}_trainonly_val.npy').exists()]
assert not miss1, f'n2cpv3 / train-only files missing for {miss1}'
miss3 = [m for m in Q2A8 if not has(m, S3)]
print(f'n2cpv3s3 complete: {28 - len(miss3)}/28' + (f' MISSING {miss3}' if miss3 else ''))
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); se = lambda d: float(d.std() / np.sqrt(len(d)))
corr = lambda a, b: float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
W = pmix.weights()
cache = {}
def ld(m, suf):
    k = (m, suf)
    if k not in cache:
        cache[k] = (zv(np.load(P/f'nn_{m}_{suf}_val_oof.npy')), zv(np.load(P/f'nn_{m}_trainonly_val.npy')))
    return cache[k]
# ---- 1. member level ----
print('MEMBER LEVEL: FT OOF val NDCG (final EMA preds, raw) n2cpv3 -> n2cpv3s3; paired per-feed delta +- SE; train-only val for reference')
mem = {}
for fam, ms in FAM.items():
    for m in ms:
        a = ev(np.load(P/f'nn_{m}_{S1}_val_oof.npy')); to = ev(np.load(P/f'nn_{m}_trainonly_val.npy')).mean()
        row = dict(family=fam, to_val=float(to), ft1=float(a.mean()))
        if m not in miss3:
            b = ev(np.load(P/f'nn_{m}_{S3}_val_oof.npy')); d = b - a
            row.update(ft3=float(b.mean()), d=float(d.mean()), d_se=se(d), corr=corr(ld(m, S1)[0], ld(m, S3)[0]))
            print(f'  {m:22s} {fam:6s} to {to:.5f} | FT1 {a.mean():.5f} FT3 {b.mean():.5f} | d {d.mean():+.5f}+-{se(d):.5f} (z {d.mean()/max(se(d),1e-9):+.2f}) | z-corr FT1/FT3 {row["corr"]:.4f}')
        else:
            print(f'  {m:22s} {fam:6s} to {to:.5f} | FT1 {a.mean():.5f} FT3 MISSING')
        mem[m] = row
done = [m for m in Q2A8 if m not in miss3]
summ = {}
def msum(name, ms):
    ds = np.array([mem[m]['d'] for m in ms if 'd' in mem[m]])
    if len(ds) == 0: return
    s = float(ds.std(ddof=1) / np.sqrt(len(ds))) if len(ds) > 1 else float('nan')
    summ[name] = dict(n=int(len(ds)), mean=float(ds.mean()), se_across=s, min=float(ds.min()), max=float(ds.max()), n_pos=int((ds > 0).sum()))
    print(f'  BAGGING GAIN {name:6s} n={len(ds):2d}: mean d {ds.mean():+.5f} +- {s:.5f} (SE across members; min {ds.min():+.5f} max {ds.max():+.5f}; {int((ds > 0).sum())}/{len(ds)} positive)')
for fam, ms in FAM.items(): msum(fam, ms)
msum('G3', G3); msum('ALL28', Q2A8)
# ---- 2. pools ----
def pool(ms, suf):
    ft = zv(np.mean([ld(m, suf)[0] for m in ms], 0)); to = zv(np.mean([ld(m, suf)[1] for m in ms], 0))
    sv = 0.5 * ft + 0.5 * to
    return ev(sv), sv
C = {'G3': (G3, S1), 'Q2a8': (Q2A8, S1), 'C2=G3(s3)': (G3, S3), 'C3=Q2a8(s3)': (Q2A8, S3)}
R = {}
for k, (ms, suf) in C.items():
    if any(m in miss3 for m in ms) and suf == S3:
        print(f'POOL {k}: NOT computed (missing members)'); continue
    R[k] = pool(ms, suf)
    print(f'POOL {k:12s} n={len(ms):2d} FT {suf:9s} val {R[k][0].mean():.5f}')
def contrast(a, b):
    dv = R[a][0] - R[b][0]; pm, pse = pmix.wmean_se(dv, W['private'])
    sl = {n: dict(d=float(dv[m].mean()), se=se(dv[m]), z=float(dv[m].mean() / max(se(dv[m]), 1e-9)), n=int(m.sum())) for n, m in SL}
    o = dict(V=float(dv.mean()), V_se=se(dv), z=float(dv.mean() / max(se(dv), 1e-9)), pmix=float(pm), pmix_se=float(pse), slices=sl,
             zcorr=corr(zv(R[a][1]), zv(R[b][1])))
    print(f'  {a} - {b}: V {o["V"]:+.5f}+-{o["V_se"]:.5f} (z {o["z"]:+.2f}) pmix {pm:+.5f}+-{pse:.5f} (z {pm/max(pse,1e-9):+.2f}) | '
          + ' '.join(f'{n} {s["d"]:+.5f}+-{s["se"]:.5f}({s["z"]:+.1f})' for n, s in sl.items()) + f' | z-corr {o["zcorr"]:.4f}')
    return o
print('CONTRASTS (paired per feed; slices: delta +- SE (z)):')
con = {}
for a, b in [('C2=G3(s3)', 'G3'), ('C3=Q2a8(s3)', 'Q2a8'), ('C3=Q2a8(s3)', 'G3'), ('Q2a8', 'G3'), ('C3=Q2a8(s3)', 'C2=G3(s3)')]:
    if a in R and b in R: con[f'{a} - {b}'] = contrast(a, b)
# ---- 3. rule (i) ----
rule = {}
for cand, ref, tag, out in [('C2=G3(s3)', 'G3', 'G3s3', 'cand_G3_ftbag3_memeq.parquet'), ('C3=Q2a8(s3)', 'Q2a8', 'Q2a8s3', 'cand_Q2a8_ftbag3_memeq.parquet')]:
    key = f'{cand} - {ref}'
    if key not in con: rule[cand] = dict(build=False, reason='pool incomplete'); print(f'RULE {cand}: pool incomplete -> not built'); continue
    o = con[key]; ok = o['V'] >= -o['V_se']
    rule[cand] = dict(build=bool(ok), V=o['V'], V_se=o['V_se'], tag=tag, out=out, members=C[cand][0])
    print(f'RULE (i) {cand}: V - V({ref}) = {o["V"]:+.5f} >= -1 SE ({-o["V_se"]:+.5f}): {ok} -> ' + (f'BUILD TAG {tag} subs/{out} + presub_gate' if ok else 'NOT built (below the n2cpv3 pool by > 1 SE)'))
json.dump(dict(missing_s3=miss3, members=mem, bagging=summ, pools={k: float(v[0].mean()) for k, v in R.items()}, contrasts=con, rule=rule), open(OUT, 'w'), indent=1)
print('written', OUT)
