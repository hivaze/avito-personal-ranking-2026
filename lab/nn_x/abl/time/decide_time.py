"""Pool read-out for the TIME arm family (pre-declared nn_x/abl/ledger/ledger.md section 9; run ONLY after the time screen PASSES and
the 10-epoch family abl_e10_time_s951-954 is fine-tuned). The pool computation is EXACTLY nn_x/abl/pop1/x8/decide_x8.py's (= decide9.py's:
ld / pool / slices / pmix / corr copied verbatim); the candidate set and the rule clauses are the section-9 ones, and the in-period
holdout is ALSO read on its WEEKEND / WEEKDAY slices (the feed's Moscow weekday is Sat / Sun; 9,632 / 20,368 of the 30K feeds):

  decide_time.py Q4   -> G3 (20), Q4 = G3 + time(951-954) [24], member-equal 0.5 z(ft) + 0.5 z(to), paired per feed vs G3.
      Q4 QUALIFIES iff (a) H_we(Q4 - G3) > +1 SE     (holdout WEEKEND slice; PRIMARY)
                       (b) H(Q4 - G3) >= -1 SE       (holdout overall)
                       (c) pmix(Q4 - G3) >= -1 SE    (veto)
                       (d) V(Q4 - G3) >= -2 SE and no val slice (M<50, M>=50, Tue, Thu) < -2 SE   (sanity veto; V is expected ~0)
                       (e) presub_gate PASS (run by the chain only if (a)-(d) hold).
      Information: V, the holdout weekday slice, the family's own member numbers, family-pool z-corr to G3 and to the base w5_e10 pool.
Output: prints + nn_x/abl/time/decide_time_Q4.json ('build' = the qualifier, if any). CPU only, read-only on every prediction file.
Members with incomplete files are dropped (disclosed); the family needs >= 2 members.
"""
import sys, json, os; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
MODE = sys.argv[1] if len(sys.argv) > 1 else 'Q4'
assert MODE == 'Q4', 'usage: decide_time.py Q4'
P = Path('<repo>/preds'); H = P / 'holdout'; OUT = Path(f'<repo>/nn_x/abl/time/decide_time_{MODE}.json')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; G3 = F10 + F7x + F7x70
files = lambda m: [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_trainonly_val.npy', H/f'nn_{m}_holdout.npy', P/f'nn_{m}_n2cpv3_test.npy', P/f'nn_{m}_trainonly_test.npy']
complete = lambda m: all(p.exists() for p in files(m))
PLUMB = os.environ.get('TIME_PLUMB', '')  # plumbing test only (never set by the chain): stand-in members for the time family
if PLUMB: OUT = OUT.with_name(OUT.stem + '_plumbing.json'); print('TIME_PLUMB plumbing test (stand-in family, NOT a read-out):', PLUMB)
def fam(arm, seeds):
    allm = [f'abl_e10_{arm}_s{s}' for s in seeds]
    if PLUMB: allm = PLUMB.split(',')
    ok = [m for m in allm if complete(m)]
    print(f'{arm} seeds {seeds}: complete {len(ok)}/{len(allm)} {ok}' + (f' MISSING {[m for m in allm if m not in ok]}' if len(ok) < len(allm) else ''))
    return ok
miss = [m for m in G3 if not complete(m)]; assert not miss, f'G3 missing {miss}'
PT = fam('time', range(951, 955)); assert len(PT) >= 2, 'fewer than 2 time members complete: no read-out'
C = {'G3': G3, 'Q4=G3+time(4)': G3 + PT}
CAND = ['Q4=G3+time(4)']; REFV = 'G3'
TAGS = {'Q4=G3+time(4)': ('Q4time', 'cand_Q4_G3time_memeq.parquet')}
FAMS = {'time(951-954) to': PT}
# ---- decide9.py machinery, verbatim ----
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT/'val/cand_target.npy')
Mh, oh = np.load(H/'holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(H/'holdout_y.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]; zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(s, yv, ov, Mv); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh); se = lambda d: d.std() / np.sqrt(len(d))
corr = lambda a, b: float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])
cache = {}
def ld(m):
    if m not in cache:
        cache[m] = (zv(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')), zv(np.load(P/f'nn_{m}_trainonly_val.npy')), zh(np.load(H/f'nn_{m}_holdout.npy')))
    return cache[m]
def pool(ms):  # final val score (0.5 z(ft) + 0.5 z(to), member-equal), holdout z-mean; both returned as scores too
    ft = zv(np.mean([ld(m)[0] for m in ms], 0)); to = zv(np.mean([ld(m)[1] for m in ms], 0)); ho = zh(np.mean([ld(m)[2] for m in ms], 0))
    sv = 0.5 * ft + 0.5 * to
    return ev(sv), eh(ho), sv, ho
R = {k: pool(ms) for k, ms in C.items()}
v0, h0, sv0, sh0 = R['G3']
tue = (fv['timestamp'].to_numpy() + 10800) // 86400 < 20720
SL = [('M<50', Mv < 50), ('M>=50', Mv >= 50), ('Tue', tue), ('Thu', ~tue)]
# ---- TIME arm addition: holdout weekend / weekday slices (the rng-12345 subset, as train.py / r6_eval derive it) ----
ftr, Mtr, _ = feed_meta('train')
prand = np.nan_to_num(np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
prand[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((prand > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
assert np.array_equal(Mtr[hold], Mh)
hwe = (((ftr['timestamp'].to_numpy().astype(np.int64)[hold] + 10800) // 86400 + 3) % 7) >= 5
assert int(hwe.sum()) == 9632, int(hwe.sum())
HSL = [('weekend', hwe), ('weekday', ~hwe)]
W = pmix.weights(); out = {}
# ---- per-member information ----
print('MEMBERS (train-only val / holdout NDCG (+ weekend slice), FT n2cpv3 OOF NDCG):')
mem = {}
for name, ms in FAMS.items():
    for m in ms:
        a = ev(np.load(P/f'nn_{m}_trainonly_val.npy')).mean(); hh = eh(np.load(H/f'nn_{m}_holdout.npy')); b = hh.mean(); c = ev(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')).mean()
        mem[m] = dict(to_val=float(a), hold=float(b), hold_weekend=float(hh[hwe].mean()), hold_weekday=float(hh[~hwe].mean()), ft_oof=float(c))
        print(f'  {m:22s} to val {a:.5f} hold {b:.5f} (weekend {hh[hwe].mean():.5f} weekday {hh[~hwe].mean():.5f}) | FT OOF {c:.5f}')
# ---- pools vs G3 ----
for k, (v, h, sv, sh) in R.items():
    dv, dh = v - v0, h - h0
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    dsl = {n: (float(dv[m].mean()), float(se(dv[m]))) for n, m in SL}
    hsl = {n: dict(delta=float(dh[m].mean()), se=float(se(dh[m])), z=float(dh[m].mean() / max(se(dh[m]), 1e-9)), n=int(m.sum())) for n, m in HSL}
    pm, pse = pmix.wmean_se(dv, W['private'])
    cz = (corr(zv(sv), zv(sv0)), corr(zh(sh), zh(sh0))) if k != 'G3' else (1.0, 1.0)
    out[k] = dict(n=len(C[k]), members=C[k], val=float(v.mean()), hold=float(h.mean()), hold_weekend=float(h[hwe].mean()), hold_weekday=float(h[~hwe].mean()),
                  V=float(dv.mean()), V_se=float(se(dv)), H=float(dh.mean()), H_se=float(se(dh)), H_slices=hsl,
                  pmix=float(pm), pmix_se=float(pse), slices_z=zs, slices=dsl, zcorr_to_G3_val=cz[0], zcorr_to_G3_hold=cz[1])
    print(f'{k:16s} n={len(C[k]):2d} val {v.mean():.5f} hold {h.mean():.5f} (we {h[hwe].mean():.5f} wd {h[~hwe].mean():.5f}) | V {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean()/max(se(dv),1e-9):+.2f}) '
          f'H {dh.mean():+.5f}+-{se(dh):.5f} (z {dh.mean()/max(se(dh),1e-9):+.2f}) H_we {hsl["weekend"]["delta"]:+.5f}+-{hsl["weekend"]["se"]:.5f} (z {hsl["weekend"]["z"]:+.2f}) '
          f'H_wd {hsl["weekday"]["delta"]:+.5f}+-{hsl["weekday"]["se"]:.5f} (z {hsl["weekday"]["z"]:+.2f}) pmix {pm:+.5f}+-{pse:.5f} (z {pm/max(pse,1e-9):+.2f}) | '
          + ' '.join(f'{n} {d:+.5f}({z:+.1f})' for (n, (d, s_)), z in zip(dsl.items(), zs.values()))
          + (f' | z-corr to G3 val {cz[0]:.4f} hold {cz[1]:.4f}' if k != 'G3' else ''))
verdict = {}
for k in CAND:
    o = out[k]; we = o['H_slices']['weekend']
    cl = {'a_Hweekend_gt_1SE': we['delta'] > we['se'],
          'b_H_ge_m1SE': o['H'] >= -o['H_se'],
          'c_pmix_ge_m1SE': o['pmix'] >= -o['pmix_se'],
          'd_V_ge_m2SE_and_slices': (o['V'] >= -2 * o['V_se']) and (min(o['slices_z'].values()) >= -2)}
    q = all(cl.values())
    verdict[k] = dict(clauses=cl, ref=REFV, qualifies_a_to_d=bool(q), tag=TAGS[k][0], out=TAGS[k][1])
    print(f'RULE {k} (ledger section 9; ref = {REFV}):')
    print(f"  (a) H_weekend(Q4 - G3) = {we['delta']:+.5f} > +1 SE {we['se']:.5f} (z {we['z']:+.2f}, n {we['n']}): {cl['a_Hweekend_gt_1SE']}   [PRIMARY]")
    print(f"  (b) H(Q4 - G3) = {o['H']:+.5f} >= -1 SE {o['H_se']:.5f} (z {o['H']/max(o['H_se'],1e-9):+.2f}): {cl['b_H_ge_m1SE']}")
    print(f"  (c) pmix(Q4 - G3) = {o['pmix']:+.5f} >= -1 SE {o['pmix_se']:.5f}: {cl['c_pmix_ge_m1SE']}   [veto]")
    print(f"  (d) V(Q4 - G3) = {o['V']:+.5f} >= -2 SE {o['V_se']:.5f} and min val slice z {min(o['slices_z'], key=o['slices_z'].get)} {min(o['slices_z'].values()):+.2f} >= -2: {cl['d_V_ge_m2SE_and_slices']}   [sanity veto]")
    print(f"  (e) presub_gate: pending (run by the chain only if (a)-(d) hold)")
    print(f"  info: H_weekday(Q4 - G3) = {o['H_slices']['weekday']['delta']:+.5f}+-{o['H_slices']['weekday']['se']:.5f} (z {o['H_slices']['weekday']['z']:+.2f})")
    print(f"  => {k}: {'QUALIFIES on (a)-(d) -> build TAG ' + TAGS[k][0] + ' subs/' + TAGS[k][1] + ' + presub_gate (PUBLIC PROBE)' if q else 'does NOT qualify -> nothing built'}")
# ---- family information ----
info = {}
g3v = zv(np.mean([ld(m)[1] for m in G3], 0)); g3f = zv(np.mean([ld(m)[0] for m in G3], 0)); g3h = zh(np.mean([ld(m)[2] for m in G3], 0))
for name, ms in FAMS.items():
    tv, th, tf = zv(np.mean([ld(m)[1] for m in ms], 0)), zh(np.mean([ld(m)[2] for m in ms], 0)), zv(np.mean([ld(m)[0] for m in ms], 0))
    info[f'{name} vs G3'] = dict(val=corr(tv, g3v), hold=corr(th, g3h), ft_val=corr(tf, g3f))
    print(f'  family pool z-corr to the G3 pool: {name:20s} train-only val {corr(tv, g3v):.4f} holdout {corr(th, g3h):.4f} | ft val {corr(tf, g3f):.4f}')
    base = [f'w5_e10_base_s{s}' for s in range(951, 955)]
    if all(complete(m) for m in base):
        bv, bh = zv(np.mean([ld(m)[1] for m in base], 0)), zh(np.mean([ld(m)[2] for m in base], 0))
        info[f'{name} vs w5_e10_base(951-954)'] = dict(val=corr(tv, bv), hold=corr(th, bh))
        print(f'  family pool z-corr to the base w5_e10 pool (951-954): train-only val {corr(tv, bv):.4f} holdout {corr(th, bh):.4f}')
        pb = pool(base); pt = pool(ms); d = pt[0] - pb[0]; dh = pt[1] - pb[1]
        info['time(4) vs base(4) pools'] = dict(V=float(d.mean()), V_se=float(se(d)), H=float(dh.mean()), H_se=float(se(dh)),
                                                H_weekend=float(dh[hwe].mean()), H_weekend_se=float(se(dh[hwe])), H_weekday=float(dh[~hwe].mean()), H_weekday_se=float(se(dh[~hwe])))
        print(f"  time(4) pool vs base w5_e10(4) pool (same strength test): V {d.mean():+.5f}+-{se(d):.5f} H {dh.mean():+.5f}+-{se(dh):.5f} "
              f"H_we {dh[hwe].mean():+.5f}+-{se(dh[hwe]):.5f} H_wd {dh[~hwe].mean():+.5f}+-{se(dh[~hwe]):.5f}")
picks = [k for k in CAND if verdict[k]['qualifies_a_to_d']]
json.dump({'mode': MODE, 'candidates': out, 'verdict': verdict, 'members': mem, 'info': info, 'picks': picks,
           'build': [{'tag': TAGS[k][0], 'out': TAGS[k][1], 'members': C[k]} for k in picks]}, open(OUT, 'w'), indent=1)
print('QUALIFYING (pending presub_gate):', picks or 'none -> nothing built'); print('written', OUT)
