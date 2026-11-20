"""Pool read-out for the POPULARITY-VIEW FAMILIES follow-up (Task A pre-declared nn_x/abl/pop1/x8/PREDECLARED_pop1x8.txt 14:21Z;
Task B pre-declared nn_x/abl/ledger/ledger.md section 8). The pool computation is EXACTLY nn_x/abl/pop1/chain/decide9.py's
(ld / pool / slices / pmix / corr copied verbatim); only the candidate sets and the rule clauses differ:

  decide_x8.py Q2a8   -> G3 (20), Q2a = G3 + pop1(951-954), Q2a' = G3 + pop1(955-958), Q2a8 = G3 + pop1(951-958)
                         rule (a) V(Q2a8 - G3) > +1 SE; (b) V(Q2a8 - Q2a) >= -1 SE; (c) pmix(Q2a8 - G3) >= pmix(Q2a - G3) - 1 SE
                         [SE = pmix SE of the paired delta Q2a8 - Q2a]; (d) no val slice of Q2a8 - G3 < -2 SE; (e) H(Q2a8 - G3) >= -1 SE
                         (veto; the in-period holdout under-values popularity-light members); (f) presub_gate PASS (run by the chain).
  decide_x8.py Q3     -> G3, Q3 = G3 + pop2(951-954), Q3b = G3 + pop1(951-954) + pop2(951-954); same rule with G3 in (b)
                         (V(X - G3) >= -1 SE) and (c) pmix(X - G3) >= -1 SE.
Replication information (Q2a8 mode): V / H / pmix of Q2a' - G3 and Q2a' - Q2a, and the within-feed z-corr of the 4-member
train-only pools pop1(955-958) vs pop1(951-954) (val and holdout) and each vs the G3 train-only pool.
Output: prints + nn_x/abl/pop1/x8/decide_x8_<mode>.json. CPU only, read-only on every prediction file. Members with incomplete
files are dropped (disclosed in the output); a family needs >= 2 members.
"""
import sys, json, os; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
MODE = sys.argv[1] if len(sys.argv) > 1 else 'Q2a8'
assert MODE in ('Q2a8', 'Q3'), 'usage: decide_x8.py Q2a8 | Q3'
P = Path('<repo>/preds'); H = P / 'holdout'; OUT = Path(f'<repo>/nn_x/abl/pop1/x8/decide_x8_{MODE}.json')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; G3 = F10 + F7x + F7x70
files = lambda m: [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_trainonly_val.npy', H/f'nn_{m}_holdout.npy', P/f'nn_{m}_n2cpv3_test.npy', P/f'nn_{m}_trainonly_test.npy']
complete = lambda m: all(p.exists() for p in files(m))
PLUMB = os.environ.get('X8_PLUMB_PB', '')  # plumbing test only (never set by the chains): stand-in members for the SECOND family
if PLUMB: OUT = OUT.with_name(OUT.stem + '_plumbing.json'); print('X8_PLUMB_PB plumbing test (stand-in second family, NOT a read-out):', PLUMB)
def fam(arm, seeds):
    allm = [f'abl_e10_{arm}_s{s}' for s in seeds]; ok = [m for m in allm if complete(m)]
    if PLUMB and (seeds[0] == 955 or arm == 'pop2'): allm = PLUMB.split(','); ok = [m for m in allm if complete(m)]
    print(f'{arm} seeds {seeds}: complete {len(ok)}/{len(allm)} {ok}' + (f' MISSING {[m for m in allm if m not in ok]}' if len(ok) < len(allm) else ''))
    return ok
miss = [m for m in G3 if not complete(m)]; assert not miss, f'G3 missing {miss}'
C = {'G3': G3}; TAGS = {}
if MODE == 'Q2a8':
    PA, PB = fam('pop1', range(951, 955)), fam('pop1', range(955, 959))
    assert len(PA) == 4, 'the 4 existing pop1 members must be complete'
    assert len(PB) >= 2, 'fewer than 2 of pop1 s955-958 complete: no read-out'
    C['Q2a=G3+pop1(951-954)'] = G3 + PA; C["Q2a'=G3+pop1(955-958)"] = G3 + PB; C['Q2a8=G3+pop1(951-958)'] = G3 + PA + PB
    CAND = ['Q2a8=G3+pop1(951-958)']; REFV = 'Q2a=G3+pop1(951-954)'
    TAGS = {'Q2a8=G3+pop1(951-958)': ('Q2a8pop1', 'cand_Q2a8_G3pop1x8_memeq.parquet')}
    FAMS = {'pop1(951-954) to': PA, 'pop1(955-958) to': PB, 'pop1(951-958) to': PA + PB}
else:
    PA, PB = fam('pop1', range(951, 955)), fam('pop2', range(951, 955))
    assert len(PA) == 4 and len(PB) >= 2
    C['Q3=G3+pop2(4)'] = G3 + PB; C['Q3b=G3+pop1(4)+pop2(4)'] = G3 + PA + PB
    CAND = ['Q3=G3+pop2(4)', 'Q3b=G3+pop1(4)+pop2(4)']; REFV = 'G3'
    TAGS = {'Q3=G3+pop2(4)': ('Q3pop2', 'cand_Q3_G3pop2_memeq.parquet'), 'Q3b=G3+pop1(4)+pop2(4)': ('Q3bpop12', 'cand_Q3b_G3pop1pop2_memeq.parquet')}
    FAMS = {'pop1(951-954) to': PA, 'pop2(951-954) to': PB, 'pop1+pop2 to': PA + PB}
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
W = pmix.weights(); out = {}
# ---- per-member information ----
print('MEMBERS (train-only val / holdout NDCG, FT n2cpv3 OOF NDCG):')
mem = {}
for name, ms in FAMS.items():
    if name.startswith('pop1(951-958)') or name.startswith('pop1+pop2'): continue
    for m in ms:
        a = ev(np.load(P/f'nn_{m}_trainonly_val.npy')).mean(); b = eh(np.load(H/f'nn_{m}_holdout.npy')).mean(); c = ev(np.load(P/f'nn_{m}_n2cpv3_val_oof.npy')).mean()
        mem[m] = dict(to_val=float(a), hold=float(b), ft_oof=float(c)); print(f'  {m:22s} to val {a:.5f} hold {b:.5f} | FT OOF {c:.5f}')
# ---- pools vs G3 ----
for k, (v, h, sv, sh) in R.items():
    dv, dh = v - v0, h - h0
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    dsl = {n: (float(dv[m].mean()), float(se(dv[m]))) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, W['private'])
    cz = (corr(zv(sv), zv(sv0)), corr(zh(sh), zh(sh0))) if k != 'G3' else (1.0, 1.0)
    out[k] = dict(n=len(C[k]), members=C[k], val=float(v.mean()), hold=float(h.mean()), V=float(dv.mean()), V_se=float(se(dv)), H=float(dh.mean()), H_se=float(se(dh)),
                  pmix=float(pm), pmix_se=float(pse), slices_z=zs, slices=dsl, zcorr_to_G3_val=cz[0], zcorr_to_G3_hold=cz[1])
    print(f'{k:26s} n={len(C[k]):2d} val {v.mean():.5f} hold {h.mean():.5f} | V {dv.mean():+.5f}+-{se(dv):.5f} (z {dv.mean()/max(se(dv),1e-9):+.2f}) H {dh.mean():+.5f}+-{se(dh):.5f} (z {dh.mean()/max(se(dh),1e-9):+.2f}) '
          f'pmix {pm:+.5f}+-{pse:.5f} (z {pm/max(pse,1e-9):+.2f}) | ' + ' '.join(f'{n} {d:+.5f}({z:+.1f})' for (n, (d, s_)), z in zip(dsl.items(), zs.values()))
          + (f' | z-corr to G3 val {cz[0]:.4f} hold {cz[1]:.4f}' if k != 'G3' else ''))
# ---- pairwise reads needed by the rule: candidate vs REFV (val, pmix) ----
def pair(a, b):
    dv = R[a][0] - R[b][0]; pm, pse = pmix.wmean_se(dv, W['private']); dh = R[a][1] - R[b][1]
    return dict(V=float(dv.mean()), V_se=float(se(dv)), pmix=float(pm), pmix_se=float(pse), H=float(dh.mean()), H_se=float(se(dh)))
verdict = {}
for k in CAND:
    o = out[k]; pr = pair(k, REFV) if REFV != 'G3' else dict(V=o['V'], V_se=o['V_se'], pmix=o['pmix'], pmix_se=o['pmix_se'], H=o['H'], H_se=o['H_se'])
    pref = out[REFV]['pmix'] if REFV != 'G3' else 0.0
    cl = {'a_V_gt_1SE': o['V'] > o['V_se'],
          'b_V_vs_ref_ge_m1SE': pr['V'] >= -pr['V_se'],
          'c_pmix_vs_ref_ge_m1SE': (o['pmix'] - pref) >= -pr['pmix_se'],
          'd_slices_ge_m2SE': min(o['slices_z'].values()) >= -2,
          'e_H_ge_m1SE': o['H'] >= -o['H_se']}
    q = all(cl.values())
    verdict[k] = dict(clauses=cl, vs_ref=pr, ref=REFV, qualifies_a_to_e=bool(q), tag=TAGS[k][0], out=TAGS[k][1])
    print(f'RULE {k} (ref for (b)/(c) = {REFV}):')
    print(f"  (a) V - V(G3) = {o['V']:+.5f} > +1 SE {o['V_se']:.5f}: {cl['a_V_gt_1SE']}")
    print(f"  (b) V - V({REFV}) = {pr['V']:+.5f} >= -1 SE {pr['V_se']:.5f}: {cl['b_V_vs_ref_ge_m1SE']}")
    print(f"  (c) pmix(cand - G3) {o['pmix']:+.5f} >= pmix({REFV} - G3) {pref:+.5f} - 1 SE {pr['pmix_se']:.5f} (diff {o['pmix'] - pref:+.5f}): {cl['c_pmix_vs_ref_ge_m1SE']}")
    print(f"  (d) min val slice z {min(o['slices_z'], key=o['slices_z'].get)} {min(o['slices_z'].values()):+.2f} >= -2: {cl['d_slices_ge_m2SE']}")
    print(f"  (e) H - H(G3) = {o['H']:+.5f} >= -1 SE {o['H_se']:.5f} (z {o['H']/max(o['H_se'],1e-9):+.2f}): {cl['e_H_ge_m1SE']}")
    print(f"  (f) presub_gate: pending (run by the chain only if (a)-(e) hold)")
    print(f"  => {k}: {'QUALIFIES on (a)-(e) -> build TAG ' + TAGS[k][0] + ' subs/' + TAGS[k][1] + ' + presub_gate' if q else 'does NOT qualify -> nothing built'}")
# ---- replication / family information ----
info = {}
if MODE == 'Q2a8':
    kp, ka = "Q2a'=G3+pop1(955-958)", 'Q2a=G3+pop1(951-954)'
    pr = pair(kp, ka); info['Qp_vs_Qa'] = pr
    print(f"REPLICATION: Q2a' - G3: V {out[kp]['V']:+.5f}+-{out[kp]['V_se']:.5f} H {out[kp]['H']:+.5f}+-{out[kp]['H_se']:.5f} pmix {out[kp]['pmix']:+.5f}+-{out[kp]['pmix_se']:.5f} | "
          f"Q2a' - Q2a: V {pr['V']:+.5f}+-{pr['V_se']:.5f} H {pr['H']:+.5f}+-{pr['H_se']:.5f} pmix {pr['pmix']:+.5f}+-{pr['pmix_se']:.5f}")
g3v = zv(np.mean([ld(m)[1] for m in G3], 0)); g3f = zv(np.mean([ld(m)[0] for m in G3], 0)); g3h = zh(np.mean([ld(m)[2] for m in G3], 0))
fp = {name: (zv(np.mean([ld(m)[1] for m in ms], 0)), zh(np.mean([ld(m)[2] for m in ms], 0)), zv(np.mean([ld(m)[0] for m in ms], 0))) for name, ms in FAMS.items()}
names = list(FAMS)
for name in names:
    tv, th, tf = fp[name]
    info[f'{name} vs G3'] = dict(val=corr(tv, g3v), hold=corr(th, g3h), ft_val=corr(tf, g3f))
    print(f'  family pool z-corr to the G3 pool: {name:20s} train-only val {corr(tv, g3v):.4f} holdout {corr(th, g3h):.4f} | ft val {corr(tf, g3f):.4f}')
a, b = names[0], names[1]
info[f'{a} vs {b}'] = dict(val=corr(fp[a][0], fp[b][0]), hold=corr(fp[a][1], fp[b][1]), ft_val=corr(fp[a][2], fp[b][2]))
print(f'  family pool z-corr {a} vs {b}: train-only val {corr(fp[a][0], fp[b][0]):.4f} holdout {corr(fp[a][1], fp[b][1]):.4f} | ft val {corr(fp[a][2], fp[b][2]):.4f}')
# same-recipe reference: F7x vs F7x70 train-only pools (near-copy members of G3) and F10 vs F7x
for x, y, nm in [(F7x, F7x70, 'F7x vs F7x70 (same recipe)'), (F10, F7x, 'F10 vs F7x')]:
    cv_ = corr(zv(np.mean([ld(m)[1] for m in x], 0)), zv(np.mean([ld(m)[1] for m in y], 0))); info[nm] = cv_
    print(f'  reference pool z-corr {nm}: train-only val {cv_:.4f}')
picks = [k for k in CAND if verdict[k]['qualifies_a_to_e']]
json.dump({'mode': MODE, 'candidates': out, 'verdict': verdict, 'members': mem, 'info': info, 'picks': picks,
           'build': [{'tag': TAGS[k][0], 'out': TAGS[k][1], 'members': C[k]} for k in picks]}, open(OUT, 'w'), indent=1)
print('QUALIFYING (pending presub_gate):', picks or 'none -> nothing built'); print('written', OUT)
