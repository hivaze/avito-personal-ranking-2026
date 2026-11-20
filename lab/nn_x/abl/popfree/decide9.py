"""decide9 - pop-free family STAGE 2 (the decision), rule PRE-DECLARED in CURRENT_STATUS 03:50 UTC 2026-11-13 before any
pop-free run finished (decide7.py adapted), AMENDED 04:33 UTC (before any pop-free member existed): two FT variants per
member and three candidates. Members PF = abl_e10_xdrop_pop_s951-954 that are PRESENT (all five files in preds/: n2cpv3
val_oof + test, trainonly val + test, holdout/; at least 2); PFnp = the ALIAS members <v>np (same train-only / holdout
preds, the n2cpv3np = n2c_pre_v3_nopop fine-tune copied under the uniform n2cpv3 suffix by chain_popfree.sh).
Candidates vs G3 (F10 + F7x + F7x70, 20): Q1a = G3 + PF (standard n2cpv3 FT), Q1b = G3 + PFnp (nopop FT). A candidate
qualifies iff vs G3 (paired, member-equal 0.5 z(ft) + 0.5 z(to) on val, z-mean train-only on holdout): H > +1 SE (holdout);
V >= 0 (val); no val slice (M<50, M>=50, Tue, Thu) < -2 SE; pmix private-mix >= -1 SE; presub_gate PASS (run afterwards by
chain_popfree.sh on the pick only). Pick = the qualifier with the highest H. The pick REPLACES pick 1 only under the frozen
11-13 selection rule (H >= +2 SE and V >= 0; printed per qualifier, not decided here); with H between +1 and +2 SE it is a
recorded diversity candidate. Also printed: the within-feed z-corr of the Q1a / Q1b final val scores (and holdout pools)
to G3, and of the pop-free family pools (PF, PFnp) to the G3 pool.
Usage: decide9.py [arm=xdrop_pop] -> prints, writes nn_x/abl/popfree/decide9_pick.json ({pick, members, tag, out, ...})."""
import sys, json, os; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
ARM = sys.argv[1] if len(sys.argv) > 1 else 'xdrop_pop'
P = Path('<repo>/preds'); H = P / 'holdout'; OUT = Path('<repo>/nn_x/abl/popfree/decide9_pick.json')
F10 = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]; F7x = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
F7x70 = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]; G3 = F10 + F7x + F7x70
files = lambda m: [P/f'nn_{m}_n2cpv3_val_oof.npy', P/f'nn_{m}_trainonly_val.npy', H/f'nn_{m}_holdout.npy', P/f'nn_{m}_n2cpv3_test.npy', P/f'nn_{m}_trainonly_test.npy']
complete = lambda m: all(p.exists() for p in files(m))
PF_ALL = [f'abl_e10_{ARM}_s{s}' for s in range(951, 955)]
if os.environ.get('PF_OVERRIDE'):  # plumbing test only (never in the chain): any complete members stand in for the pop-free family
    PF_ALL = os.environ['PF_OVERRIDE'].split(','); print('PF_OVERRIDE (plumbing test, NOT the pop-free family):', PF_ALL)
NP = os.environ.get('NP_SUFFIX', 'np')  # alias suffix of the nopop-FT members (the chain copies <v>_n2cpv3np_* -> <v>np_n2cpv3_*)
PF = [m for m in PF_ALL if complete(m)]; PFnp = [m + NP for m in PF_ALL if complete(m + NP)]
print(f'pop-free members present: standard FT {len(PF)}/4 {PF}; nopop FT (alias {NP}) {len(PFnp)}/4 {PFnp}')
miss = [m for m in G3 if not complete(m)]; assert not miss, f'G3 missing {miss}'
TAGS = {'Q1a=G3+popfree': ('Q1apopfree', 'cand_Q1a_G3popfree_memeq.parquet'), 'Q1b=G3+popfree_np': ('Q1bpopfree', 'cand_Q1b_G3popfree_memeq.parquet')}
C = {'G3': G3}
if len(PF) >= 2: C['Q1a=G3+popfree'] = G3 + PF
if len(PFnp) >= 2: C['Q1b=G3+popfree_np'] = G3 + PFnp
if len(C) == 1:
    print('fewer than 2 pop-free members with complete preds in both FT variants -> no candidate; G3 stays')
    json.dump({'pick': None, 'members': [], 'tag': None, 'out': None, 'popfree_members': PF, 'popfree_members_np': PFnp, 'reason': 'fewer than 2 pop-free members present'}, open(OUT, 'w')); sys.exit(0)
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
for k, (v, h, sv, sh) in R.items():
    dv, dh = v - v0, h - h0
    zs = {n: float(dv[m].mean() / max(se(dv[m]), 1e-9)) for n, m in SL}
    pm, pse = pmix.wmean_se(dv, W['private'])
    q = k != 'G3' and dh.mean() > se(dh) and dv.mean() >= 0 and min(zs.values()) >= -2 and pm >= -pse
    rep = k != 'G3' and dh.mean() >= 2 * se(dh) and dv.mean() >= 0
    cz = (corr(zv(sv), zv(sv0)), corr(zh(sh), zh(sh0))) if k != 'G3' else (1.0, 1.0)
    out[k] = dict(n=len(C[k]), val=float(v.mean()), hold=float(h.mean()), V=float(dv.mean()), V_se=float(se(dv)), H=float(dh.mean()), H_se=float(se(dh)),
                  pmix=float(pm), pmix_se=float(pse), slices_z=zs, qualifies=bool(q), replace_pick1=bool(rep), zcorr_to_G3_val=cz[0], zcorr_to_G3_hold=cz[1])
    print(f'{k:18s} n={len(C[k]):2d} val {v.mean():.5f} hold {h.mean():.5f} | V {dv.mean():+.5f}+-{se(dv):.5f} H {dh.mean():+.5f}+-{se(dh):.5f} (H z {dh.mean()/max(se(dh),1e-9):+.2f}) '
          f'pmix {pm:+.5f}+-{pse:.5f} | ' + ' '.join(f'{n} {z:+.1f}' for n, z in zs.items()) + (f' | z-corr to G3 val {cz[0]:.4f} hold {cz[1]:.4f}' if k != 'G3' else '') + (' | QUALIFIES' if q else ''))
    if k != 'G3':
        print(f'  rule: H > +1 SE {dh.mean() > se(dh)}; V >= 0 {dv.mean() >= 0}; min slice z >= -2 {min(zs.values()) >= -2} ({min(zs, key=zs.get)} {min(zs.values()):+.2f}); '
              f'pmix >= -1 SE {pm >= -pse}; presub_gate: pending (chain, on the pick only)')
        if q:
            print(f'  PICK-1 REPLACEMENT threshold (frozen 11-13: H >= +2 SE and V >= 0): ' + ('MET' if rep else 'NOT met -> a recorded DIVERSITY CANDIDATE (Dec-7 freeze), nothing submitted'))
# family pools (the pop-free members alone) vs the G3 pool: the diversity numbers
g3v = zv(np.mean([ld(m)[1] for m in G3], 0)); g3f = zv(np.mean([ld(m)[0] for m in G3], 0)); g3h = zh(np.mean([ld(m)[2] for m in G3], 0))
for name, ms in [('popfree (to)', PF), ('popfree (ft n2cpv3)', PF), ('popfree_np (ft n2cpv3np)', PFnp)]:
    if len(ms) < 2: continue
    i = 1 if name.endswith('(to)') else 0; ref = g3v if i == 1 else g3f
    print(f'  family pool z-corr to the G3 {"train-only" if i == 1 else "ft"} pool: {name:26s} val {corr(zv(np.mean([ld(m)[i] for m in ms], 0)), ref):.4f}'
          + (f' | holdout {corr(zh(np.mean([ld(m)[2] for m in ms], 0)), g3h):.4f}' if i == 1 else ''))
def choose_pick(out, qs, eps=1e-12):
    """Pick among the qualifiers: highest H; H tied (Q1a / Q1b share the holdout pool, so their H is identical) -> higher pmix
    delta; still tied -> Q1b (pre-declared amendment 2, CURRENT_STATUS 04:44 UTC). Returns (pick or None, reason)."""
    if not qs:
        return None, 'no qualifier'
    Hmax = max(out[k]['H'] for k in qs); top = [k for k in qs if abs(out[k]['H'] - Hmax) <= eps]
    if len(top) == 1:
        return top[0], 'highest H'
    Pmax = max(out[k]['pmix'] for k in top); top2 = [k for k in top if abs(out[k]['pmix'] - Pmax) <= eps]
    if len(top2) == 1:
        return top2[0], f'H tied among {top} -> higher pmix delta'
    q1b = [k for k in top2 if k.startswith('Q1b')]
    return (q1b[0] if q1b else top2[0]), f'H and pmix tied among {top2} -> Q1b by rule'
qs = [k for k in out if out[k]['qualifies']]
pick, why = choose_pick(out, qs)
print('PICK (pending presub_gate PASS):', pick or 'none -> G3 stays', f'[{why}] -> TAG {TAGS[pick][0]} subs/{TAGS[pick][1]}' if pick else '')
json.dump({'pick': pick, 'members': C.get(pick, []), 'tag': TAGS[pick][0] if pick else None, 'out': TAGS[pick][1] if pick else None, 'tie_break': why,
           'popfree_members': PF, 'popfree_members_np': PFnp, 'candidates': out, 'replace_pick1': bool(pick and out[pick]['replace_pick1'])}, open(OUT, 'w'), indent=1)
print('written', OUT)
