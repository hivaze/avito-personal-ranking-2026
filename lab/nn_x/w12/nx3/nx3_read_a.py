"""READ TOOL of nx3 STAGE A (rule: ledger nn_x/w5/logs/E_kd_predeclared.txt, block "TRAINED-INPUT TEST nx3" of 2026-11-16T03:45:41Z, the designer's
clauses AS AMENDED by A1-A6; A7 changes the order of training only). CPU only, read-only on every prediction.
WRITE-NEW copy of reports/INPUTS_1116/design/read_nx.py: members(), load(), contrast(), stat() and line() are that tool's code; added are the
harness phase (A3 ORDER), the holdout feeds WITHOUT the view (A3 REPORT), the stale-val clause (designer), the VALUE clause and its val report
(A3 VALUE / REPORT) and the amended wording of the outcome (A3 WORDING). Called by nn_x/w12/nx3/read_stage_a.sh; nothing is selected by this tool.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
     POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=nn_x/w12/nx3/out/numba_cache .venv/bin/python nn_x/w12/nx3/nx3_read_a.py ...
     harness <FRESH> <STORED> <out file>                          e.g.  abl_e10_base:957-958   w5_e10_base:951-956
     read    <ARM> <CTL> <rescore dir> <harness file> <out file>  e.g.  abl_e10_nx3:951-958    w5_e10_base:951-956+abl_e10_base:957-958
  member lists "prefix:s1-s2[+prefix:s1-s2...]", resolved by nn_x/w12/recheck/pool_composition/pc_common.paths (train-only val, holdout; selected, @last).
  The harness phase opens the predictions of the members it is given and nothing else (no arm file is touched).
  Environment (dry tests only): NX3_SKIP_VALUE=1 / NX3_SKIP_SVL=1 leave that clause uncomputed (then NO verdict is printed); NX3_REREAD=1 = the one
  re-read of 12 + 12 after a REPLICATE (the band then reads as "not PASS").

Statistics (all paired per feed over the same feeds; identical to read_nx.py):
  member level = mean over the arm's members of per-feed NDCG minus the same for the control; SE = max(per-feed paired SE,
                 seed-level SE sqrt(sd_arm^2 / k_arm + sd_ctl^2 / k_ctl) of the member means); z = delta / SE
  pool level   = NDCG of the member-equal mean of per-feed z-scores (the project's pool) of the arm minus that of the control
  val          = 14,423 feeds; also with the private / public feed-mix weights of src/pmix.py, and M < 50 / M >= 50
  holdout      = the 30,000 in-period feeds; DECISION read on the lag-view subset = the 15,225 holdout feeds of Sep 12-19 (slot >= 6, has = 1);
                 REPORT on the 14,775 feeds of Sep 6-11 (has = 0)
  pooled z     = (z_val + z_holdout_lagview) / sqrt(2), member level, selected checkpoints
  stale-val    = val re-scored under scenario SVL (both_pre2lt3), arm members with nx_rp3_z / nx_rp3_rk replaced by their stale-val twins
                 (nx3_rescore.py svl); member delta arm - control, same SE form
  VALUE        = the arm's members re-scored with the two value columns permuted within the feed (nx3_rescore.py perm): member-mean per-feed
                 NDCG(natural) - NDCG(permuted), both from the same code path; SE = paired per-feed SE of that difference
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '12')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition'); sys.path.insert(0, '<repo>/src')
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import time
from pathlib import Path
import numpy as np
import pc_common as pc
import pmix
import hz2_common as hc

ROOT = Path('<repo>')
POWER = '0.71-0.80 for a member-level +0.001, 0.24-0.29 for +0.0005, 0.09-0.13 for +0.0003; critic/out/c04_rule_null.txt'


def members(spec):
    out = []
    for part in spec.split('+'):
        pre, rng_ = part.split(':'); a, b = (rng_.split('-') + [rng_])[:2]
        out += [f'{pre}_s{s}' for s in range(int(a), int(b) + 1)]
    return out


class Reader:
    def __init__(self, out_path):
        self.fo = open(out_path, 'w') if out_path else None
        self.m = m = pc.Meta(); W = pmix.weights()
        self.wpriv = np.asarray(W['private'], np.float64); self.wpub = np.asarray(W['public'], np.float64)
        ftr = hc.feeds('train'); hold, _ = hc.holdout_feeds(ftr)
        slot = (hc.day_of(ftr['timestamp'].to_numpy()[hold]) - hc.DAY0).astype(np.int64)
        assert len(slot) == len(m.Mh)
        self.sub = dict(all=np.ones(len(slot), bool), lagview=slot >= 6, vallike=(slot >= 6) & (slot <= 9), testlike=slot >= 11, noview=slot < 6)
        # label-free check of the subset against the installed input: has = non-NaN of nx_rp3_z on the first row of each holdout feed
        z = np.load(ROOT / 'nn/feat_cache/train/nx_rp3_z.npy', mmap_mode='r')
        has = ~np.isnan(np.asarray(z[ftr['cand_off'].to_numpy().astype(np.int64)[hold]]))
        assert (has == self.sub['lagview']).all(), 'lag-view holdout feeds (slot >= 6) differ from has:nx_rp3_z = 1 on the installed file'
        assert int(self.sub['lagview'].sum()) == 15225 and int(self.sub['noview'].sum()) == 14775, 'holdout subset counts differ from the rule (15,225 / 14,775)'

    def P(self, *a):
        s = ' '.join(str(x) for x in a); print(s, flush=True)
        if self.fo: self.fo.write(s + '\n'); self.fo.flush()

    def nd(self, sp, z):
        m = self.m
        return pc.ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp])

    def load(self, v, suf):
        pv, ph, _ = pc.paths(v, suf)
        zv = self.m.z('val', np.load(pv)); zh = self.m.z('hold', np.load(ph))
        return zv, zh, self.nd('val', zv), self.nd('hold', zh)

    def contrast(self, arm, ctl, suf):
        A = [self.load(v, suf) for v in arm]; C = [self.load(v, suf) for v in ctl]
        res = {}
        for tag, i_z, i_n, sp in (('val', 0, 2, 'val'), ('hold', 1, 3, 'hold')):
            na = np.stack([a[i_n] for a in A]); nc = np.stack([c[i_n] for c in C])
            pa = self.nd(sp, np.mean([a[i_z] for a in A], 0).astype(np.float32))
            pc_ = self.nd(sp, np.mean([c[i_z] for c in C], 0).astype(np.float32))
            res[tag] = (na, nc, pa - pc_)
        return res

    @staticmethod
    def stat(na, nc, dpool, mask=None, w=None):
        if mask is None: mask = np.ones(na.shape[1], bool)
        d = na[:, mask].mean(0) - nc[:, mask].mean(0); dp = dpool[mask]
        if w is None:
            mu, se_pf = d.mean(), d.std() / np.sqrt(len(d)); mup, sep = dp.mean(), dp.std() / np.sqrt(len(dp))
            ma, mc = na[:, mask].mean(1), nc[:, mask].mean(1)
        else:
            ww = w[mask]; mu, se_pf = pmix.wmean_se(d, ww); mup, sep = pmix.wmean_se(dp, ww)
            ma = np.array([np.average(x[mask], weights=ww) for x in na]); mc = np.array([np.average(x[mask], weights=ww) for x in nc])
        se_seed = np.sqrt(ma.var(ddof=1) / len(ma) + mc.var(ddof=1) / len(mc)) if len(ma) > 1 and len(mc) > 1 else 0.0
        se = max(se_pf, se_seed)
        return dict(d=mu, se=se, se_pf=se_pf, se_seed=se_seed, z=mu / se, dp=mup, sep=sep, zp=mup / sep, n=int(mask.sum()),
                    pos=int((ma > mc.mean()).sum()), k=len(ma))

    def line(self, lab, r):
        self.P(f'  {lab:38s} n {r["n"]:6d} | member {r["d"]:+.5f} +- {r["se"]:.5f} (z {r["z"]:+.2f}; per-feed SE {r["se_pf"]:.5f}, seed SE {r["se_seed"]:.5f}; '
               f'{r["pos"]}/{r["k"]} arm members above the control mean) | pool {r["dp"]:+.5f} +- {r["sep"]:.5f} (z {r["zp"]:+.2f})')


HOLD_LINES = (('all', 'holdout, all 30,000 feeds'), ('lagview', 'holdout, lag-view feeds (Sep 12-19)'), ('vallike', 'holdout, Sep 12-14 (val-like lag)'),
              ('testlike', 'holdout, Sep 17-19 (test-like lag)'), ('noview', 'holdout, feeds WITHOUT the view (Sep 6-11)'))


def harness(fresh_spec, stored_spec, out_path):
    fresh, stored = members(fresh_spec), members(stored_spec)
    R = Reader(out_path); P = R.P
    P(f'=== nx3 STAGE A HARNESS CHECK (A3 ORDER: read and written to the ledger BEFORE any prediction of the arm is opened); computed '
      f'{time.strftime("%FT%TZ", time.gmtime())} by nn_x/w12/nx3/nx3_read_a.py harness ===')
    P(f' FRESH controls ({len(fresh)}): {", ".join(fresh)}    minus    STORED controls ({len(stored)}): {stored[0]} .. {stored[-1]}')
    P(' Rule: member level, |z| <= 2 on val AND on the holdout (30,000 feeds), selected checkpoints; otherwise 6 more fresh controls (951-956) replace the stored ones.')
    P(' Only the predictions of these control members were opened.')
    P(f' The pool columns compare pools of unequal size ({len(fresh)} against {len(stored)} members) and are no part of the check.')
    out = {}
    for suf, name in (('', 'selected checkpoints (decision)'), ('@last', 'last step (information)')):
        C = R.contrast(fresh, stored, suf); P(f'== {name}')
        na, nc, dp = C['val']; out[suf, 'val'] = R.stat(na, nc, dp); R.line('val', out[suf, 'val'])
        R.line('val, private mix', R.stat(na, nc, dp, w=R.wpriv))
        na, nc, dp = C['hold']
        for k_, lab in HOLD_LINES:
            out[suf, k_] = R.stat(na, nc, dp, R.sub[k_]); R.line(lab, out[suf, k_])
    zv, zh = out['', 'val']['z'], out['', 'all']['z']
    ok = abs(zv) <= 2 and abs(zh) <= 2
    P(f'== HARNESS DECISION (selected checkpoints, member level): z_val {zv:+.2f}, z_holdout(30,000 feeds) {zh:+.2f}; bar |z| <= 2 on both')
    if ok:
        P('   -> HARNESS OK: the fresh controls sit with the stored ones; the control set of the read is the 6 stored w5_e10_base_s951-956 + the 2 fresh abl_e10_base_s957, s958.')
    else:
        P('   -> HARNESS FAIL: by the rule 6 more fresh controls abl_e10_base_s951-956 REPLACE the stored ones before the read '
          '(launch them with nn_x/w12/nx3/queue_nx3.sh "951:base 952:base 953:base 954:base 955:base 956:base"; then read with CTL=abl_e10_base:951-958).')
    P(f'HARNESS_VERDICT={"OK" if ok else "FAIL"}')
    return 0


def read(arm_spec, ctl_spec, resc, harness_file, out_path):
    arm, ctl = members(arm_spec), members(ctl_spec); resc = Path(resc)
    skip_value, skip_svl, reread = (os.environ.get(k, '0') == '1' for k in ('NX3_SKIP_VALUE', 'NX3_SKIP_SVL', 'NX3_REREAD'))
    R = Reader(out_path); P = R.P; m = R.m; sub = R.sub
    P(f'=== nx3 STAGE A READ (one look; computed {time.strftime("%FT%TZ", time.gmtime())} by nn_x/w12/nx3/nx3_read_a.py read) ===')
    P(f'ARM  ({len(arm)}): {arm[0]} .. {arm[-1]}'); P(f'CTL  ({len(ctl)}): {ctl[0]} .. {ctl[-1]}   [{ctl_spec}]')
    hv = [l.rstrip('\n') for l in open(harness_file)] if harness_file and Path(harness_file).exists() else []
    hdec = [l for l in hv if l.startswith('== HARNESS DECISION') or l.startswith('   -> HARNESS') or l.startswith('HARNESS_VERDICT=')]
    P(f'HARNESS CHECK on record ({harness_file}):'); [P('  ' + l) for l in (hdec or ['  (no harness file given)'])]
    out = {}; keep = {}
    for suf, name in (('', 'selected checkpoints'), ('@last', 'last step')):
        C = R.contrast(arm, ctl, suf); P(f'== {name}'); keep[suf] = C
        na, nc, dp = C['val']; big = m.Mv >= 50
        out[suf, 'val'] = R.stat(na, nc, dp); R.line('val', out[suf, 'val'])
        out[suf, 'priv'] = R.stat(na, nc, dp, w=R.wpriv); R.line('val, private mix', out[suf, 'priv'])
        R.line('val, public mix', R.stat(na, nc, dp, w=R.wpub)); R.line('val M>=50', R.stat(na, nc, dp, big)); R.line('val M<50', R.stat(na, nc, dp, ~big))
        na, nc, dp = C['hold']
        for k_, lab in HOLD_LINES:
            out[suf, k_] = R.stat(na, nc, dp, sub[k_]); R.line(lab, out[suf, k_])
    v, p, h, hl, nv = out['', 'val'], out['', 'priv'], out['', 'lagview'], out['@last', 'lagview'], out['', 'noview']
    pooled = (v['z'] + h['z']) / np.sqrt(2)

    # ---------------- stale-val clause (SVL; arm members with the NX twins)
    sv = {}
    if skip_svl:
        P('== stale-val: NOT COMPUTED (NX3_SKIP_SVL=1, dry test)')
    else:
        P(f'== stale-val, scenario SVL (both_pre2lt3); arm members read nx_rp3_z / nx_rp3_rk from their stale-val twins ({resc}/svl)')
        for suf, name in (('', 'selected (the clause)'), ('@last', 'last step (information)')):
            def ld(vn):
                f = resc / 'svl' / f'nn_{vn}{suf}_staleval.npy'
                assert f.exists(), f'missing {f}: run nx3_rescore.py svl first'
                z = m.z('val', np.load(f)); return z, R.nd('val', z)
            A = [ld(x) for x in arm]; C = [ld(x) for x in ctl]
            na = np.stack([a[1] for a in A]); nc = np.stack([c[1] for c in C])
            dpool = R.nd('val', np.mean([a[0] for a in A], 0).astype(np.float32)) - R.nd('val', np.mean([c[0] for c in C], 0).astype(np.float32))
            sv[suf] = R.stat(na, nc, dpool); R.line(f'stale-val, {name}', sv[suf])
            nat = keep[suf]['val']
            P(f'     levels, member mean: arm natural {nat[0].mean():.5f} -> stale {na.mean():.5f} ({na.mean() - nat[0].mean():+.5f}); '
              f'control natural {nat[1].mean():.5f} -> stale {nc.mean():.5f} ({nc.mean() - nat[1].mean():+.5f})')

    # ---------------- VALUE clause (natural - permuted inside the arm's members; selected checkpoints)
    val_ = {}
    if skip_value:
        P('== VALUE: NOT COMPUTED (NX3_SKIP_VALUE=1, dry test)')
    else:
        P(f'== VALUE (A3): the arm\'s {len(arm)} members re-scored with nx_rp3_z and nx_rp3_rk permuted within the feed (one permutation per feed for both, has kept at 1); '
          f'selected checkpoints ({resc}/perm)')
        def ldp(vn, sp, kind):
            f = resc / 'perm' / f'nn_{vn}_{"val" if sp == "val" else "hold"}_{kind}.npy'
            assert f.exists(), f'missing {f}: run nx3_rescore.py perm first'
            return R.nd(sp, m.z(sp, np.load(f)))
        D = {}
        for sp in ('val', 'hold'):
            nat = np.stack([ldp(x, sp, 'nat') for x in arm]); per = np.stack([ldp(x, sp, 'perm') for x in arm])
            stored = keep['']['val' if sp == 'val' else 'hold'][0]
            P(f'  check, {sp}: re-scored natural vs the stored predictions of the trainer, member NDCG: max |difference| {np.abs(nat.mean(1) - stored.mean(1)).max():.2e} '
              f'(per feed: max {np.abs(nat - stored).max():.2e})')
            D[sp] = nat - per

        def vstat(d, mask=None):
            if mask is not None: d = d[:, mask]
            pf = d.mean(0); mm = d.mean(1)
            return dict(d=pf.mean(), se=pf.std() / np.sqrt(len(pf)), se_seed=(mm.std(ddof=1) / np.sqrt(len(mm)) if len(mm) > 1 else 0.0), pos=int((mm > 0).sum()), k=len(mm), n=len(pf))

        def vline(lab, r):
            P(f'  {lab:38s} n {r["n"]:6d} | natural - permuted {r["d"]:+.5f} +- {r["se"]:.5f} (paired per-feed SE; z {r["d"] / r["se"] if r["se"] > 0 else float("nan"):+.2f}; '
              f'seed-level SE {r["se_seed"]:.5f}; {r["pos"]}/{r["k"]} members > 0)')
        val_['val'] = vstat(D['val']); vline('val', val_['val'])
        val_['lagview'] = vstat(D['hold'], sub['lagview']); vline('holdout, lag-view feeds (Sep 12-19)', val_['lagview'])
        for k_, lab in (('vallike', 'holdout, Sep 12-14 (val-like lag)'), ('testlike', 'holdout, Sep 17-19 (test-like lag)')):
            vline(lab, vstat(D['hold'], sub[k_]))
        mx = float(np.abs(D['hold'][:, sub['noview']]).max())
        P(f'  check: on the {int(sub["noview"].sum()):,} holdout feeds without the view the permutation is a no-op: max |natural - permuted| per feed = {mx:.2e}')

    # ---------------- FINAL BLOCK
    ok = lambda b: 'ok' if b else 'FAIL'
    P('')
    P('=== nx3 STAGE A: FINAL BLOCK (rule block 2026-11-16T03:45:41Z as amended A1-A6; selected checkpoints, member level unless stated) ===')
    P(f' ARM {arm[0]} .. {arm[-1]} ({len(arm)})   CTL {ctl_spec} ({len(ctl)})   harness: {(hdec[-1] if hdec else "no harness file")}')
    c1 = v['z'] >= 1; c2 = h['z'] >= 1; c3 = pooled >= 2; c4 = p['z'] >= -1; c5 = v['dp'] >= 0 and h['dp'] >= 0; c6 = hl['d'] >= 0
    P(f' C1 val                      z >= +1.0            : member {v["d"]:+.5f} +- {v["se"]:.5f}, z {v["z"]:+.2f}  -> {ok(c1)}')
    P(f' C2 lag-view holdout (15,225) z >= +1.0            : member {h["d"]:+.5f} +- {h["se"]:.5f}, z {h["z"]:+.2f}  -> {ok(c2)}')
    P(f' C3 pooled z = (z_val + z_holdout) / sqrt(2) >= 2.0 : {pooled:+.2f}  -> {ok(c3)}')
    P(f' C4 private mix (src/pmix.py) member z >= -1.0     : member {p["d"]:+.5f} +- {p["se"]:.5f}, z {p["z"]:+.2f}  -> {ok(c4)}')
    P(f' C5 pool minus pool >= 0 on val AND lag-view holdout: val {v["dp"]:+.5f} +- {v["sep"]:.5f}, holdout {h["dp"]:+.5f} +- {h["sep"]:.5f}  -> {ok(c5)}')
    P(f' C6 last step, lag-view holdout, member delta >= 0  : {hl["d"]:+.5f} +- {hl["se"]:.5f}  -> {ok(c6)}')
    if sv:
        s0 = sv['']; c7 = s0['d'] >= -s0['se']
        P(f' C7 stale-val (SVL + NX twins) member delta >= -1 SE: {s0["d"]:+.5f} +- {s0["se"]:.5f} (z {s0["z"]:+.2f})  -> {ok(c7)}   '
          f'[last step, information: {sv["@last"]["d"]:+.5f} +- {sv["@last"]["se"]:.5f}]')
    else:
        c7 = None; P(' C7 stale-val: NOT COMPUTED (dry test)')
    if val_:
        vv, vh = val_['val'], val_['lagview']; c8 = (vh['d'] >= 0.5 * h['d']) and (vv['d'] > 0)
        P(f' C8 VALUE: natural - permuted >= 0.5 x (arm - control) on the lag-view holdout AND > 0 on val:')
        P(f'      lag-view holdout {vh["d"]:+.5f} +- {vh["se"]:.5f} against 0.5 x {h["d"]:+.5f} = {0.5 * h["d"]:+.5f} -> {ok(vh["d"] >= 0.5 * h["d"])};  '
          f'val {vv["d"]:+.5f} +- {vv["se"]:.5f} -> {ok(vv["d"] > 0)}   -> {ok(c8)}')
    else:
        c8 = None; P(' C8 VALUE: NOT COMPUTED (dry test)')
    band = 1.6 <= pooled < 2 and v['z'] >= 0 and h['z'] >= 0
    P(f' REPLICATE band: pooled z in [1.6, 2.0) with z_val >= 0 and z_holdout >= 0 : {"yes" if band else "no"}{" (this IS the one re-read: no second replicate)" if reread else ""}')
    P(f' REPORT R1 (A3) arm - control on the 14,775 holdout feeds WITHOUT the view (expected 0): member {nv["d"]:+.5f} +- {nv["se"]:.5f}, z {nv["z"]:+.2f}'
      + ('  -> |z| > 2: SIDE EFFECT OF THE ARM, written next to the verdict' if abs(nv['z']) > 2 else '  -> |z| <= 2: no side effect shown'))
    if val_:
        P(f' REPORT R2 (A3) natural - permuted on val = the size the trained members take from the values: {vv["d"]:+.5f} +- {vv["se"]:.5f} (paired SE)')
    if c7 is None or c8 is None:
        P(' VERDICT: NONE (dry test: a clause was not computed)')
    else:
        designed = c1 and c2 and c3 and c4 and c5 and c6 and c7
        closed = (vv['d'] + 2 * vv['se']) < 0.0005
        ctxt = (f'natural - permuted on val + 2 SE = {vv["d"] + 2 * vv["se"]:+.5f} '
                + ('< +0.0005 -> "CLOSED at +0.0005 for this recipe"' if closed else '>= +0.0005 -> not closed at +0.0005'))
        if designed and c8:
            verdict = ('PASS. Stage B is read next (A4; A7.5 fixes the order). NOTE (A3): the effect quoted here is biased upward by the selection through the rule '
                       '(a PASS needs about +0.0007 on val by construction); the estimate of record is stage B\'s.')
        elif designed and not c8:
            verdict = ('PASS NOT ATTRIBUTABLE TO THE VALUES (every designed clause holds, the VALUE clause fails: the has flag is 1 exactly on Sep 12-19, the regime '
                       f'indicator of map l.273): no stage B as a test; A7.3 governs students already trained. [{ctxt}]')
        elif band and not reread:
            verdict = 'REPLICATE: 4 more seeds per arm (959-962), ONE re-read of 12 + 12 by the same clauses.'
        elif closed:
            verdict = (f'CLOSED at +0.0005 for this recipe (PASS is not reached AND {ctxt}; without that second condition the wording would be '
                       f'"NOT SHOWN at this power ({POWER})"). "New train-stage inputs" is NOT closed as a class: the arm holds ONE piece of information '
                       '(one scalar in two forms). A7.3: stage B, if trained, is read once as a descriptive record.')
        else:
            verdict = (f'NOT SHOWN at this power ({POWER}). {ctxt}. "New train-stage inputs" is not closed as a class: the arm holds ONE piece of '
                       'information (one scalar in two forms). A7.3: stage B, if trained, is read once as a descriptive record.')
        P(f' VERDICT: {verdict}')
        if abs(nv['z']) > 2:
            P(f' NEXT TO THE VERDICT (A3 REPORT): side effect of the arm on the feeds without the view, z {nv["z"]:+.2f}.')
    return 0


if __name__ == '__main__':
    if sys.argv[1] == 'harness':
        sys.exit(harness(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))
    elif sys.argv[1] == 'read':
        sys.exit(read(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] if len(sys.argv) > 6 else None))
    raise SystemExit('usage: nx3_read_a.py harness <FRESH> <STORED> [out] | read <ARM> <CTL> <rescore dir> <harness file> [out]')
