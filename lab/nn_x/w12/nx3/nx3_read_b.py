"""READ TOOL of nx3 STAGE B (rule: ledger nn_x/w5/logs/E_kd_predeclared.txt, block "TRAINED-INPUT TEST nx3" of 2026-11-16T03:45:41Z: the designer's STAGE B
paragraph "Same clauses, same tool", AS AMENDED by A4 (pool-level clause, terminal branches), A7.2 / A7.3 / A7.5 (speculative training: B is a TEST only
after a PASS of A, otherwise ONE descriptive read; order of reads harness -> A -> B) and the LEAD'S RULINGS R1-R3 of 05:32:27Z). CPU only, read-only on
every prediction. Called by nn_x/w12/nx3/read_stage_b.sh; nothing is selected by this tool.

"SAME TOOL": Reader (load, nd, stat, line, the lag-view holdout subsets and their label-free check against the installed input file), members() and
HOLD_LINES are IMPORTED from nn_x/w12/nx3/nx3_read_a.py, the read tool of stage A (its md5 is printed in the header). Statistics are therefore the
stage-A statistics: member level = mean over the arm's members of per-feed NDCG minus the same for the reference members, SE = max(paired per-feed SE,
seed-level SE sqrt(sd_arm^2 / k_arm + sd_ref^2 / k_ref)); pool = member-equal mean of per-feed z-scores; pooled z = (z_val + z_holdout_lagview) / sqrt(2).

  nx3_read_b.py read <ARM> <REF1> <REF2> <rescore dir> <stage-A read file> <out file>
      ARM  = abl_e10_nx3kd05w192:951-958                      the 8 new wide students with the bundle (teacher_mean11)
      REF1 = w5_e10_kd05w192:951-958   REF2 = w5_e10_kd05t25w192:951-958     the 16 stored wide students = the two stored 8-pools
  Environment: NX3_SKIP_SVL=1 / NX3_SKIP_VALUE=1 (dry tests: that part is not computed; without SVL NO verdict is printed);
               NX3_REREAD=1 (the one re-read after "4 more students": the band then reads as not PASS);
               NX3_DRY_A="PASS:+2.31" | "NOT:+0.40" (dry tests only: the stage-A outcome is imagined instead of read from the stage-A file).

CLAUSES OF STAGE B as this tool reads them (selected checkpoints, member level against the 16 stored students unless stated):
  B1 val z >= +1.0            B2 lag-view holdout (15,225 feeds) z >= +1.0            B3 pooled z_B = (z_val + z_holdout) / sqrt(2) >= 2.0
  B4 private mix (src/pmix.py) member z >= -1.0                                      B6 last-step lag-view holdout member delta >= 0
  B5 pool minus pool >= 0 on val AND on the lag-view holdout feeds     } both on the A4 contrast: the 8-pool of the new students minus EACH stored
  B8 (A4) pool level: pooled z (val, lag-view holdout) >= 1.0          } 8-pool (w192, t25w192), MEAN of the two; per feed d = N(arm pool) -
                                                                         0.5 (N(pool w192) + N(pool t25w192)), paired per-feed SE, z = mean / SE,
                                                                         pooled z = (z_val + z_holdout) / sqrt(2)
  B7 stale-val (scenario SVL; the new students read nx_rp3_z / nx_rp3_rk from the stale-val twins) member delta >= -1 SE
  TERMINAL BRANCHES (A4), on pooled z_B = B3's statistic:  >= 2 and every clause -> PASS;  [1.6, 2) -> 4 more students, ONE re-read;  < 1.6 -> STOP.
  READING NOTES (for the lead; nothing here was tuned on a number): (a) the designer's C5 "8-pool minus 8-pool" has no equal-size reference pool when
  the reference is 16 students, so B5 is read on A4's own construction (mean of the two 8-v-8 contrasts); the 8-v-16 pool delta is printed as
  information (it is biased against the arm by the pool size). (b) VALUE (A3) is a clause of stage A; A4 does not add it to B. It is computed for the
  B students and printed as a REPORT; it does not enter B's verdict. (c) the REPLICATE band is read by A4's letter (pooled z_B alone); whether the
  designer's side conditions z_val >= 0 and z_holdout >= 0 also hold is printed.
  A7.3: if stage A is not a PASS, this read is DESCRIPTIVE (no file becomes a pick candidate from it); a replication block is opened only if
  pooled z_B >= 3.0 AND A's pooled z >= 0.
"""
import hashlib, os, re, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_k, '12')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/nx3')
import time
from pathlib import Path
import numpy as np
import nx3_read_a as RA
from nx3_read_a import members, HOLD_LINES

ROOT = Path('<repo>')
LIMITS = (' LIMITS ON RECORD (rulings R1, R2; quoted with every read): J0 train feed mix (all has = 1 train rows: C4 PSI of nx_rp3_z 0.127 / 0.260 / 0.346 against val / public / private; the '
          'pool-feed reading of BUILD_SPEC binds); J1 P(walk = 0 | seq_cv_invd20 > 0) 0.122 / 0.228 on lag-view rows against 0.0004-0.0005 on val / test rows; J2 one-step coefficient '
          '0.030 on lag-view holdout rows against 0.055 (val); J3 the lag view is fresher than its reference rows in distribution; J4 the clamp at 6 saturates 1.4-1.8% of rows.')


def a_state(a_file):
    """-> (kind, pooled z of A or None, verdict text). kind: PASS | NOTATTR | REPLICATE | NOT (CLOSED / NOT SHOWN) | UNKNOWN."""
    dry = os.environ.get('NX3_DRY_A', '')
    if dry:
        k, z = dry.split(':'); return ('PASS' if k == 'PASS' else 'NOT'), float(z), f'(DRY TEST: imagined stage-A outcome {dry})'
    txt = [l.rstrip('\n') for l in open(a_file)]
    ver = [l for l in txt if l.startswith(' VERDICT: ')]; pz = [l for l in txt if l.startswith(' C3 pooled z')]
    if not ver: return 'UNKNOWN', None, 'no VERDICT line in the stage-A read file'
    v = ver[-1][len(' VERDICT: '):]
    mz = re.search(r':\s*([+-]\d+\.\d+)\s+->', pz[-1]) if pz else None
    z = float(mz.group(1)) if mz else None
    if v.startswith('PASS NOT ATTRIBUTABLE'): k = 'NOTATTR'
    elif v.startswith('PASS'): k = 'PASS'
    elif v.startswith('REPLICATE'): k = 'REPLICATE'
    elif v.startswith(('CLOSED', 'NOT SHOWN')): k = 'NOT'
    else: k = 'UNKNOWN'
    return k, z, v


def read(arm_spec, r1_spec, r2_spec, resc, a_file, out_path):
    arm, r1, r2 = members(arm_spec), members(r1_spec), members(r2_spec); ctl = r1 + r2; resc = Path(resc)
    skip_value, skip_svl, reread = (os.environ.get(k, '0') == '1' for k in ('NX3_SKIP_VALUE', 'NX3_SKIP_SVL', 'NX3_REREAD'))
    R = RA.Reader(out_path); P = R.P; m = R.m; sub = R.sub; stat = RA.Reader.stat
    md5a = hashlib.md5(open(RA.__file__, 'rb').read()).hexdigest()
    P(f'=== nx3 STAGE B READ (one look; computed {time.strftime("%FT%TZ", time.gmtime())} by nn_x/w12/nx3/nx3_read_b.py read; Reader / statistics imported from nx3_read_a.py md5 {md5a}) ===')
    P(f'ARM  ({len(arm)}): {arm[0]} .. {arm[-1]}   ' + ('(wide distilled students WITH the bundle, old teacher teacher_mean11)' if all('nx3kd05w192' in x and '_smoke_' not in x for x in arm) else '(NOT the stage-B arm: a stand-in of a dry test)'))
    P(f'REF  ({len(ctl)}): REF1 {r1[0]} .. {r1[-1]} ({len(r1)}; same teacher) + REF2 {r2[0]} .. {r2[-1]} ({len(r2)}; teacher_mean25) = the stored wide students')
    akind, az, averd = a_state(a_file)
    P(f'STAGE A on record ({a_file}): pooled z {"%+.2f" % az if az is not None else "n/a"}; VERDICT: {averd[:260]}')
    hf = Path(a_file).parent / 'stage_a_harness.txt'
    hv = [l.rstrip('\n') for l in open(hf) if l.startswith(('== HARNESS DECISION', '   -> HARNESS', 'HARNESS_VERDICT='))] if hf.exists() and not os.environ.get('NX3_DRY_A') else []
    P('HARNESS CHECK of stage A on record (the arm of B is trained in nn_x/abl, the references in nn_x/w5): ' + (' | '.join(x.strip() for x in hv)[:400] if hv else '(not read: dry test or no file)'))
    cache = {}

    def L(v, suf):
        if (v, suf) not in cache: cache[v, suf] = R.load(v, suf)
        return cache[v, suf]

    def contrast(al, cl, suf):
        A = [L(v, suf) for v in al]; C = [L(v, suf) for v in cl]; res = {}
        for tag, i_z, i_n, sp in (('val', 0, 2, 'val'), ('hold', 1, 3, 'hold')):
            na = np.stack([a[i_n] for a in A]); nc = np.stack([c[i_n] for c in C])
            pa = R.nd(sp, np.mean([a[i_z] for a in A], 0).astype(np.float32)); pc_ = R.nd(sp, np.mean([c[i_z] for c in C], 0).astype(np.float32))
            res[tag] = (na, nc, pa - pc_, pa, pc_)
        return res

    def pstat(d, mask=None, w=None):  # pool-level: per-feed delta d -> mean, paired per-feed SE, z
        import pmix
        if mask is not None: d = d[mask]
        if w is not None:
            mu, se = pmix.wmean_se(d, w if mask is None else w[mask])
        else:
            mu, se = d.mean(), d.std() / np.sqrt(len(d))
        return dict(d=float(mu), se=float(se), z=float(mu / se) if se > 0 else float('nan'), n=len(d))

    out, keep, pool = {}, {}, {}
    for suf, name in (('', 'selected checkpoints'), ('@last', 'last step')):
        C = contrast(arm, ctl, suf); C1 = contrast(arm, r1, suf); C2 = contrast(arm, r2, suf); keep[suf] = C
        P(f'== {name}: member level, the {len(arm)} new students against the {len(ctl)} stored students (the pool column of these lines is {len(arm)} v {len(ctl)}: information only)')
        na, nc, dp = C['val'][:3]; big = m.Mv >= 50
        out[suf, 'val'] = stat(na, nc, dp); R.line('val', out[suf, 'val'])
        out[suf, 'priv'] = stat(na, nc, dp, w=R.wpriv); R.line('val, private mix', out[suf, 'priv'])
        R.line('val, public mix', stat(na, nc, dp, w=R.wpub)); R.line('val M>=50', stat(na, nc, dp, big)); R.line('val M<50', stat(na, nc, dp, ~big))
        na, nc, dp = C['hold'][:3]
        for k_, lab in HOLD_LINES:
            out[suf, k_] = stat(na, nc, dp, sub[k_]); R.line(lab, out[suf, k_])
        P(f'   by reference arm (information; member level and the EQUAL-SIZE pool contrast {len(arm)} v {len(r1)}):')
        for tag, Cx, rr in (('REF1 w192 (same teacher)', C1, r1), ('REF2 t25w192', C2, r2)):
            na, nc, dp = Cx['val'][:3]; R.line(f'val, vs {tag}', stat(na, nc, dp))
            na, nc, dp = Cx['hold'][:3]; R.line(f'lag-view holdout, vs {tag}', stat(na, nc, dp, sub['lagview']))
        # A4 pool-level contrast: arm pool minus EACH stored 8-pool, mean of the two
        dv = C1['val'][3] - 0.5 * (C1['val'][4] + C2['val'][4]); dh = C1['hold'][3] - 0.5 * (C1['hold'][4] + C2['hold'][4])
        pool[suf, 'val'] = pstat(dv); pool[suf, 'lagview'] = pstat(dh, sub['lagview']); pool[suf, 'priv'] = pstat(dv, w=R.wpriv)
        pool[suf, 'pz'] = (pool[suf, 'val']['z'] + pool[suf, 'lagview']['z']) / np.sqrt(2)
        P(f'   A4 pool level ({len(arm)}-pool of the new students minus each stored 8-pool, mean of the two; paired per-feed SE):')
        for lab, r_ in (('val', pool[suf, 'val']), ('lag-view holdout (Sep 12-19)', pool[suf, 'lagview']), ('val, private mix', pool[suf, 'priv'])):
            P(f'     {lab:34s} n {r_["n"]:6d} | {r_["d"]:+.5f} +- {r_["se"]:.5f} (z {r_["z"]:+.2f})')
        P(f'     pooled z (val, lag-view holdout) = {pool[suf, "pz"]:+.2f}')
    v, p, h, hl, nv = out['', 'val'], out['', 'priv'], out['', 'lagview'], out['@last', 'lagview'], out['', 'noview']
    pooled = (v['z'] + h['z']) / np.sqrt(2)

    # ---------------- stale-val clause (SVL; arm members with the NX twins), member level against the 16
    sv = {}
    if skip_svl:
        P('== stale-val: NOT COMPUTED (NX3_SKIP_SVL=1, dry test)')
    else:
        P(f'== stale-val, scenario SVL (both_pre2lt3); the new students read nx_rp3_z / nx_rp3_rk from their stale-val twins ({resc}/svl)')
        for suf, name in (('', 'selected (the clause)'), ('@last', 'last step (information)')):
            def ld(vn):
                f = resc / 'svl' / f'nn_{vn}{suf}_staleval.npy'
                assert f.exists(), f'missing {f}: run nx3_rescore.py svl first'
                z = m.z('val', np.load(f)); return z, R.nd('val', z)
            A = [ld(x) for x in arm]; C = [ld(x) for x in ctl]
            na = np.stack([a[1] for a in A]); nc = np.stack([c[1] for c in C])
            dpool = R.nd('val', np.mean([a[0] for a in A], 0).astype(np.float32)) - R.nd('val', np.mean([c[0] for c in C], 0).astype(np.float32))
            sv[suf] = stat(na, nc, dpool); R.line(f'stale-val, {name}', sv[suf])
            nat = keep[suf]['val']
            P(f'     levels, member mean: arm natural {nat[0].mean():.5f} -> stale {na.mean():.5f} ({na.mean() - nat[0].mean():+.5f}); '
              f'reference natural {nat[1].mean():.5f} -> stale {nc.mean():.5f} ({nc.mean() - nat[1].mean():+.5f})')

    # ---------------- VALUE report (A3's construction on the B students; NOT a clause of B)
    val_ = {}
    if skip_value:
        P('== VALUE report: NOT COMPUTED (NX3_SKIP_VALUE=1)')
    else:
        P(f'== VALUE report (A3\'s construction; not a clause of B): the {len(arm)} new students re-scored with nx_rp3_z and nx_rp3_rk permuted within the feed ({resc}/perm)')
        def ldp(vn, sp, kind):
            f = resc / 'perm' / f'nn_{vn}_{"val" if sp == "val" else "hold"}_{kind}.npy'
            assert f.exists(), f'missing {f}: run nx3_rescore.py perm first'
            return R.nd(sp, m.z(sp, np.load(f)))
        D = {}
        for sp in ('val', 'hold'):
            nat = np.stack([ldp(x, sp, 'nat') for x in arm]); per = np.stack([ldp(x, sp, 'perm') for x in arm])
            stored = keep['']['val' if sp == 'val' else 'hold'][0]
            P(f'  check, {sp}: re-scored natural vs the stored predictions of the trainer, member NDCG: max |difference| {np.abs(nat.mean(1) - stored.mean(1)).max():.2e} (per feed: max {np.abs(nat - stored).max():.2e})')
            D[sp] = nat - per

        def vstat(d, mask=None):
            if mask is not None: d = d[:, mask]
            pf = d.mean(0); mm = d.mean(1)
            return dict(d=pf.mean(), se=pf.std() / np.sqrt(len(pf)), se_seed=(mm.std(ddof=1) / np.sqrt(len(mm)) if len(mm) > 1 else 0.0), pos=int((mm > 0).sum()), k=len(mm), n=len(pf))

        def vline(lab, r):
            P(f'  {lab:38s} n {r["n"]:6d} | natural - permuted {r["d"]:+.5f} +- {r["se"]:.5f} (paired per-feed SE; z {r["d"] / r["se"] if r["se"] > 0 else float("nan"):+.2f}; seed-level SE {r["se_seed"]:.5f}; {r["pos"]}/{r["k"]} members > 0)')
        val_['val'] = vstat(D['val']); vline('val', val_['val'])
        val_['lagview'] = vstat(D['hold'], sub['lagview']); vline('holdout, lag-view feeds (Sep 12-19)', val_['lagview'])
        P(f'  check: on the {int(sub["noview"].sum()):,} holdout feeds without the view the permutation is a no-op: max |natural - permuted| per feed = {float(np.abs(D["hold"][:, sub["noview"]]).max()):.2e}')

    # ---------------- FINAL BLOCK
    ok = lambda b: 'ok' if b else 'FAIL'
    pv, ph, pz = pool['', 'val'], pool['', 'lagview'], pool['', 'pz']
    P('')
    P('=== nx3 STAGE B: FINAL BLOCK (rule block 2026-11-16T03:45:41Z, STAGE B as amended by A4; A7.2 / A7.3 / A7.5; selected checkpoints, member level against the 16 stored students unless stated) ===')
    P(f' ARM {arm[0]} .. {arm[-1]} ({len(arm)})   REF {r1_spec} + {r2_spec} ({len(ctl)})   stage A: {akind}{" (pooled z %+.2f)" % az if az is not None else ""}')
    c1 = v['z'] >= 1; c2 = h['z'] >= 1; c3 = pooled >= 2; c4 = p['z'] >= -1; c5 = pv['d'] >= 0 and ph['d'] >= 0; c6 = hl['d'] >= 0; c8 = pz >= 1
    P(f' B1 val                       z >= +1.0            : member {v["d"]:+.5f} +- {v["se"]:.5f}, z {v["z"]:+.2f}  -> {ok(c1)}')
    P(f' B2 lag-view holdout (15,225) z >= +1.0            : member {h["d"]:+.5f} +- {h["se"]:.5f}, z {h["z"]:+.2f}  -> {ok(c2)}')
    P(f' B3 pooled z_B = (z_val + z_holdout) / sqrt(2) >= 2.0: {pooled:+.2f}  -> {ok(c3)}   [bands (A4): >= 2 with every clause PASS | [1.6, 2) 4 more students | < 1.6 STOP]')
    P(f' B4 private mix (src/pmix.py) member z >= -1.0     : member {p["d"]:+.5f} +- {p["se"]:.5f}, z {p["z"]:+.2f}  -> {ok(c4)}')
    P(f' B5 pool minus pool >= 0 on val AND lag-view holdout (new 8-pool minus each stored 8-pool, mean of the two): val {pv["d"]:+.5f} +- {pv["se"]:.5f}, holdout {ph["d"]:+.5f} +- {ph["se"]:.5f}  -> {ok(c5)}'
      f'   [8 v 16 pool, information: val {v["dp"]:+.5f} +- {v["sep"]:.5f}, holdout {h["dp"]:+.5f} +- {h["sep"]:.5f}]')
    P(f' B6 last step, lag-view holdout, member delta >= 0  : {hl["d"]:+.5f} +- {hl["se"]:.5f}  -> {ok(c6)}')
    if sv:
        s0 = sv['']; c7 = s0['d'] >= -s0['se']
        P(f' B7 stale-val (SVL + NX twins) member delta >= -1 SE: {s0["d"]:+.5f} +- {s0["se"]:.5f} (z {s0["z"]:+.2f})  -> {ok(c7)}   [last step, information: {sv["@last"]["d"]:+.5f} +- {sv["@last"]["se"]:.5f}]')
    else:
        c7 = None; P(' B7 stale-val: NOT COMPUTED (dry test)')
    P(f' B8 (A4) pool level, pooled z (val, lag-view holdout) >= 1.0: z_val {pv["z"]:+.2f}, z_holdout {ph["z"]:+.2f}, pooled {pz:+.2f}  -> {ok(c8)}')
    sides = v['z'] >= 0 and h['z'] >= 0
    band = 1.6 <= pooled < 2
    P(f' BAND (A4): pooled z_B in [1.6, 2.0): {"yes" if band else "no"}; designer\'s side conditions z_val >= 0 and z_holdout >= 0: {"hold" if sides else "do NOT hold"}'
      + (' (this IS the one re-read: no second round)' if reread else ''))
    P(f' REPORT R1 arm - reference on the 14,775 holdout feeds WITHOUT the view (expected 0): member {nv["d"]:+.5f} +- {nv["se"]:.5f}, z {nv["z"]:+.2f}'
      + ('  -> |z| > 2: side effect of the arm, written next to the verdict' if abs(nv['z']) > 2 else '  -> |z| <= 2: no side effect shown'))
    if val_:
        vv, vh = val_['val'], val_['lagview']
        P(f' REPORT VALUE (not a clause of B): natural - permuted, lag-view holdout {vh["d"]:+.5f} +- {vh["se"]:.5f} against 0.5 x (arm - reference) = {0.5 * h["d"]:+.5f} '
          f'[{"at or above" if vh["d"] >= 0.5 * h["d"] else "BELOW"}]; val {vv["d"]:+.5f} +- {vv["se"]:.5f} [{"> 0" if vv["d"] > 0 else "NOT > 0"}] = the size the students take from the values')
    else:
        P(' REPORT VALUE: not computed')
    P(LIMITS)
    if c7 is None:
        P(' VERDICT: NONE (dry test: the stale-val clause was not computed)')
        return 0
    clauses = dict(B1=c1, B2=c2, B3=c3, B4=c4, B5=c5, B6=c6, B7=c7, B8=c8); failed = [k for k, x in clauses.items() if not x]
    if akind == 'PASS':
        if not failed:
            verdict = ('PASS (stage A = PASS; pooled z_B >= 2 and every clause holds). The 16 all-data students with the bundle (trained under A7.1) are admissible; '
                       'next = the FILE RULE of the block for FDN16P20 (gate PASS + construction review, final val delta vs FD16P20 >= +1 SE, private-mix delta >= 0, val M<50 >= -1 SE). '
                       'The estimate of record of the effect is THIS read (A3 NOTE).')
        elif band and not reread:
            verdict = (f'4 MORE STUDENTS (A4: 1.6 <= pooled z_B < 2): seeds 959-962 through launch_nx3_kd_a7.sh, then ONE re-read of 12 '
                       f'(REREAD=1 ARM=abl_e10_nx3kd05w192:951-962 bash nn_x/w12/nx3/read_stage_b.sh). Clauses failing now: {failed}.'
                       + ('' if sides else ' NOTE: the designer\'s side conditions (z_val >= 0, z_holdout >= 0) do not hold; the lead rules whether the band applies.'))
        elif pooled < 1.6:
            verdict = (f'STOP (A4: pooled z_B < 1.6). No file, no all-data students as pick candidates, no teacher regeneration'
                       + (' unless its own block is opened (A had pooled z >= 3).' if az is not None and az >= 3 else '.') + f' Clauses failing: {failed}.')
        else:
            verdict = (f'NOT PASSED: pooled z_B = {pooled:+.2f}' + (' (in the band, but this is the re-read)' if band else ' >= 2') + f' and clause(s) {failed} fail. A4 names no branch for this case; '
                       'by the designer\'s text "anything else: NOT PASSED". No file becomes a pick candidate.')
    elif akind == 'REPLICATE':
        verdict = 'NONE: stage A reads REPLICATE and its re-read is not on file; by A7.5 stage B is read after A is final. (This line should not be reachable through read_stage_b.sh.)'
    else:
        rep = pooled >= 3.0 and az is not None and az >= 0
        verdict = (f'DESCRIPTIVE RECORD (A7.3: stage A is not a PASS [{akind}]; this is not a second test and no file becomes a pick candidate from it). '
                   f'By the clauses of B it would read: {"every clause holds, pooled z_B >= 2" if not failed else "clauses failing " + str(failed)} (pooled z_B {pooled:+.2f}). '
                   f'Replication block (pooled z_B >= 3.0 AND A\'s pooled z >= 0; A pooled z {"%+.2f" % az if az is not None else "n/a"}): '
                   + ('CONDITION MET -> a replication block may be opened under its own rule (new seeds); nothing is promoted before it reads.' if rep else 'condition not met -> none.'))
    P(f' VERDICT: {verdict}')
    if abs(nv['z']) > 2:
        P(f' NEXT TO THE VERDICT (R1 report): side effect of the arm on the feeds without the view, z {nv["z"]:+.2f}.')
    return 0


if __name__ == '__main__':
    if len(sys.argv) >= 7 and sys.argv[1] == 'read':
        sys.exit(read(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] if len(sys.argv) > 7 else None))
    raise SystemExit('usage: nx3_read_b.py read <ARM> <REF1> <REF2> <rescore dir> <stage-A read file> [out]')
