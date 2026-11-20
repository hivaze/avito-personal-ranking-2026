"""Twin-free acceptance of the 16 ALL-DATA nx3 students (abl_fd10_nx3kd05w192_s951-958, abl_fd10_nx3kd05t25w192_s951-958; file key FDN16P20).
WRITE-NEW adaptation of nn_x/w12/stale/fulldata/accept_wide_twinfree.py (the acceptance that the FD files use for members WITHOUT a held-out twin:
rule G, FD32P20 / FDB16P20; nn_x/w12/stale/fulldata/accept_pf_twinfree.py is the same form for the nn_x/abl harness). Why twin-free: accept_twins.py
(stage F / P) needs the held-out twin of every member; the teacher_mean25 arm has no held-out twin at all (stage B trains teacher_mean11 only) and
the teacher_mean11 twins (stage-B students abl_e10_nx3kd05w192_s951-958) are not fine-tuned and may not exist yet. The checks are therefore made
against the ACCEPTED all-data wide students of the same teacher and seed (w5_fd10_<arm>_s951-958, the members these students replace) and against
the held-out wide students (w5_e10_<arm>_s951-958). CPU, read-only.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/nx3/accept_fdn_twinfree.py structure | full [selftest]

  structure   A1 only. NO metric of any kind is computed (row counts, finiteness, log / cfg lines): may run before the reads of stage A / B (A7.5).
  full        A1 + A2 + A3 (+ T when the stage-B students exist). A2 / A3 / T are numbers of the arm's members: by A7.5 they belong AFTER the reads of
              stage A and stage B (nn_x/w12/nx3/finish_fdn.sh enforces that order; this script does not).
  selftest    (second argument) run the same code on the ACCEPTED all-data pop1 students abl_fd10_pop1kd05w192_s951-954 standing in for the nx3
              students (their held-out students abl_e10_pop1kd05w192 as the twins): proves the code path on stored members only.

  A1 structure   log: train feeds 3,151,208, total steps 99,840, --holdout_in_train True, seed, d 192, teacher file of the arm, 37 train-stage inputs =
                 the reference student's 34 + nx_rp3_z, nx_rp3_rk, has:nx_rp3_z; checkpoint cfg n_dense 107; the four prediction files with the
                 right row counts, finite; the IN-SAMPLE holdout file (and NO file under the out-of-sample name); fine-tune log "abl xt_add honoured".
  A2 positive control   IN-SAMPLE holdout NDCG of each student minus the mean OUT-OF-SAMPLE holdout NDCG of the held-out wide students of its teacher
                 (w5_e10_<arm>_s951-958): must sit in the band of the accepted all-data wide students' lifts (min - 0.002 .. max + 0.002) = "the
                 holdout feeds reached the batches". (The band is wide against any effect the bundle can have on the holdout.)
  A3 level on val (information; no band: the inputs differ)   train-only and fine-tuned val per member; 8-pools and the 16-pool against the pools
                 of the replaced w5_fd10 students, paired per feed.
  T  twins (information; only if all 8 stage-B students are complete)   nx3kd05w192 all-data 8-pool minus the stage-B 8-pool on train-only val, and
                 the in-sample holdout lift against the twins' own out-of-sample holdout."""
import ast, json, os, re, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
mode = sys.argv[1] if len(sys.argv) > 1 else 'structure'; selftest = 'selftest' in sys.argv[2:]
assert mode in ('structure', 'full')
A = ROOT / 'nn_x/abl'; H = ROOT / 'nn_x/w5'; NEW = ['nx_rp3_z', 'nx_rp3_rk', 'has:nx_rp3_z']; S8 = range(951, 959); bad = 0
NV, NT, NH = 1389971, 2904806, 4630966
if selftest:   # stored members only: the accepted all-data pop1 wide students stand in (27 inputs, n_dense 93, teacher_mean11)
    ARMS = {'pop1kd05w192': ('kd05w192', 'teacher_mean11.npy')}; S8 = range(951, 955)
    new_v = lambda arm, s: f'abl_fd10_{arm}_s{s}'; twin_v = lambda s: f'abl_e10_pop1kd05w192_s{s}'; TWIN_ARM = 'pop1kd05w192'
else:
    ARMS = {'nx3kd05w192': ('kd05w192', 'teacher_mean11.npy'), 'nx3kd05t25w192': ('kd05t25w192', 'teacher_mean25.npy')}
    new_v = lambda arm, s: f'abl_fd10_{arm}_s{s}'; twin_v = lambda s: f'abl_e10_nx3kd05w192_s{s}'; TWIN_ARM = 'nx3kd05w192'
print(f'=== accept_fdn_twinfree.py {mode}{" SELFTEST on stored pop1 members" if selftest else ""}: {len(ARMS) * len(S8)} all-data students ===')


def args_of(p):
    ln = open(p).readline(); return ast.literal_eval(ln[ln.index('args: ') + 6:].strip())


def xt_of(p):
    for ln in open(p):
        mm = re.match(r'train-stage extra features \((\d+)\): (.*)', ln)
        if mm: return ast.literal_eval(mm.group(2))


def ld(v, s):
    for d in (A / 'preds', ROOT / 'preds'):
        if (d / f'nn_{v}_{s}.npy').exists(): return np.load(d / f'nn_{v}_{s}.npy')
    return None


def structure(v, arm, s):
    ref_arm, teacher = ARMS[arm]; why = []
    lp = A / f'logs/{v}.log'
    if not (A / f'logs/{v}.json').exists() or not lp.exists(): return False, ['not finished (no logs/<v>.json)']
    log = lp.read_text(); a = args_of(lp); xt = xt_of(lp); ref_xt = xt_of(H / f'logs/w5_fd10_{ref_arm}_s{s}.log')
    if 'train feeds 3151208' not in log or 'total steps 99840' not in log: why.append('train feeds / total steps')
    if a.get('holdout_in_train') is not True or 'FULLDATA --holdout_in_train' not in log: why.append('holdout_in_train')
    if a['seed'] != s or a['d'] != 192 or os.path.basename(a['kd_file']) != teacher or a['kd_w'] != 0.5: why.append(f'seed / d / teacher ({a["seed"]}, {a["d"]}, {os.path.basename(a["kd_file"])})')
    if not selftest:
        if a.get('xt_add') != ','.join(NEW) or a.get('xt_exclude'): why.append(f'xt_add {a.get("xt_add")!r} xt_exclude {a.get("xt_exclude")!r}')
        if xt != ref_xt + NEW: why.append(f'input list ({len(xt or [])} inputs) != the reference student\'s 34 + the bundle')
        import torch
        ck = torch.load(A / f'ckpt/{v}.pt', map_location='cpu', weights_only=False)
        if ck['cfg']['n_dense'] != 107 or ck['cfg']['xt_names'] != ref_xt + NEW: why.append(f'checkpoint cfg n_dense {ck["cfg"]["n_dense"]}')
        fl = A / f'logs/{v}_n2cpv3s3_ft.log'
        if not fl.exists() or "-> 37 xt inputs, n_dense 107" not in fl.read_text() or "abl xt_add honoured: ['nx_rp3_z', 'nx_rp3_rk', 'has:nx_rp3_z'] appended last" not in fl.read_text():
            why.append('fine-tune log: no "37 xt inputs, n_dense 107 ... abl xt_add honoured" line')
    for suf, n in (('trainonly_val', NV), ('trainonly_test', NT), ('n2cpv3s3_val_oof', NV), ('n2cpv3s3_test', NT)):
        x = ld(v, suf)
        if x is None or x.shape != (n,) or not np.isfinite(x).all(): why.append(f'{suf}: {"missing" if x is None else x.shape}')
    hp = A / f'preds/holdout/nn_{v}_holdoutINSAMPLE.npy'
    if not hp.exists() or np.load(hp, mmap_mode='r').shape != (NH,) or not np.isfinite(np.load(hp)).all(): why.append('in-sample holdout file')
    if (A / f'preds/holdout/nn_{v}_holdout.npy').exists(): why.append('a holdout file under the OUT-OF-SAMPLE name exists')
    return not why, why


ok_all = {}
print('A1 structure (no metric is computed):')
for arm in ARMS:
    for s in S8:
        v = new_v(arm, s); ok, why = structure(v, arm, s); ok_all[v] = ok; bad += (not ok)
        print(f'  {v}: A1 {"ok" if ok else "DEFECT " + "; ".join(why)}')
if mode == 'structure':
    print('RESULT (structure only):', 'no structural defect' if bad == 0 else f'{bad} DEFECT(S) / unfinished'); sys.exit(0 if bad == 0 else 1)

m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
f = lambda d: f'{d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f}'
zv = lambda v, suf: m.z('val', ld(v, suf).astype(np.float32))
print('A2 positive control (IN-SAMPLE holdout lift) and A3 member level on val:')
for arm, (ref_arm, _) in ARMS.items():
    if selftest:
        ho = np.mean([eh(m.z('hold', np.load(A / f'preds/holdout/nn_abl_e10_pop1kd05w192_s{s}_holdout.npy').astype(np.float32))).mean() for s in S8]); lo, hi = 0.012, 0.022
        print(f' {arm}: held-out pop1 wide students mean out-of-sample holdout {ho:.5f}; selftest band +0.012..+0.022 (accept_pf_twinfree.py: +0.016..+0.018)')
    else:
        ho = np.mean([eh(m.z('hold', np.load(H / f'preds/holdout/nn_w5_e10_{ref_arm}_s{s}_holdout.npy').astype(np.float32))).mean() for s in S8])
        lifts = [eh(m.z('hold', np.load(H / f'preds/holdout/nn_w5_fd10_{ref_arm}_s{s}_holdoutINSAMPLE.npy').astype(np.float32))).mean() - ho for s in S8]
        lo, hi = min(lifts) - 0.002, max(lifts) + 0.002
        rv = [ev(zv(f'w5_fd10_{ref_arm}_s{s}', 'trainonly_val')).mean() for s in S8]; rf = [ev(zv(f'w5_fd10_{ref_arm}_s{s}', 'n2cpv3s3_val_oof')).mean() for s in S8]
        print(f' {arm}: held-out wide students w5_e10_{ref_arm}_s951-958 mean out-of-sample holdout {ho:.5f}; accepted all-data students w5_fd10_{ref_arm}_s951-958: in-sample lift '
              f'{min(lifts):+.5f}..{max(lifts):+.5f} (band {lo:+.5f}..{hi:+.5f}), train-only val {np.mean(rv):.5f} (min {min(rv):.5f}, max {max(rv):.5f}), FT val {np.mean(rf):.5f}')
    for s in S8:
        v = new_v(arm, s)
        if not ok_all[v]: print(f'  {v}: A1 defect / unfinished: not read'); continue
        hl = eh(m.z('hold', np.load(A / f'preds/holdout/nn_{v}_holdoutINSAMPLE.npy').astype(np.float32))).mean(); lift = hl - ho; a2 = lo <= lift <= hi; bad += (not a2)
        print(f'  {v}: A2 in-sample holdout {hl:.5f} lift {lift:+.5f} {"ok" if a2 else "OUT OF BAND"} | A3 train-only val {ev(zv(v, "trainonly_val")).mean():.5f}, FT val {ev(zv(v, "n2cpv3s3_val_oof")).mean():.5f}')
if all(ok_all.values()) and not selftest:
    print('A3 pools on val (paired per feed; information, no band):')
    for suf in ('trainonly_val', 'n2cpv3s3_val_oof'):
        tot_n, tot_r = [], []
        for arm, (ref_arm, _) in ARMS.items():
            zn = [zv(new_v(arm, s), suf) for s in S8]; zr = [zv(f'w5_fd10_{ref_arm}_s{s}', suf) for s in S8]; tot_n += zn; tot_r += zr
            pn, pr = ev(np.mean(zn, 0)), ev(np.mean(zr, 0))
            print(f'  {suf}: 8-pool {arm} {pn.mean():.5f} minus 8-pool w5_fd10_{ref_arm} ({pr.mean():.5f}): {f(pn - pr)}')
        pn, pr = ev(np.mean(tot_n, 0)), ev(np.mean(tot_r, 0))
        print(f'  {suf}: 16-pool nx3 {pn.mean():.5f} minus 16-pool w5_fd10 (the base pool of FD16P20) ({pr.mean():.5f}): {f(pn - pr)}')
tw = [twin_v(s) for s in S8]
if all((A / f'logs/{t}.json').exists() and (A / f'preds/holdout/nn_{t}_holdout.npy').exists() for t in tw) and all(ok_all[new_v(TWIN_ARM, s)] for s in S8):
    hot = np.mean([eh(m.z('hold', np.load(A / f'preds/holdout/nn_{t}_holdout.npy').astype(np.float32))).mean() for t in tw])
    his = np.mean([eh(m.z('hold', np.load(A / f'preds/holdout/nn_{new_v(TWIN_ARM, s)}_holdoutINSAMPLE.npy').astype(np.float32))).mean() for s in S8])
    pa = ev(np.mean([zv(new_v(TWIN_ARM, s), 'trainonly_val') for s in S8], 0)); pt = ev(np.mean([m.z('val', np.load(A / f'preds/nn_{t}_trainonly_val.npy').astype(np.float32)) for t in tw], 0))
    print(f'T  twins of {TWIN_ARM} (held-out students {tw[0]} .. {tw[-1]}, selected checkpoints): in-sample holdout {his:.5f} minus the twins\' out-of-sample holdout {hot:.5f} = {his - hot:+.5f}; '
          f'train-only val, all-data {len(tw)}-pool {pa.mean():.5f} minus held-out {len(tw)}-pool {pt.mean():.5f}: {f(pa - pt)}  (stored wide students, accept_twins F at 16: A3 +0.00004 +- 0.00042)')
else:
    print(f'T  twins of {TWIN_ARM}: the held-out twins are not all complete: twin lines not computed (the twin-free checks above are the acceptance)')
print('RESULT:', 'no structural defect, positive control in band' if bad == 0 else f'{bad} DEFECT(S) / out of band / unfinished')
sys.exit(0 if bad == 0 else 1)
