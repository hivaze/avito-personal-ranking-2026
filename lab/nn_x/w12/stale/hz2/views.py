"""hz2 driver. Label-free (no target file is opened).
  python views.py v1                 code check V1: val rows, stored (transductive) rule and strictly causal sparse rule, vs nn/feat_cache/val
  python views.py v2 [n_feeds]       code check V2: random train feeds, as-built rules, vs nn/feat_cache/train
  python views.py calib              calibration of m on trPool private-like feeds (it_hist_all) -> cache/calib.json
  python views.py masks <m>          per-feed lag files + per-row masks for ALL train rows -> cache/, feats/train/hz2_*.npy
  python views.py build_it           it_hist_hz_{all,et1..4} + xp_imp_hz_c50 for all train rows -> feats/train/
  python views.py build_seq <i> <n>  seq kernel on feed part i of n -> cache/seq_part_<i>_<n>.npy
  python views.py finish_seq <n>     seq_cv_hz_{invd20,pop,norm} -> feats/train/
"""
import json
from hz2_common import *
import numba as nb
nb.set_num_threads(min(12, int(os.environ.get('HZ2_THREADS', '12'))))
import kernels as K

VAR = os.environ.get('HZ2_VAR', '')          # '' = primary view (feats/), 'L' = lag-first variant (feats_L/)
SUF = ('_' + VAR) if VAR else ''
OUT = HERE / ('feats' + SUF)


def load_cache(seq=True):
    C = {n: cload(n) for n in ('g_ts', 'g_user', 'g_H', 'g_hoff', 'g_slot', 'g_istr', 'g_sp', 'later_mask', 'lat_start', 'lat_list', 'hitem', 'het',
                               'nidx', 'DM', 'SM', 'IM')}
    if seq:
        C.update({n: cload(n) for n in ('hpos', 'p_start', 'p_feed', 'p_pos', 'dcum', 's_start', 's_feed', 's_pos', 'scum')})
    return C


def rows_of(off, M):
    return np.repeat(off - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(int(M.sum()))


def local_query(split, sel, gbase):
    """Compact row space of the feeds `sel` of `split`: (qg global feed ids, coff, M, citem, f4, crow)."""
    f = feeds(split)
    M = f['M'].to_numpy().astype(np.int64)[sel]
    off = f['cand_off'].to_numpy().astype(np.int64)[sel]
    crow = rows_of(off, M)
    citem = np.asarray(np.load(FLAT / split / 'cand_item_id.npy', mmap_mode='r')[crow]).astype(np.int32)
    f4 = np.asarray(np.load(FLAT / split / 'cand_f4.npy', mmap_mode='r')[crow]).astype(np.float32)
    coff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    return (np.asarray(sel, np.int64) + gbase), coff, M, citem, f4, crow


def feed_smask(d, L):
    """Sparse days of a feed with day slot d and lag L: (d - L) + PAT, restricted to slots 0..d."""
    c = d - L
    sm = np.zeros(len(d), np.uint32)
    for o in PAT:
        s = c + o
        ok = (s >= 0) & (s <= d)
        sm[ok] |= (np.uint32(1) << s[ok].astype(np.uint32))
    return sm


def run_it(C, qg, coff, M, citem, dmask, rfac, smask):
    out = np.zeros((len(citem), 5), np.float32)
    K.ithist_kernel(qg, coff, M, citem, C['nidx'], dmask, rfac, smask, C['DM'], C['SM'], C['g_sp'], C['g_slot'], C['g_istr'], C['g_hoff'], C['g_H'],
                    C['hitem'], C['het'], C['g_user'], C['lat_start'], C['lat_list'], out)
    return out


def run_xp(C, coff, M, citem, dmask, rfac):
    out = np.zeros(len(citem), np.float32)
    K.xp_kernel(coff, M, citem, C['nidx'], dmask, rfac, C['IM'], out)
    return out


def run_seq(C, qg, coff, M, citem, dmask, smask):
    out = np.zeros((len(citem), 4), np.float32)
    K.seq_kernel(qg, coff, M, citem, C['nidx'], dmask, smask, C['hpos'], C['g_hoff'], C['g_H'], C['hitem'], C['g_user'], C['g_ts'], C['later_mask'],
                 C['p_start'], C['dcum'], C['p_feed'], C['p_pos'], C['s_start'], C['scum'], C['s_feed'], C['s_pos'], out)
    return out


def cmp(name, a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    rel = d / (1.0 + np.abs(b))
    lg(f'  {name:18s} rows {len(a)}  exact {float((a == b).mean()):.6f}  max|diff| {d.max():.4g}  max rel {rel.max():.3g}  rows rel > 1e-4: {int((rel > 1e-4).sum())}')
    return dict(exact=float((a == b).mean()), maxabs=float(d.max()), maxrel=float(rel.max()), n_bad=int((rel > 1e-4).sum()))


def nine_cols(o5, o4, xp, rfac):
    invd, pop, norm = K.combine_seq(o4, rfac)
    cols = dict(seq_cv_invd20=invd, seq_cv_pop=pop, seq_cv_norm=norm, xp_imp_c50=xp)
    for i, n in enumerate(LUO):
        cols[n] = o5[:, i]
    return cols


def cmd_v1(split='val'):
    C = load_cache()
    nF = len(feeds('train')); fv = feeds(split); nV = len(fv)
    qg, coff, M, citem, f4, crow = local_query(split, np.arange(nV), nF + (len(feeds('val')) if split == 'test' else 0))
    n = len(citem)
    res = {}
    own_slot = C['g_slot'][qg].astype(np.int64)
    for mode in ('stored', 'causal'):
        dmask = np.full(n, ALL_TR_BITS, np.uint16); rfac = np.ones(n, np.float32)
        if mode == 'stored':
            smask = np.full(nV, ((1 << NSLOT) - 1) & ~ALL_TR_BITS, np.uint32)
        else:
            smask = np.array([sum(1 << s for s in range(NTR, int(o) + 1)) for o in own_slot], np.uint32)
        o5 = run_it(C, qg, coff, M, citem, dmask, rfac, smask)
        o4 = run_seq(C, qg, coff, M, citem, dmask, smask)
        xp = run_xp(C, coff, M, citem, dmask, rfac)
        cols = nine_cols(o5, o4, xp, rfac)
        lg(f'V1 {split} rows, sparse rule = {mode} ({"every val / test snapshot" if mode == "stored" else "sparse days <= own day"}):')
        res[mode] = {c: cmp(c, cols[c], np.load(FC / split / f'{c}.npy')) for c in COLS9}
        if mode == 'causal':
            st = np.load(FC / split / 'it_hist_all_luo.npy')
            lg(f'  causal vs stored it_hist_all_luo: zero share {float((cols["it_hist_all_luo"] == 0).mean()):.4f} vs {float((st == 0).mean()):.4f}; '
               f'rows lower {float((cols["it_hist_all_luo"] < st).mean()):.4f}; mean log1p {float(np.log1p(cols["it_hist_all_luo"]).mean()):.4f} vs {float(np.log1p(st).mean()):.4f}')
    ok = all(res['stored'][c]['exact'] == 1.0 for c in LUO + ['seq_cv_pop']) and all(res['stored'][c]['n_bad'] == 0 for c in COLS9)
    lg(f'V1 {"PASS" if ok else "FAIL"} (stored rule: it_hist_* and seq_cv_pop exact, the others within 1e-4 relative)')
    json.dump(res, open(CACHE / ('v1.json' if split == 'val' else 'v1_test.json'), 'w'), indent=1)


def cmd_v2(n_feeds=20000):
    C = load_cache()
    ftr = feeds('train'); nF = len(ftr)
    sel = np.sort(np.random.default_rng(1).choice(nF, n_feeds, replace=False))
    qg, coff, M, citem, f4, crow = local_query('train', sel, 0)
    n = len(citem)
    slot = C['g_slot'][:nF].astype(np.int64)
    d = slot[sel]
    rfac = np.ones(n, np.float32)
    feed_of_row = np.repeat(np.arange(len(sel)), M)
    # it_hist as built: every train snapshot + every val / test snapshot, minus later same-user snapshots
    dmask = np.full(n, ALL_TR_BITS, np.uint16)
    smask = np.full(len(sel), ((1 << NSLOT) - 1) & ~ALL_TR_BITS, np.uint32)
    o5 = run_it(C, qg, coff, M, citem, dmask, rfac, smask)
    # seq_cv as built: train days before the cutoff day = largest full train day <= own day
    full = DENSE_SLOTS
    cut = full[np.searchsorted(full, d, side='right') - 1]
    dm_seq = ((1 << cut) - 1).astype(np.uint16)[feed_of_row]
    o4 = run_seq(C, qg, coff, M, citem, dm_seq, np.zeros(len(sel), np.uint32))
    # xp_imp as built: train days before the own day
    dm_xp = ((1 << d) - 1).astype(np.uint16)[feed_of_row]
    xp = run_xp(C, coff, M, citem, dm_xp, rfac)
    cols = nine_cols(o5, o4, xp, rfac)
    lg(f'V2 {n_feeds} random train feeds ({n} rows), as-built rules vs nn/feat_cache/train:')
    res = {c: cmp(c, cols[c], np.asarray(np.load(FC / 'train' / f'{c}.npy', mmap_mode='r')[crow])) for c in COLS9}
    ok = all(res[c]['exact'] == 1.0 for c in LUO + ['seq_cv_pop']) and all(res[c]['n_bad'] == 0 for c in COLS9)
    lg(f'V2 {"PASS" if ok else "FAIL"}')
    json.dump(res, open(CACHE / 'v2.json', 'w'), indent=1)


def feed_params(ftr):
    """Per-train-feed hz2 parameters: part_pub, L, day slot d, sparse mask."""
    fid = ftr['feed_id'].to_numpy()
    part_pub, L, dist, ppub = assign_lags(fid)
    d = (day_of(ftr['timestamp'].to_numpy()) - DAY0).astype(np.int64)
    return part_pub, L, d, feed_smask(d, L), dist, ppub


def masks_for(coff, M, f4, ts, d, L, m):
    n = int(M.sum())
    dmask = np.zeros(n, np.uint16); rfac = np.ones(n, np.float32); xtest = np.zeros(n, np.int8); pslot = np.zeros(n, np.int16)
    trunc = np.zeros(n, np.bool_)
    K.row_masks(coff, M, f4, knots().astype(np.float64), ts.astype(np.int64), d.astype(np.int64), L.astype(np.int64), int(m), DPAT, DENSE_BITS,
                dmask, rfac, xtest, pslot, trunc)
    masks_for.trunc = trunc
    return dmask, rfac, xtest, pslot


def bucket_of(f4, ts_row):
    return age_bucket(age_hours(f4, knots())).astype(np.int8)


def ref_rows(part):
    """Test rows of feeds with M >= 50 of the part: (bucket, {col: model-scale values})."""
    fte = feeds('test')
    tts = fte['timestamp'].to_numpy().astype(np.int64); M = fte['M'].to_numpy().astype(np.int64)
    sel = np.flatnonzero(((tts < PUB_TS) if part == 'pub' else (tts >= PUB_TS)) & (M >= 50))
    crow = rows_of(fte['cand_off'].to_numpy().astype(np.int64)[sel], M[sel])
    f4 = np.load(FLAT / 'test/cand_f4.npy')[crow]
    b = age_bucket(age_hours(f4, knots()))
    vals = {c: xt_transform(c, np.load(FC / 'test' / f'{c}.npy')[crow]) for c in COLS9}
    return b, vals, len(sel)


def gate_col(vb, vx, rb, rx):
    """poplag gate statistics of one column: view rows (bucket vb, model-scale vx) vs reference rows (rb, rx)."""
    sd = float(rx.std())
    w = np.bincount(vb, minlength=7) / len(vb)
    zv = np.array([float((vx[vb == k] == 0).mean()) if (vb == k).any() else np.nan for k in range(7)])
    zr = np.array([float((rx[rb == k] == 0).mean()) for k in range(7)])
    mv = np.array([float(vx[vb == k].mean()) if (vb == k).any() else np.nan for k in range(7)])
    mr = np.array([float(rx[rb == k].mean()) for k in range(7)])
    gz = np.abs(zv - zr); gm = np.abs(mv - mr) / sd
    ok = w > 0
    dZ = float(np.sum(w[ok] * gz[ok])); dM = float(np.sum(w[ok] * gm[ok]))
    big = float(np.max(np.where(w >= 0.05, gz, 0.0)))
    return dict(dZ=dZ, dM=dM, big=big, w=w.tolist(), zv=zv.tolist(), zr=zr.tolist(), mv=mv.tolist(), mr=mr.tolist(), sd=sd,
                zero_view=float((vx == 0).mean()), zero_ref=float((rx == 0).mean()))


def cmd_calib(late=False):
    C = load_cache(seq=False)
    ftr = feeds('train'); nF = len(ftr)
    hold, p = holdout_feeds(ftr)
    Mf = ftr['M'].to_numpy().astype(np.int64)
    pool = (p > 0.5) & (Mf >= 50); pool[hold] = False
    part_pub, L, d, smask_f, dist, ppub = feed_params(ftr)
    sel = np.flatnonzero(pool & ~part_pub & ((d >= 7) if late else True))
    lg(f'trPool feeds {int(pool.sum())}; private-like {"late (>= Sep 13) " if late else ""}{len(sel)}; public share {ppub:.4f}; lag dist {dist}')
    qg, coff, M, citem, f4, crow = local_query('train', sel, 0)
    ts = ftr['timestamp'].to_numpy().astype(np.int64)[sel]
    vb = bucket_of(f4, None).astype(np.int64)
    rb, rvals, nref = ref_rows('priv')
    rx = rvals['it_hist_all_luo']
    asb = xt_transform('it_hist_all_luo', np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[crow]))
    g0 = gate_col(vb, asb, rb, rx)
    lg(f'as-built it_hist_all_luo on these rows: dZ {g0["dZ"]:.4f} dM {g0["dM"]:.4f} largest bucket {g0["big"]:.4f}')
    res = {'asbuilt': g0, 'grid': {}}
    for m in (0, 2, 3, 4, 5, 6, 8, 12):
        dmask, rfac, xtest, pslot = masks_for(coff, M, f4, ts, d[sel], L[sel], m)
        o5 = run_it(C, qg, coff, M, citem, dmask, rfac, smask_f[sel])
        g = gate_col(vb, xt_transform('it_hist_all_luo', o5[:, 0]), rb, rx)
        trunc = float((np.array(dmask) != 0).mean())
        res['grid'][m] = g
        lg(f'm = {m:2d}: dZ {g["dZ"]:.4f}  largest bucket (>= 5% rows) {g["big"]:.4f}  dM {g["dM"]:.4f}  zero share view {g["zero_view"]:.4f} ref {g["zero_ref"]:.4f}  '
           f'G1 {"ok" if g["dZ"] <= 0.05 else "no"} G2 {"ok" if g["big"] <= 0.10 else "no"} | zero share by bucket ' + ' '.join(f'{z:.3f}' for z in g['zv']))
    lg('reference zero share by bucket (private, M >= 50): ' + ' '.join(f'{z:.3f}' for z in g0['zr']) + ' | view row share ' + ' '.join(f'{z:.3f}' for z in g0['w']))
    okm = [m for m, g in res['grid'].items() if g['dZ'] <= 0.05 and g['big'] <= 0.10]
    mstar = min(okm) if okm else min(res['grid'], key=lambda m: res['grid'][m]['dZ'])
    res['m_star'] = int(mstar); res['rule'] = 'smallest m meeting G1 and G2' if okm else 'argmin dZ (no m met G1 and G2)'
    lg(f'CALIBRATION: m* = {mstar} ({res["rule"]}); m meeting G1 and G2: {okm}')
    res['m_ok'] = [int(x) for x in okm]
    json.dump(res, open(CACHE / ('calib_late.json' if late else 'calib.json'), 'w'), indent=1)


def cmd_masks(m):
    ftr = feeds('train'); nF = len(ftr)
    part_pub, L, d, smask_f, dist, ppub = feed_params(ftr)
    M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
    f4 = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r'))
    ts = ftr['timestamp'].to_numpy().astype(np.int64)
    dmask, rfac, xtest, pslot = masks_for(coff, M, f4, ts, d, L, m)
    trunc = masks_for.trunc
    np.save(CACHE / f'row_dmask{SUF}.npy', dmask); np.save(CACHE / f'row_rfac{SUF}.npy', rfac); np.save(CACHE / 'row_xtest.npy', xtest); np.save(CACHE / 'row_pslot.npy', pslot)
    np.save(CACHE / 'row_trunc.npy', trunc); np.save(CACHE / 'feed_smask.npy', smask_f)
    drow = np.repeat(d, M)
    keep = trunc & (drow <= 2)
    nd = np.zeros(len(dmask), np.int8)
    for s_ in range(15):
        nd += ((dmask >> np.uint16(s_)) & np.uint16(1)).astype(np.int8)
    lg(f'rows {len(dmask)}: exact {float((~trunc).mean()):.4f} (of which no dense day by rule, X_test = 0: {float(((~trunc) & (xtest == 0)).mean()):.4f}); truncated {float(trunc.mean()):.4f}; '
       f'truncated without any dense day {float((trunc & (nd == 0)).mean()):.4f}; mean depth factor of truncated rows {float(rfac[trunc].mean()):.3f}; keep-mask rows (truncated, Sep 6-8) {float(keep.mean()):.4f}')
    (OUT / 'train').mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'train' / 'hz2_lag.npy', L.astype(np.int8))                 # per-feed lag in days (6..9)
    np.save(OUT / 'train' / 'hz2_part_pub.npy', part_pub)                     # per-feed part (True = public-like)
    np.save(OUT / 'train' / 'hz2_cut_slot.npy', (d - L).astype(np.int16))     # per-feed lagged cutoff, day slot (0 = Sep 6)
    np.save(OUT / 'train' / 'hz2_keep_rows.npy', keep)                        # truncated rows of feeds on Sep 6-8 (option HZ2_KEEP of the drop-in)
    lg(f'm = {m}; lag counts {dict(zip(*[x.tolist() for x in np.unique(L, return_counts=True)]))}; public-like share {float(part_pub.mean()):.4f} (test {ppub:.4f})')
    json.dump(dict(m=int(m), lag_counts={int(a): int(b) for a, b in zip(*np.unique(L, return_counts=True))}, pub_share=float(part_pub.mean())),
              open(CACHE / f'masks{SUF}.json', 'w'))
    lg('masks saved')


def cmd_build_it():
    C = load_cache(seq=False)
    ftr = feeds('train'); nF = len(ftr)
    M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
    citem = np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r'))
    dmask, rfac, smask = cload(f'row_dmask{SUF}'), cload(f'row_rfac{SUF}'), cload('feed_smask')
    o5 = run_it(C, np.arange(nF, dtype=np.int64), coff, M, citem, dmask, rfac, smask)
    lg('it_hist kernel done')
    for i, n in enumerate(LUO):
        np.save(OUT / 'train' / f'{HZ2_NAME[n]}.npy', np.ascontiguousarray(o5[:, i]))
    lg('it_hist_hz_* written; zero share all ' + f'{float((o5[:, 0] == 0).mean()):.4f}')
    del o5
    xp = run_xp(C, coff, M, citem, dmask, rfac)
    np.save(OUT / 'train' / f'{HZ2_NAME["xp_imp_c50"]}.npy', xp)
    lg(f'xp_imp_hz_c50 written; zero share {float((xp == 0).mean()):.4f}')


def cmd_build_seq(i, n):
    C = load_cache()
    ftr = feeds('train'); nF = len(ftr)
    M = ftr['M'].to_numpy().astype(np.int64); coff = ftr['cand_off'].to_numpy().astype(np.int64)
    bounds = np.searchsorted(np.cumsum(M), np.linspace(0, M.sum(), n + 1)[1:-1]).tolist()
    lo = [0] + bounds; hi = bounds + [nF]
    a, b = lo[i], hi[i]
    r0, r1 = int(coff[a]), int(coff[b - 1] + M[b - 1])
    citem = np.asarray(np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[r0:r1])
    dmask = cload(f'row_dmask{SUF}')[r0:r1]; smask = cload('feed_smask')[a:b]
    lg(f'part {i}/{n}: feeds {a}..{b} rows {r0}..{r1}')
    CH = 100000
    out = np.zeros((r1 - r0, 4), np.float32)
    for s in range(a, b, CH):
        e = min(b, s + CH)
        q0, q1 = int(coff[s]) - r0, int(coff[e - 1] + M[e - 1]) - r0
        o = run_seq(C, np.arange(s, e, dtype=np.int64), coff[s:e] - coff[s], M[s:e], citem[q0:q1], dmask[q0:q1], smask[s - a:e - a])
        out[q0:q1] = o
        if ((s - a) // CH) % 3 == 0:
            lg(f'  feeds {s}..{e} done')
    np.save(CACHE / f'seq_part{SUF}_{i}_{n}.npy', out)
    json.dump(dict(a=a, b=b, r0=r0, r1=r1), open(CACHE / f'seq_part{SUF}_{i}_{n}.json', 'w'))
    lg('part done')


def cmd_finish_seq(n):
    rfac = cload(f'row_rfac{SUF}')
    N = len(rfac)
    invd = np.zeros(N, np.float32); pop = np.zeros(N, np.float32); norm = np.zeros(N, np.float32)
    for i in range(n):
        j = json.load(open(CACHE / f'seq_part{SUF}_{i}_{n}.json'))
        o = np.load(CACHE / f'seq_part{SUF}_{i}_{n}.npy')
        a, b, c = K.combine_seq(o, rfac[j['r0']:j['r1']])
        invd[j['r0']:j['r1']] = a; pop[j['r0']:j['r1']] = b; norm[j['r0']:j['r1']] = c
    for nme, x in (('seq_cv_invd20', invd), ('seq_cv_pop', pop), ('seq_cv_norm', norm)):
        np.save(OUT / 'train' / f'{HZ2_NAME[nme]}.npy', x)
    lg(f'seq_cv_hz_* written; pop zero share {float((pop == 0).mean()):.4f}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd in ('masks', 'build_it', 'finish_seq') and (HERE / 'LIVE_DO_NOT_REBUILD').exists() and os.environ.get('HZ2_FORCE') != '1':
        raise SystemExit(f'views.py {cmd}: refused, {HERE}/LIVE_DO_NOT_REBUILD exists (the view files are read by live training lanes; see that file)')
    if cmd == 'v1':
        cmd_v1()
    elif cmd == 'v1t':
        cmd_v1('test')
    elif cmd == 'v2':
        cmd_v2(int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    elif cmd == 'calib':
        cmd_calib()
    elif cmd == 'calib_late':
        cmd_calib(late=True)
    elif cmd == 'masks':
        cmd_masks(int(sys.argv[2]))
    elif cmd == 'build_it':
        cmd_build_it()
    elif cmd == 'build_seq':
        cmd_build_seq(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == 'finish_seq':
        cmd_finish_seq(int(sys.argv[2]))
    lg('done')
