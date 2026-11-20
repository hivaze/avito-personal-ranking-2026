"""hz2 diagnostic AFTER the calibration result (information; label-free): it_hist_all on trPool private-like rows by feed-day group, age bucket and m.
Also the transductive-sparse variant (sparse days after the row counted as on a test row) at the chosen m."""
from views import *
C = load_cache(seq=False)
ftr = feeds('train'); nF = len(ftr)
hold, p = holdout_feeds(ftr); Mf = ftr['M'].to_numpy().astype(np.int64)
pool = (p > 0.5) & (Mf >= 50); pool[hold] = False
part_pub, L, d, smask_f, dist, ppub = feed_params(ftr)
sel = np.flatnonzero(pool & ~part_pub)
qg, coff, M, citem, f4, crow = local_query('train', sel, 0)
ts = ftr['timestamp'].to_numpy().astype(np.int64)[sel]
vb = bucket_of(f4, None).astype(np.int64)
rb, rvals, nref = ref_rows('priv'); rx = rvals['it_hist_all_luo']
zr = [float((rx[rb == k] == 0).mean()) for k in range(7)]; mr = [float(rx[rb == k].mean()) for k in range(7)]
dsel = d[sel]; dg = np.where(dsel <= 2, 0, np.where(dsel <= 6, 1, np.where(dsel <= 9, 2, 3)))
dgr = np.repeat(dg, M); Lr = np.repeat(L[sel], M); dr = np.repeat(dsel, M)
NAMES = ['Sep 6-8', 'Sep 9-12', 'Sep 13-14', 'Sep 17-19']
lg('reference private zero share by bucket ' + ' '.join(f'{z:.3f}' for z in zr) + ' | mean log1p ' + ' '.join(f'{z:.3f}' for z in mr))
lg('feeds by day group ' + str(np.bincount(dg, minlength=4).tolist()))
asb = xt_transform('it_hist_all_luo', np.asarray(np.load(FC / 'train/it_hist_all_luo.npy', mmap_mode='r')[crow]))
for g in range(4):
    mk = dgr == g
    lg(f'as-built   {NAMES[g]:10s}: zero ' + ' '.join(f'{float((asb[mk & (vb == k)] == 0).mean()):.3f}' for k in range(7)) + ' | mean ' + ' '.join(f'{float(asb[mk & (vb == k)].mean()):.3f}' for k in range(7)))
for m in (0, 2, 3, 4, 6, 12):
    dmask, rfac, xtest, pslot = masks_for(coff, M, f4, ts, dsel, L[sel], m)
    o5 = run_it(C, qg, coff, M, citem, dmask, rfac, smask_f[sel])
    x = xt_transform('it_hist_all_luo', o5[:, 0])
    nd = np.zeros(len(dmask), np.int64); last = np.full(len(dmask), -1, np.int64)
    for s in range(15):
        b = ((dmask >> np.uint16(s)) & np.uint16(1)).astype(bool); nd += b; last[b] = s
    old = vb >= 5
    for g in range(4):
        mk = dgr == g
        st = (dr - last)[mk & old & (nd > 0)]
        lg(f'm={m:2d} {NAMES[g]:10s}: zero ' + ' '.join(f'{float((x[mk & (vb == k)] == 0).mean()):.3f}' for k in range(7)) + ' | mean ' + ' '.join(f'{float(x[mk & (vb == k)].mean()):.3f}' for k in range(7))
           + f' | old rows: dense days {float(nd[mk & old].mean()):.2f}, no dense day {float((nd[mk & old] == 0).mean()):.3f}, nearest dense day {float(st.mean()) if len(st) else float("nan"):.2f} d before, rfac {float(rfac[mk & old].mean()):.2f}')
    if m in (3, 12):
        c = dsel - L[sel]; smt = np.zeros(len(sel), np.uint32)
        for o in PAT:
            s = c + o; ok = (s >= 0) & (s <= 14); smt[ok] |= (np.uint32(1) << s[ok].astype(np.uint32))
        o5t = run_it(C, qg, coff, M, citem, dmask, rfac, smt)
        xt_ = xt_transform('it_hist_all_luo', o5t[:, 0])
        lg(f'm={m:2d} TRANSDUCTIVE sparse (all days): zero ' + ' '.join(f'{float((xt_[vb == k] == 0).mean()):.3f}' for k in range(7)) + ' | causal: ' + ' '.join(f'{float((x[vb == k] == 0).mean()):.3f}' for k in range(7)))
        for g in range(4):
            mk = dgr == g
            lg(f'       transductive {NAMES[g]:10s}: zero ' + ' '.join(f'{float((xt_[mk & (vb == k)] == 0).mean()):.3f}' for k in range(7)))
