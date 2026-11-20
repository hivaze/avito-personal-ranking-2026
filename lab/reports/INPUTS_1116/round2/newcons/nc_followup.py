"""newcons follow-up reads F1-F4 of FOLLOWUP.txt (2026-11-16T05:14:30Z). No new construction: the six raw columns of nc_build.py / nc_build_E.py, the same forms.
Witnesses: E = holdout feeds of slots 1-5, fresh view (new feeds); H = holdout lag view; V = val stale-val twin (TO16); VN = natural val.
Null model on the holdout: pool + control block + nx3 files (lag view) + the fresh analogue of the walk on the E feeds. On val: as nc_gates.py.
Subset forms: z / rk re-computed among the rows of the subset (cold = no admitted item-level posting; warm = the rest), 0 elsewhere.
CPU only, 12 threads, read-only. Output out/followup.txt (exclusive create)."""
from nc_common import *
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
P = Log(NCO / 'followup.txt')
nb.set_num_threads(NT)
CONS = [('N1 cw_item', 4, True), ('N2 cw_cell', 5, True), ('N3 aud_cell', 6, False), ('N4 fwd20', 0, True), ('N5 rp3_l8', 9, True), ('N6 aud_item', 2, False)]
CELLC = CONS[:3]
F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
Mt = F['train']['M'].to_numpy().astype(np.int64); Mv = F['val']['M'].to_numpy().astype(np.int64)
tslot = (c9.day_of(F['train']['timestamp'].to_numpy().astype(np.int64)) - hc.DAY0).astype(np.int64); vslot_f = (c9.day_of(F['val']['timestamp'].to_numpy().astype(np.int64)) - hc.DAY0).astype(np.int64)


def offs(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


hold, _p = hc.holdout_feeds(F['train']); Mh = Mt[hold]; offh = offs(Mh); offv = offs(Mv)
co = F['train']['cand_off'].to_numpy().astype(np.int64)[hold]; rows_h = np.repeat(co - offh, Mh) + np.arange(int(Mh.sum()))
HOLD = np.load(NRAW / 'hold_lag.npy'); EH = np.load(NRAW / 'hold_early_fresh.npy'); VN = np.load(NRAW / 'val_nat.npy'); VT = np.load(NRAW / 'val_twin.npy')
hrow = ~np.isnan(HOLD[:, 0]); erow = ~np.isnan(EH[:, 0]); assert not (hrow & erow).any()
slot_h = np.repeat(tslot[hold], Mh); vslot = np.repeat(vslot_f, Mv); vbig = np.repeat(Mv >= 50, Mv)
P(f'E rows {int(erow.sum())} in {int((np.maximum.reduceat(erow.astype(np.int8), offh) > 0).sum())} feeds (slots {np.unique(slot_h[erow]).tolist()}); H rows {int(hrow.sum())}; rows in neither (slot 0) {int((~hrow & ~erow).sum())}')
KK = {nm: (float(1.0 / np.median(VN[:, j][VN[:, j] > 0])) if lg_ else 1.0) for nm, j, lg_ in CONS}
Kr = float(np.load(NX3 / 'out/raw/K.npy')[0])


def forms_of(x, off, M, k, lg_):
    x = np.asarray(x, np.float64)
    return lens.feed_z(np.log1p(k * x) if lg_ else x, off, M, 5.0), lens.feed_rk(x, off, M)


def within(x, mask, off, M, k, lg_):
    y = np.array(x, np.float64); y[~mask] = np.nan
    return forms_of(y, off, M, k, lg_)


@nb.njit(cache=True)
def perm_subset(x, mask, off, M, seed):
    np.random.seed(seed); out = x.copy()
    for f in range(len(M)):
        o = off[f]; m = M[f]; idx = np.empty(m, np.int64); n = 0
        for i in range(m):
            if mask[o + i]:
                idx[n] = o + i; n += 1
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1); t = out[idx[i]]; out[idx[i]] = out[idx[j]]; out[idx[j]] = t
    return out


@nb.njit(parallel=True, cache=True)
def feed_auc(x, y, mask, off, M):
    """per-feed AUC of x among the masked rows (ties 0.5); NaN if the feed has no positive or no negative among them"""
    out = np.full(len(M), np.nan)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; num = 0.0; npair = 0.0
        for i in range(m):
            if mask[o + i] and y[o + i] > 0:
                for k in range(m):
                    if mask[o + k] and y[o + k] == 0:
                        npair += 1.0
                        if x[o + i] > x[o + k]: num += 1.0
                        elif x[o + i] == x[o + k]: num += 0.5
        if npair > 0: out[f] = num / npair
    return out


def auc_line(tag, raw, y, valid, cold, off, M):
    s = []
    for nm, j, _ in CONS:
        x = np.nan_to_num(raw[:, j]); a = feed_auc(x, y, valid, off, M); ac = feed_auc(x, y, valid & cold, off, M); aw = feed_auc(x, y, valid & ~cold, off, M)
        s.append(f'{nm} {np.nanmean(a):.4f} / {np.nanmean(ac):.4f} / {np.nanmean(aw):.4f}')
    P(f'  F4 [{tag}] mean within-feed AUC of the raw column alone (all rows / cold rows / warm rows; feeds: {int(np.isfinite(feed_auc(np.zeros(len(y)), y, valid, off, M)).sum())} / {int(np.isfinite(feed_auc(np.zeros(len(y)), y, valid & cold, off, M)).sum())} / {int(np.isfinite(feed_auc(np.zeros(len(y)), y, valid & ~cold, off, M)).sum())}): ' + ' | '.join(s))


def sub_reads(tag, N, D, raw, off, M, valid, cold, seed0, slices=()):
    """cold-only and warm-only forms of N1-N3 with size, placebo among the subset rows, and feed slices"""
    pls = []
    for ci, (nm, j, lg_) in enumerate(CELLC):
        for sn, mk in (('COLD', valid & cold), ('WARM', valid & ~cold)):
            z, rk = within(raw[:, j], mk, off, M, KK[nm], lg_); a = N.score(z); b = N.score(rk)
            pp = [N.score(perm_subset(z, mk, D['off'], D['M'], seed0 + 100 * ci + 10 * (sn == 'WARM') + s))['cz'] for s in range(5)]; pls += pp
            sl = ''.join(f' | {sname}: {N.score(np.where(smask, z, 0).astype(np.float32))["cz"]:+.2f} / {N.score(np.where(smask, rk, 0).astype(np.float32))["cz"]:+.2f}' for sname, smask in slices)
            P(f'  F1 [{tag}] {nm} {sn}-only: z {a["cz"]:+.2f} / rk {b["cz"]:+.2f} (z: b1 {a["b1"]:+.4f}, dnats per feed {a["dnats"]:.6f}; rk: b1 {b["b1"]:+.4f}, dnats {b["dnats"]:.6f}; rows {int(mk.sum())}) | placebo among the subset rows: ' + ' '.join(f'{x:+.2f}' for x in pp) + sl)
    pls = np.array(pls); P(f'  F3 [{tag}] subset placebo block: {len(pls)} permutations, mean {pls.mean():+.2f}, sd {pls.std():.2f}, max |cz| {np.abs(pls).max():.2f}')


# ---------------------------------------------------------------- holdout: E and H
D = lens.load_split('hold', 'TO16'); condlib.sp = 'hold'; nm_, cl = condlib.control_block(D); assert (D['rows'] == rows_h).all()
zx = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows_h]).astype(np.float32), nan=0.0)
rx = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows_h]).astype(np.float32), nan=0.0)
zE, rE = forms_of(EH[:, 8], offh, Mh, Kr, True)
N0 = condlib.Null(D, nm_, cl)
a, b = N0.score(zE), N0.score(rE); a2, b2 = N0.score(zx), N0.score(rx)
P(f'[holdout] the walk of record under the null WITHOUT it: E (fresh view, slots 1-5) cz {a["cz"]:+.2f} / {b["cz"]:+.2f} (dnats {a["dnats"]:.5f}); H (lag view) {a2["cz"]:+.2f} / {b2["cz"]:+.2f} (dnats {a2["dnats"]:.5f})')
Nx = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk', 'rp3_fresh_z', 'rp3_fresh_rk'], cl + [zx, rx, zE, rE])
coldE = np.nan_to_num(EH[:, 3], nan=1.0) == 0; coldH = np.nan_to_num(HOLD[:, 3], nan=1.0) == 0
P(f'cold share: E rows {coldE[erow].mean():.4f}, H rows {coldH[hrow].mean():.4f}')
pls = []
for ci, (nm, j, lg_) in enumerate(CONS):
    z, rk = forms_of(EH[:, j], offh, Mh, KK[nm], lg_); a = Nx.score(z); b = Nx.score(rk); a0 = N0.score(z); b0 = N0.score(rk)
    pp = [Nx.score(condlib.perm_within(z.astype(np.float32), D['off'], D['M'], 10000 + 10 * ci + s))['cz'] for s in range(5)]; pls += pp
    P(f'  F1 [E whole feed] {nm}: given the walk z {a["cz"]:+.2f} / rk {b["cz"]:+.2f} (b1 {a["b1"]:+.4f}, dnats {a["dnats"]:.6f}, R2 {a["r2"]:.2f}) | without the walk in the null {a0["cz"]:+.2f} / {b0["cz"]:+.2f} | non-zero share {np.mean(EH[erow, j] != 0):.4f} | placebo ' + ' '.join(f'{x:+.2f}' for x in pp))
pls = np.array(pls); P(f'  [E whole feed] placebo block: 30 permutations, mean {pls.mean():+.2f}, sd {pls.std():.2f}, max |cz| {np.abs(pls).max():.2f}')
sub_reads('E', Nx, D, EH, offh, Mh, erow, coldE, 11000, slices=tuple((f'slot {s}', slot_h == s) for s in (1, 2, 3, 4, 5)))
sub_reads('H', Nx, D, HOLD, offh, Mh, hrow, coldH, 12000, slices=(('slots 6-8', (slot_h >= 6) & (slot_h <= 8)), ('slots 11-13', slot_h >= 11)))
auc_line('E', EH, D['y'], erow, coldE, offh, Mh); auc_line('H', HOLD, D['y'], hrow, coldH, offh, Mh)
pb = D['base']
P(f'  F4 pool score alone, mean within-feed AUC (all / cold / warm): E {np.nanmean(feed_auc(pb, D["y"], erow, offh, Mh)):.4f} / {np.nanmean(feed_auc(pb, D["y"], erow & coldE, offh, Mh)):.4f} / {np.nanmean(feed_auc(pb, D["y"], erow & ~coldE, offh, Mh)):.4f}; H {np.nanmean(feed_auc(pb, D["y"], hrow, offh, Mh)):.4f} / {np.nanmean(feed_auc(pb, D["y"], hrow & coldH, offh, Mh)):.4f} / {np.nanmean(feed_auc(pb, D["y"], hrow & ~coldH, offh, Mh)):.4f}')
P(f'  positives on cold rows: E {D["y"][erow & coldE].sum():.0f} of {D["y"][erow].sum():.0f}; H {D["y"][hrow & coldH].sum():.0f} of {D["y"][hrow].sum():.0f}; click rate cold / warm: E {D["y"][erow & coldE].mean():.5f} / {D["y"][erow & ~coldE].mean():.5f}; H {D["y"][hrow & coldH].mean():.5f} / {D["y"][hrow & ~coldH].mean():.5f}')
del N0, Nx, D
# ---------------------------------------------------------------- val: twin and natural
ZT = np.load(NX3 / 'staleval/nx_rp3_z.npy'); RT = np.load(NX3 / 'staleval/nx_rp3_rk.npy'); ZN = np.load(NX3 / 'out/files/val/nx_rp3_z.npy'); RN = np.load(NX3 / 'out/files/val/nx_rp3_rk.npy')
D = lens.load_split('val', 'TO16'); condlib.sp = 'val'; nm_, cl = condlib.control_block(D); allv = np.ones(NROWS['val'], bool)
SL = (('slot 16', vslot == 16), ('slot 18', vslot == 18), ('M < 50', ~vbig), ('M >= 50', vbig))
Nx = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [ZT, RT])
sub_reads('V twin', Nx, D, VT, offv, Mv, allv, VT[:, 3] == 0, 13000, slices=SL)
auc_line('V twin', VT, D['y'], allv, VT[:, 3] == 0, offv, Mv)
del Nx
Nn = condlib.Null(D, nm_ + ['nx_rp3_z', 'nx_rp3_rk'], cl + [ZN, RN])
sub_reads('V natural', Nn, D, VN, offv, Mv, allv, VN[:, 3] == 0, 14000, slices=SL)
auc_line('V natural', VN, D['y'], allv, VN[:, 3] == 0, offv, Mv)
pb = D['base']; cv_ = VT[:, 3] == 0
P(f'  F4 pool score alone, mean within-feed AUC on val (all / twin-cold / twin-warm): {np.nanmean(feed_auc(pb, D["y"], allv, offv, Mv)):.4f} / {np.nanmean(feed_auc(pb, D["y"], cv_, offv, Mv)):.4f} / {np.nanmean(feed_auc(pb, D["y"], ~cv_, offv, Mv)):.4f}')
P(f'  positives on cold rows: V twin {D["y"][cv_].sum():.0f} of {D["y"].sum():.0f}; natural {D["y"][VN[:, 3] == 0].sum():.0f}; click rate twin-cold / twin-warm {D["y"][cv_].mean():.5f} / {D["y"][~cv_].mean():.5f}; share of val rows in feeds with M < 50: {np.mean(~vbig):.4f}, of val positives {D["y"][~vbig].sum() / D["y"].sum():.4f}')
P('DONE')
