"""Private-like witness for pop-light rules on the FULL file form, held-out twin W16P20 (16 w5_e10 students + 4 P1w).
TO base members: stored stale-val replays. FT base members: delta transfer, z(FT_m) + [z(TO_m stale) - z(TO_m fresh)]
(the fine-tuned member starts from the train-only one and sees the same stale inputs at test time; assumption).
Pop-light members fresh (their stale replays move by +0.00001/+0.00012, out/stale_pl_check.log).
Each rule vs the current rule m0.4_0.2, on fresh val and on this stale witness. Information only."""
import ph, numpy as np, pmix
D = ph.Data("val"); P = "<repo>/preds"; SP = "<repo>/lab/nn_x/w5/preds/stale"
W16 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
P1w = [f'abl_e10_pop1kd05w192_s{s}' for s in range(951, 955)]
z = lambda f: ph.zf(np.load(f).astype(np.float32), D.off, D.M)
tof, tos, ftf, fts = [], [], [], []
for m in W16:
    a, b, c = z(f"{P}/nn_{m}_trainonly_val.npy"), z(f"{SP}/nn_{m}_staleval.npy"), z(f"{P}/nn_{m}_n2cpv3s3_val_oof.npy")
    tof.append(a); tos.append(b); ftf.append(c); fts.append(c + (b - a))
pool = lambda L: ph.zf(np.mean(L, 0), D.off, D.M)
TOF, TOS, FTF, FTS = pool(tof), pool(tos), pool(ftf), pool(fts)
PLT = pool([z(f"{P}/nn_{m}_trainonly_val.npy") for m in P1w]); PLF = pool([z(f"{P}/nn_{m}_n2cpv3s3_val_oof.npy") for m in P1w])
Mr = np.repeat(D.M, D.M); big = D.M >= 50; W = pmix.weights()
def fin(ws, wl, FT, TO):
    w = np.where(Mr >= 50, wl, ws)
    return 0.5 * ph.zf((1 - w) * FT + w * PLF, D.off, D.M) + 0.5 * ph.zf((1 - w) * TO + w * PLT, D.off, D.M)
fmt = lambda d, w=None: "{:+.5f}+-{:.5f}".format(*pmix.wmean_se(d, w))
for nm, FT, TO in (("FRESH", FTF, TOF), ("STALE(delta-transfer)", FTS, TOS)):
    ref = D.ev(fin(0.4, 0.2, FT, TO)); print(f"=== {nm}: m0.4_0.2 = {ref.mean():.5f}")
    for ws, wl in ((0.2, 0.2), (0.4, 0.3), (0.4, 0.4), (0.5, 0.3), (0.5, 0.4), (0.5, 0.5), (0.3, 0.3)):
        d = D.ev(fin(ws, wl, FT, TO)) - ref
        print(f"  m{ws}_{wl}: all {fmt(d)} | pub {fmt(d, W['public'])} | priv {fmt(d, W['private'])} | M<50 {fmt(d[~big])} | M>=50 {fmt(d[big])}")
