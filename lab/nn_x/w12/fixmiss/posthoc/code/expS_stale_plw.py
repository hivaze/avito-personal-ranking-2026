"""Pop-light weight curve under fresh vs stale (private-like) popularity inputs, train-only branch of the held-out twin W16P20
(16 wide students w5_e10_*; pop-light P1w = abl_e10_pop1kd05w192_s951-954). Stale = stored stale-val replays of the 16
base members (lab/nn_x/w5/preds/stale/*_staleval.npy); pop-light members kept fresh (approximation: they lack the 8
overridden popularity inputs but do read pre2, which the stale scenario also overrides). Information only, no fitting."""
import ph, numpy as np, pmix
D = ph.Data("val"); P = "<repo>/preds"; SP = "<repo>/lab/nn_x/w5/preds/stale"
W16 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
P1w = [f'abl_e10_pop1kd05w192_s{s}' for s in range(951, 955)]
zm = lambda files: ph.zf(np.mean([ph.zf(np.load(f).astype(np.float32), D.off, D.M) for f in files], 0), D.off, D.M)
to_f = zm([f"{P}/nn_{m}_trainonly_val.npy" for m in W16]); to_s = zm([f"{SP}/nn_{m}_staleval.npy" for m in W16])
pl_to = zm([f"{P}/nn_{m}_trainonly_val.npy" for m in P1w])
ft_b = zm([f"{P}/nn_{m}_n2cpv3s3_val_oof.npy" for m in W16]); ft_pl = zm([f"{P}/nn_{m}_n2cpv3s3_val_oof.npy" for m in P1w])
big = D.M >= 50; W = pmix.weights()
def row(e, ref):
    d = e - ref
    return (f"all {d.mean():+.5f}+-{d.std() / np.sqrt(len(d)):.5f} | M<50 {d[~big].mean():+.5f}+-{d[~big].std() / np.sqrt((~big).sum()):.5f} | M>=50 {d[big].mean():+.5f}+-{d[big].std() / np.sqrt(big.sum()):.5f}"
            f" | priv {pmix.wmean_se(d, W['private'])[0]:+.5f}")
e_f0 = D.ev(to_f); e_s0 = D.ev(to_s)
print(f"TO pool W16 alone: fresh {e_f0.mean():.5f}  stale {e_s0.mean():.5f}  (stale-fresh {row(e_s0, e_f0)})")
for nm, base_to, ref in (("FRESH", to_f, None), ("STALE", to_s, None)):
    e_ref = D.ev(0.8 * base_to + 0.2 * pl_to)
    print(f"--- {nm}: TO-branch mix (1-w) base + w pop-light, delta vs w=0.2 (TO only {e_ref.mean():.5f})")
    for w in (0.0, 0.1, 0.3, 0.4, 0.5, 0.6):
        print(f"  w {w:.1f}: " + row(D.ev((1 - w) * base_to + w * pl_to), e_ref))
    # inside the final file form (FT branch fresh, weight rule applied to both branches)
    def fin(w):
        ftm = ph.zf((1 - w) * ft_b + w * ft_pl, D.off, D.M); tom = ph.zf((1 - w) * base_to + w * pl_to, D.off, D.M)
        return D.ev(0.5 * ftm + 0.5 * tom)
    r = fin(0.2); print(f"  final form (FT fresh), ref w 0.2: {r.mean():.5f}")
    for w in (0.3, 0.4, 0.5):
        print(f"  final w {w:.1f}: " + row(fin(w), r))
