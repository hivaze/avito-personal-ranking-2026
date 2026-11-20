"""Spot check of part B (small segments) with own code: three paired deltas and a nested family choice on H = 0 / H 1-9 / uid0."""
from sklearn.model_selection import GroupKFold
from rv_lib import *
e = np.load(OUT / "final_ndcg.npy")
fin = lambda t: 0.5 * z(np.load(P / f"ens_{t}_ft_val.npy")) + 0.5 * z(np.load(P / f"ens_{t}_to_val.npy"))
CANDS = {"FINAL": e, "FD16 alone": ndcg(final_from(FD16, [], 0)), "pop-light pool": ndcg(final_from(FDP, [], 0)), "popularity only": ndcg(feat("it_hist_all_luo"))}
for w in (0.1, 0.3, 0.4, 0.6, 0.8):
    CANDS[f"w_pop {w}"] = ndcg(final_from(FD16, FDP, w))
for t in ("W8kd", "W16kd", "G3"):
    try:
        CANDS[t + " file"] = ndcg(fin(t))
    except FileNotFoundError:
        pass
SEG = {"H=0": H == 0, "H 1-9": (H >= 1) & (H <= 9), "uid0": UID == 0}
for sn, m in SEG.items():
    line = f"{sn} ({m.sum()} feeds): "
    for k in ("pop-light pool", "FD16 alone", "popularity only"):
        mu, se, _ = cmean(CANDS[k] - e, m)
        line += f"{k} - FINAL {mu:+.4f}+-{se:.4f} | "
    ids = np.where(m)[0]
    names = [k for k in CANDS if k != "popularity only"]
    Em = np.stack([CANDS[k] for k in names], 1)
    reps = []
    for r in range(20):
        rng = np.random.default_rng(r)
        g = CLU[ids]
        perm = rng.permutation(g.max() + 1)
        nested = np.zeros(len(ids))
        for tr_i, te_i in GroupKFold(5).split(ids, None, perm[g]):
            j = int(np.argmax(Em[ids[tr_i]].mean(0)))
            nested[te_i] = Em[ids[te_i], j] - e[ids[te_i]]
        reps.append(nested)
    nm = np.mean(reps, 0)
    full = np.full(NF, np.nan); full[ids] = nm
    mu, se, _ = cmean(full, m)
    best = max(names[1:], key=lambda k: CANDS[k][m].mean())
    line += f"best other family in sample: {best} {CANDS[best][m].mean() - e[m].mean():+.4f} | nested choice (20 x 5 folds by user) {mu:+.4f}+-{se:.4f}"
    print(line)
