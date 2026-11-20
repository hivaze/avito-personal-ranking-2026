"""s2: (a) pairwise per-feed score correlation of all pools on val, public and private (within-feed Pearson of per-feed z;
top-1 agreement); (b) per-feed val NDCG delta vs G3 by feed type for every pool, with the label-free val / public / private
composition of each feed type (over/under-representation of public vs val and vs private)."""
from lib import *  # noqa

f, M, off = meta("val"); y = val_labels()
ft, Mt, ot = meta("test"); pubm = public_mask()
names = SCORED + UNSCORED
test_names = names + OLD

# ---------------------------------------------------------------- (a) pairwise correlation
def zcache(n, split):
    p = OUT / f"z_{n}_{split}.npy"
    if p.exists():
        return np.load(p)
    f_, M_, o_ = meta(split)
    z = zf(o_, M_, pool_scores(n, split)).astype(np.float32); np.save(p, z); return z

def corr_matrix(ns, split, mask=None):
    f_, M_, o_ = meta(split); n = len(M_); fid = np.repeat(np.arange(n), M_)
    Z = {k: zcache(k, split) for k in ns}
    C = np.eye(len(ns)); T = np.eye(len(ns))
    tops = {k: top_item(pool_scores(k, split), M_, o_) for k in ns}
    sel = np.ones(n, bool) if mask is None else mask
    for i, a in enumerate(ns):
        for j, b in enumerate(ns):
            if j <= i:
                continue
            c = np.bincount(fid, Z[a] * Z[b], n) / M_
            C[i, j] = C[j, i] = c[sel].mean()
            T[i, j] = T[j, i] = (tops[a] == tops[b])[sel].mean()
    return C, T

def show(C, ns, title, fmtc="{:7.4f}"):
    print(f"\n{title}")
    print(" " * 8 + "".join(f"{k:>8s}" for k in ns))
    for i, a in enumerate(ns):
        print(f"{a:8s}" + "".join(fmtc.format(C[i, j]).rjust(8) for j in range(len(ns))))

Cv, Tv = corr_matrix(names, "val")
Cp, Tp = corr_matrix(test_names, "test", pubm)
Cq, Tq = corr_matrix(test_names, "test", ~pubm)
show(Cv, names, "(2a) mean within-feed Pearson corr of per-feed z, VAL")
show(Cp, test_names, "(2a) mean within-feed corr, TEST PUBLIC feeds")
show(Cq, test_names, "(2a) mean within-feed corr, TEST PRIVATE feeds")
show(Tp, test_names, "(2a) top-1 agreement rate, PUBLIC", "{:7.3f}")
show(Tq, test_names, "(2a) top-1 agreement rate, PRIVATE", "{:7.3f}")
np.savez(OUT / "s2_corr.npz", names_val=names, names_test=test_names, Cv=Cv, Tv=Tv, Cp=Cp, Tp=Tp, Cq=Cq, Tq=Tq)
g = test_names.index("G3")
print("\n(2a') disagreement with G3 (1 - corr) x 1e3: val | public | private | private/public ratio ; top-1 agreement public | private")
for i, k in enumerate(test_names):
    if k == "G3":
        continue
    dv = (1 - Cv[names.index(k), names.index("G3")]) * 1e3 if k in names else np.nan
    dp, dq = (1 - Cp[i, g]) * 1e3, (1 - Cq[i, g]) * 1e3
    print(f"  {k:8s} {dv:7.3f} | {dp:7.3f} | {dq:7.3f} | {dq / dp:5.3f} ; {Tp[i, g]:.3f} | {Tq[i, g]:.3f}")

# ---------------------------------------------------------------- (b) slices
dv = descriptors("val"); dt = descriptors("test")
g3pos = best_pos_pct(pool_scores("G3", "val"), y, M, off)
Sv = slicings(dv, g3pos); St = slicings(dt)
nd = {n: perfeed_val(n) for n in names}; g3 = nd["G3"]
pub_t, priv_t = dt["public"], ~dt["public"]

print("\n" + "=" * 160)
print("(2b) COMPOSITION of feed types: val% | public% | private%  and ratios pub/val, priv/val, priv/pub (label-free slicings only)")
print("=" * 160)
comp_rows = []
for var in St:
    labs = sorted(set(Sv[var]) | set(St[var]))
    print(f"\n  == {var} ==")
    for lab in labs:
        sv = (Sv[var] == lab).mean(); sp = (St[var][pub_t] == lab).mean(); sq = (St[var][priv_t] == lab).mean()
        flag = ""
        if sv > 0 and sp > 0:
            if abs(np.log(sp / sv)) > 0.3: flag += " PUB!=VAL"
            if abs(np.log(sq / sp)) > 0.3: flag += " PRIV!=PUB"
        print(f"    {lab:14s} val {sv:6.3f} | pub {sp:6.3f} | priv {sq:6.3f} | pub/val {sp / max(sv, 1e-9):5.2f} priv/val {sq / max(sv, 1e-9):5.2f} priv/pub {sq / max(sp, 1e-9):5.2f}{flag}")
        comp_rows.append(dict(var=var, slice=lab, val=sv, pub=sp, priv=sq))
pl.DataFrame(comp_rows).write_csv(OUT / "s2_composition.csv")

print("\n" + "=" * 160)
print("(2c) PER-FEED VAL NDCG DELTA vs G3 BY FEED TYPE (mean ± SE per slice). Slicings with labels (day, npos, pos_cold, pos_cat_in_hist,")
print("     g3_pos_rank = rank pct of the best positive under G3) exist for val only.")
print("=" * 160)
rows = []
groups = [("scored", [n for n in SCORED if n != "G3"]), ("unscored", UNSCORED)]
for var in Sv:
    labs = sorted(set(Sv[var]))
    for gname, gl in groups:
        print(f"\n  == {var} ==  [{gname}]   n  val% pub% priv% | " + " | ".join(f"{n:>17s}" for n in gl))
        for lab in labs:
            m = Sv[var] == lab
            sp = (St[var][pub_t] == lab).mean() if var in St else np.nan
            sq = (St[var][priv_t] == lab).mean() if var in St else np.nan
            cells = []
            for n in gl:
                d = (nd[n] - g3)[m]; mu, se = mse(d)
                cells.append(f"{mu:+.4f}±{se:.4f}")
                rows.append(dict(var=var, slice=lab, pool=n, n=int(m.sum()), val_share=m.mean(), pub_share=sp, priv_share=sq, d=mu, se=se))
            print(f"    {lab:14s} {m.sum():5d} {m.mean():5.3f} {sp:5.3f} {sq:5.3f} | " + " | ".join(f"{c:>17s}" for c in cells))
pl.DataFrame(rows).write_csv(OUT / "s2_slices.csv")

# where is G3's own loss (context): per-slice G3 NDCG and share of total loss
print("\n(2d) G3 val NDCG by the label-defined slices (context: where the remaining loss sits):")
for var in ["npos", "pos_cold", "g3_pos_rank", "day"]:
    for lab in sorted(set(Sv[var])):
        m = Sv[var] == lab
        print(f"    {var:12s} {lab:14s} n {m.sum():5d} ({m.mean():.3f})  G3 {g3[m].mean():.4f}  loss share {(1 - g3[m]).sum() / (1 - g3).sum():.3f}")
