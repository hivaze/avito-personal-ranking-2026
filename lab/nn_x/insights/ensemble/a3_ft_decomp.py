"""A3: why the FT half stopped adding at pool level.
(1) diversity accounting: cross-member correlation and pool-over-mean-member gain in the TO vs the FT branch;
(2) FT gain (final - TO, FT-only - TO) by segment for family pools, member-averaged and pool-level;
(3) candidate-level shift vectors: s_FT(fam) = z(FT pool) - z(TO pool); s_m02 = z(F7x TO) - z(F7b TO) (same roots, same
    harness); s_recool = z(F7b TO) - z(F7 TO); s_cool = z(F7 TO) - z(F7p TO) (roots 101-106). Correlations on val/test;
    cold/warm means; transplant test (TO pool + lam * s_FT(other family), lam nested over the FT user folds);
(4) what the FT shift follows: per-feature within-feed correlation and multivariate R2 of s_FT on the 42 n2c_pre_v3 extras;
(5) transductive item channel: s_FT and FT pair-gain on candidates whose item was clicked / shown in OTHER val folds,
    and the share of such candidates on test (item clicked / shown anywhere in val).
Writes a3_*.csv; stdout -> a3_ft_decomp.log"""
from lib import *  # noqa

add_v30()
vm, tm = meta("val"), meta("test")
Sv, St = segments("val"), segments("test")
fold = user_folds()
FAMS = dict(FAM); FAMS.update(FAM_EXTRA)
cold_v, cold_t = cand_cold("val"), cand_cold("test")
yv = vm["y"]


def corr_members(ms, split, br):
    zs = [member_z(m, split, br).astype(np.float64) for m in ms]
    c = []
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            c.append(np.mean(zs[i] * zs[j]))
    return np.mean(c)


# ---------- (1) diversity accounting ----------
print("(1) diversity accounting: mean pairwise member corr (candidate-level, per-feed z) and pool - mean-member gain")
print("fam | corr TO val/test | corr FT val/test | corr FIN val | ens gain TO | ens gain FT | ens gain FIN | FT-shift corr across members")
rows = []
for f in list(FAMS) + ["G3"]:
    ms = POOLS["G3"] if f == "G3" else FAMS[f]
    cto_v, cto_t = corr_members(ms, "val", "to"), corr_members(ms, "test", "to")
    cft_v, cft_t = corr_members(ms, "val", "ft"), corr_members(ms, "test", "ft")
    zfin = [0.5 * member_z(m, "val", "to").astype(np.float64) + 0.5 * member_z(m, "val", "ft") for m in ms]
    cfin = np.mean([np.mean(zfeed(zfin[i], "val") * zfeed(zfin[j], "val")) for i in range(len(ms)) for j in range(i + 1, len(ms))])
    eg = {}
    for br in ("to", "ft"):
        mem = np.mean([pf(member_z(m, "val", br)).mean() for m in ms])
        eg[br] = pf(branch(ms, "val", br)).mean() - mem
    memf = np.mean([pf(z).mean() for z in zfin])
    eg["fin"] = pf(final(ms, "val")).mean() - memf
    # FT shift vectors per member and their cross-member correlation
    sh = [member_z(m, "val", "ft").astype(np.float64) - member_z(m, "val", "to") for m in ms]
    csh = np.mean([np.corrcoef(sh[i], sh[j])[0, 1] for i in range(len(ms)) for j in range(i + 1, len(ms))])
    r = dict(fam=f, n=len(ms), corr_to_val=cto_v, corr_to_test=cto_t, corr_ft_val=cft_v, corr_ft_test=cft_t, corr_fin_val=cfin,
             ens_gain_to=eg["to"], ens_gain_ft=eg["ft"], ens_gain_fin=eg["fin"], ftshift_corr=csh)
    rows.append(r)
    print(f"{f:6s} n={len(ms):2d} | {cto_v:.4f}/{cto_t:.4f} | {cft_v:.4f}/{cft_t:.4f} | {cfin:.4f} | {eg['to']:+.5f} | {eg['ft']:+.5f} | "
          f"{eg['fin']:+.5f} | {csh:.3f}", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a3_diversity_accounting.csv")

# ---------- (2) FT gain by segment ----------
print("\n(2) FT gain by segment: pool-level (final - TO) per family pool and member-average (final - TO); x1e4, * = |t|>=2")
seg_fams = ["F5cd", "F6", "F7p", "F7", "F7b", "F7x", "F7x70", "F9", "F10", "F12"] + [f for f in NEW] + ["G3"]
pool_d, mem_d = {}, {}
for f in seg_fams:
    ms = POOLS["G3"] if f == "G3" else FAMS[f]
    zt = zfeed(branch(ms, "val", "to"), "val"); zf = zfeed(branch(ms, "val", "ft"), "val")
    a = pf(zt)
    pool_d[f] = pf(0.5 * zt + 0.5 * zf) - a
    md = np.zeros(vm["n"])
    for m in ms:
        t_ = member_z(m, "val", "to").astype(np.float64); f_ = member_z(m, "val", "ft")
        md += (pf(0.5 * t_ + 0.5 * f_) - pf(t_)) / len(ms)
    mem_d[f] = md
rows = []
for var in ["ALL", "M50", "M", "luo_cold", "cold_bitmap", "user", "cat_in_hist", "n_cat", "H", "day", "pos_cold(label)", "npos(label)"]:
    labs = ["all"] if var == "ALL" else sorted(np.unique(Sv[var]))
    for lvl, D in [("pool", pool_d), ("member", mem_d)]:
        print(f"  [{var}] {lvl}: " + " | ".join(f"{l}" for l in labs))
        for f in seg_fams:
            cells = []
            for l in labs:
                i = np.ones(vm["n"], bool) if var == "ALL" else Sv[var] == l
                mu, se = pse(D[f][i])
                cells.append(f"{mu * 1e4:+6.1f}{'*' if abs(mu) >= 2 * se else ' '}")
                rows.append(dict(var=var, seg=l, level=lvl, fam=f, n=int(i.sum()), d=mu, se=se))
            print(f"     {f:6s} " + " | ".join(cells))
pl.DataFrame(rows).write_csv(OUT / "a3_ftgain_segments.csv")

# ---------- (3) shift vectors ----------
print("\n(3) candidate-level shift vectors (per-feed z differences)")
roots6 = [f"{i}" for i in range(101, 107)]
F7_6 = [f"v17_pre2s_s{r}" for r in roots6]


def zb(ms, split, br):
    return zfeed(branch(ms, split, br), split)


S = {}
for split in ("val", "test"):
    S[split] = {
        "ft_F5cd": zb(FAM["F5cd"], split, "ft") - zb(FAM["F5cd"], split, "to"),
        "ft_F6": zb(FAM["F6"], split, "ft") - zb(FAM["F6"], split, "to"),
        "ft_F7p": zb(FAM_EXTRA["F7p"], split, "ft") - zb(FAM_EXTRA["F7p"], split, "to"),
        "ft_F7": zb(FAM["F7"], split, "ft") - zb(FAM["F7"], split, "to"),
        "ft_F7b": zb(FAM["F7b"], split, "ft") - zb(FAM["F7b"], split, "to"),
        "ft_F7x": zb(FAM["F7x"], split, "ft") - zb(FAM["F7x"], split, "to"),
        "ft_F10": zb(FAM["F10"], split, "ft") - zb(FAM["F10"], split, "to"),
        "ft_V28l": zb(FAM["V28l"], split, "ft") - zb(FAM["V28l"], split, "to"),
        "ft_G3": zb(POOLS["G3"], split, "ft") - zb(POOLS["G3"], split, "to"),
        "cool(F7-F7p)": zb(F7_6, split, "to") - zb(FAM_EXTRA["F7p"], split, "to"),
        "recool(F7b-F7)": zb(FAM["F7b"], split, "to") - zb(FAM["F7"], split, "to"),
        "m02(F7x-F7b)": zb(FAM["F7x"], split, "to") - zb(FAM["F7b"], split, "to"),
        "m02all(F7x-F7)": zb(FAM["F7x"], split, "to") - zb(FAM["F7"], split, "to"),
    }
names = list(S["val"])
print("  corr matrix (val upper, test lower):")
print("  " + " ".join(f"{n[:9]:>9s}" for n in names))
for a, na in enumerate(names):
    cells = []
    for b, nb in enumerate(names):
        sp = "val" if b > a else "test"
        cells.append("        ." if a == b else f"{np.corrcoef(S[sp][na], S[sp][nb])[0, 1]:9.3f}")
    print(f"  {na[:14]:14s}" + " ".join(cells))
print("  shift magnitude (rms) val/test and mean shift on cold (bitmap) vs warm candidates (val | test):")
rows = []
for n in names:
    sv, st = S["val"][n], S["test"][n]
    r = dict(shift=n, rms_val=np.sqrt(np.mean(sv ** 2)), rms_test=np.sqrt(np.mean(st ** 2)),
             cold_val=sv[cold_v].mean(), warm_val=sv[~cold_v].mean(), cold_test=st[cold_t].mean(), warm_test=st[~cold_t].mean(),
             pos_val=sv[yv > 0].mean(), neg_val=sv[yv == 0].mean())
    rows.append(r)
    print(f"    {n:16s} rms {r['rms_val']:.4f}/{r['rms_test']:.4f} | cold {r['cold_val']:+.4f} warm {r['warm_val']:+.4f} | "
          f"test cold {r['cold_test']:+.4f} warm {r['warm_test']:+.4f} | val pos {r['pos_val']:+.4f} neg {r['neg_val']:+.4f}")
pl.DataFrame(rows).write_csv(OUT / "a3_shift_stats.csv")

# transplant: base TO pool + lam * shift, lam nested over user folds
print("\n  transplant: score = z(base TO pool) + lam * s (lam from a grid, chosen on 4 FT folds, applied to the 5th)")
grid = np.array([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
bases = {"F7": FAM["F7"], "F7b": FAM["F7b"], "F7x": FAM["F7x"], "G3": POOLS["G3"], "V28l": FAM["V28l"]}
rows = []
for bn, bms in bases.items():
    zt = zb(bms, "val", "to"); nd0 = pf(zt)
    for sn in ["ft_F5cd", "ft_F6", "ft_F7p", "ft_F7", "ft_F7b", "ft_F7x", "ft_G3", "m02(F7x-F7b)", "cool(F7-F7p)"]:
        nds = np.stack([pf(zt + l * S["val"][sn]) for l in grid])
        nest = np.zeros(vm["n"]); pick = []
        for k in range(5):
            j = int(np.argmax(nds[:, fold != k].mean(1))); pick.append(grid[j])
            nest[fold == k] = nds[j, fold == k]
        ins = nds.mean(1)
        r = dict(base=bn, shift=sn, best_lam=grid[int(np.argmax(ins))], best_ins=ins.max() - nd0.mean(),
                 at_half=nds[list(grid).index(0.5)].mean() - nd0.mean(), nested=pse(nest - nd0)[0], nested_se=pse(nest - nd0)[1],
                 picks=",".join(str(p) for p in pick))
        rows.append(r)
        print(f"    base {bn:5s} + {sn:15s}: lam*={r['best_lam']:+.2f} in-sample {r['best_ins']:+.5f} | lam=0.5 {r['at_half']:+.5f} | "
              f"nested {r['nested']:+.5f}+-{r['nested_se']:.5f} (picks {r['picks']})", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a3_transplant.csv")

# ---------- (4) what the FT shift follows ----------
print("\n(4) FT shift vs the 42 n2c_pre_v3 extras: within-feed corr of s with feature z (val); multivariate R2")
import json
FEATS = None
for ln in open(ROOT / "nn/logs/v17_pre2s_s101_n2cpv3_ft.log"):
    if "extra features (" in ln:
        FEATS = eval(ln.split(": ", 1)[1].strip().split(" ", 1)[1]) if False else eval(ln[ln.index("["):])
        break
FC = ROOT / "nn/feat_cache"
Xv = []
for n in FEATS:
    x = np.load(FC / "val" / f"{n}.npy").astype(np.float64)
    x = np.nan_to_num(x, nan=0.0)
    if x.min() >= 0 and x.max() > 20:
        x = np.log1p(x)
    Xv.append(zfeed(x, "val"))
Xv = np.stack(Xv, 1)
Xv[~np.isfinite(Xv)] = 0
# also trunk-side signals for reference: cold flag z, TO pool score
shifts = ["ft_F5cd", "ft_F6", "ft_F7p", "ft_F7", "ft_F7b", "ft_F7x", "ft_V28l", "ft_G3", "m02(F7x-F7b)", "cool(F7-F7p)"]
rows = []
for sn in shifts:
    s = S["val"][sn]
    cors = np.array([np.corrcoef(s, Xv[:, j])[0, 1] if Xv[:, j].std() > 0 else 0 for j in range(Xv.shape[1])])
    coef, *_ = np.linalg.lstsq(Xv, s, rcond=None)
    r2 = 1 - np.var(s - Xv @ coef) / np.var(s)
    rows.append(dict(shift=sn, r2=r2, **{f: c for f, c in zip(FEATS, cors)}))
    top = np.argsort(-np.abs(cors))[:6]
    print(f"  {sn:15s} R2 {r2:.3f} | top corr: " + ", ".join(f"{FEATS[j]} {cors[j]:+.3f}" for j in top))
pl.DataFrame(rows).write_csv(OUT / "a3_shift_vs_extras.csv")
# group R2
groups = {"pre_*": [j for j, n in enumerate(FEATS) if n.startswith("pre_")],
          "luo_pop": [j for j, n in enumerate(FEATS) if n.startswith("it_hist")],
          "covis/seq": [j for j, n in enumerate(FEATS) if n.startswith("seq_")],
          "ctx/centr": [j for j, n in enumerate(FEATS) if n.startswith(("ctx_", "centrality"))]}
print("  group-only R2 (val):")
for sn in shifts:
    s = S["val"][sn]; out = []
    for g, idx in groups.items():
        c, *_ = np.linalg.lstsq(Xv[:, idx], s, rcond=None)
        out.append(f"{g} {1 - np.var(s - Xv[:, idx] @ c) / np.var(s):.3f}")
    print(f"    {sn:15s} " + " | ".join(out))

# ---------- (5) transductive item channel ----------
print("\n(5) items clicked / shown in OTHER val folds (val) vs in val at all (test)")
item_v = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64)
item_t = np.load(FLAT / "test/cand_item_id.npy").astype(np.int64)
cf = fold[vm["fid"]]
clicked_other = np.zeros(len(item_v), bool); shown_other = np.zeros(len(item_v), bool)
for k in range(5):
    oth = cf != k
    ck = np.unique(item_v[oth & (yv > 0)]); sk = np.unique(item_v[oth])
    i = cf == k
    clicked_other[i] = np.isin(item_v[i], ck); shown_other[i] = np.isin(item_v[i], sk)
clicked_val = np.isin(item_t, np.unique(item_v[yv > 0])); shown_val = np.isin(item_t, np.unique(item_v))
print(f"  val candidates: item shown in other folds {shown_other.mean():.3f}, clicked in other folds {clicked_other.mean():.3f}")
print(f"  test candidates: item shown in val {shown_val.mean():.3f}, clicked in val {clicked_val.mean():.3f} | public "
      f"{clicked_val[St['day'][tm['fid']] == 'public'].mean():.3f} private {clicked_val[St['day'][tm['fid']] == 'private'].mean():.3f}")
print(f"  positives among val candidates by class: clicked-other {yv[clicked_other].mean():.4f} | shown-other-not-clicked "
      f"{yv[shown_other & ~clicked_other].mean():.4f} | unseen {yv[~shown_other].mean():.4f}")
rows = []
for sn in ["ft_F5cd", "ft_F6", "ft_F7", "ft_F7b", "ft_F7x", "ft_V28l", "ft_G3", "m02(F7x-F7b)"]:
    sv, st = S["val"][sn], S["test"][sn]
    r = dict(shift=sn, val_clicked_other=sv[clicked_other].mean(), val_shown_only=sv[shown_other & ~clicked_other].mean(),
             val_unseen=sv[~shown_other].mean(), test_clicked_val=st[clicked_val].mean(),
             test_shown_only=st[shown_val & ~clicked_val].mean(), test_unseen=st[~shown_val].mean())
    rows.append(r)
    print(f"  {sn:15s} mean shift val: clicked-other {r['val_clicked_other']:+.4f} shown-only {r['val_shown_only']:+.4f} "
          f"unseen {r['val_unseen']:+.4f} | test: clicked-in-val {r['test_clicked_val']:+.4f} shown-only {r['test_shown_only']:+.4f} "
          f"unseen {r['test_unseen']:+.4f}")
pl.DataFrame(rows).write_csv(OUT / "a3_transductive.csv")
# how much of the pool FT gain rides on the clicked-other items: neutralise the FT shift on those candidates
print("  FT pool gain with the FT shift removed on clicked-other-fold items (score = z(TO) + 0.5*s*(1-mask)):")
for f in ["F6", "F7", "F7b", "F7x", "G3"]:
    ms = POOLS["G3"] if f == "G3" else FAM[f]
    zt = zb(ms, "val", "to"); zf = zb(ms, "val", "ft"); s = zf - zt
    a = pf(zt); full = pf(zt + 0.5 * s)
    for nm, mask in [("clicked-other", clicked_other), ("shown-other", shown_other)]:
        part = pf(zt + 0.5 * s * (~mask))
        only = pf(zt + 0.5 * s * mask)
        print(f"    {f:5s} full {fmt(full - a)} | without {nm} {fmt(part - a)} | only {nm} {fmt(only - a)}")
np.savez_compressed(CACHE / "a3_items.npz", clicked_other=clicked_other, shown_other=shown_other, clicked_val=clicked_val,
                    shown_val=shown_val)
