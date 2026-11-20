"""Summarise the s02 replay ablations: per-column / per-group paired NDCG deltas vs the member's own CPU base pass,
with SE over 14,423 val feeds and slices (M<50, M>=50, Tue, Thu, coldpos, warmpos); plus the stored stale-val
(private-like coverage) replay of the same member when it exists (nn_x/r6/preds/stale)."""
import sys, json
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
HERE = Path(__file__).resolve().parent; R = HERE / "out/replay"
XT = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo',
      'it_hist_et4_luo', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev', 'ctx_nd_knn_u', 'ctx_nd_knn_u_dev',
      'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent', 'ctx_hn_top1w', 'ctx_hn_rrw', 'ctx_hn_bestrk', 'ctx_hn_margin',
      'ctx_hn_soft05', 'ctx_hn_e1_rrw', 'ctx_hs_mutual', 'ctx_hs_k128', 'ctx_hs_e1', 'ctx_hs_t10', 'pre2_fe_z', 'pre2_new_z',
      'pre2_fe_s', 'pre2_new_s', 'pre2_hmax32', 'has:pre2_fe']
f, M, off = meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
nF = len(M); fid = np.repeat(np.arange(nF), M); npos_f = np.bincount(fid, weights=y, minlength=nF)
day = (f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
cold = np.load(feat_file("val", "it_hist_all_luo")) == 0; coldpos_f = np.bincount(fid, weights=y * cold, minlength=nF)
SL = {"M<50": M < 50, "M>=50": M >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
nd = lambda s: ndcg_per_feed(s, y, off, M)
members = sys.argv[1:] or sorted({p.name.split("_base.npy")[0] for p in R.glob("*_base.npy")})
rows = []
for m in members:
    base = nd(np.load(R / f"{m}_base.npy"))
    print(f"\n==== {m}: base {base.mean():.5f}")
    tags = sorted(p.name[len(m) + 1:-4] for p in R.glob(f"{m}_*.npy") if not p.name.endswith("_base.npy"))
    def line(tag, d):
        se = d.std() / np.sqrt(nF); s = f"{tag:22s} {d.mean():+.5f} ({se:.5f}) z {d.mean()/se:+6.2f}"
        for k, mm in SL.items():
            s += f" | {k} {d[mm].mean():+.5f}"
        print(s); r = dict(member=m, tag=tag, delta=float(d.mean()), se=float(se)); r.update({k: float(d[mm].mean()) for k, mm in SL.items()}); rows.append(r)
    for tag in tags:
        d = nd(np.load(R / f"{m}_{tag}.npy")) - base
        name = tag
        if tag[0] in "zp" and tag[1:].isdigit():
            name = f"{tag[0]}:{XT[int(tag[1:])]}"
        line(name, d)
    st = ROOT / "nn_x/r6/preds/stale" / f"nn_{m}_staleval.npy"
    if st.exists():
        line("STALE-VAL (private-like)", nd(np.load(st)) - base)
json.dump(rows, open(HERE / "out/replay_summary.json", "w"), indent=1)
