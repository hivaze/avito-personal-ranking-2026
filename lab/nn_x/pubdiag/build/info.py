"""INFO ONLY (post-hoc, not part of the pre-declared selection; nothing here can change the pick):
 (a) E1' sensitivity: if the re-cool val excess also sits in the FT half (FT starts from the re-cooled checkpoint),
     the lambda=1 estimate should remove D scaled to the full pool: E1' = Vp - D * (V / Vto)  (only when Vto > 0).
 (b) from-scratch m02 (F13l = v28_lr1_m02 s411/413/415, 3 members) vs its no-m02 recipe twin F9 (v22_pre2s_lr1 s401-406):
     does a NON-re-cooled m02 family show the val-vs-holdout gap that the re-cooled F7x/F7b do?
 (c) CatBoost vs the pool: per-feed correlation and single strength (strength-diversity law break-even)."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
import sys, json
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/nn_x/pubdiag/overfit")
import numpy as np
from lib import Split, FAM, P, H, ROOT, user_groups

OUT = ROOT / "nn_x/pubdiag/build"
J = json.load(open(OUT / "score_out.json"))
print("(a) E1' = Vp - D*(V/Vto)  [post-hoc sensitivity]")
for r in J["rows"]:
    V, Vp, Vto, D = r["V"][0], r["Vp"][0], r["Vto"][0], r["D"][0]
    e1p = Vp - D * (V / Vto) if abs(Vto) > 1e-5 else float("nan")
    print(f"  {r['key']:4s} {r['name']:6s} E0 {r['E0']:+.5f} E1 {r['E1']:+.5f} E1' {e1p:+.5f}  min(E0,E1') {min(r['E0'], e1p):+.5f}")

V_, HO = Split("val"), Split("holdout")
gv, gh = user_groups(V_.users), user_groups(HO.users)
F13l = [f"v28_lr1_m02_s{s}" for s in (411, 413, 415)]
have = [m for m in F13l if (H / f"nn_{m}_holdout.npy").exists() and (P / f"nn_{m}_trainonly_val.npy").exists()]
print(f"\n(b) F13l complete members: {have}")


def fam_stats(ms):
    v = np.array([V_.nd(V_.load(m, "to")) for m in ms]); h = np.array([HO.nd(HO.load(m, "to")) for m in ms])
    return v, h


v13, h13 = fam_stats(have); v9, h9 = fam_stats(FAM["F9"]); v7, h7 = fam_stats(FAM["F7"]); v7x, h7x = fam_stats(FAM["F7x"]); v7b, h7b = fam_stats(FAM["F7b"])
def cmp(a, b, name):
    # member-mean per-feed delta; SE from the per-feed paired difference of member means (cross-seed: includes seed noise only via per-feed variance)
    dv = a[0].mean(0) - b[0].mean(0); dh = a[1].mean(0) - b[1].mean(0)
    sv = np.sqrt((np.bincount(gv, dv - dv.mean()) ** 2).sum()) / len(dv); sh = np.sqrt((np.bincount(gh, dh - dh.mean()) ** 2).sum()) / len(dh)
    # seed-level SE of the member-mean difference (between-member spread of the scalar means)
    ss_v = np.hypot(a[0].mean(1).std(ddof=1) / np.sqrt(len(a[0])), b[0].mean(1).std(ddof=1) / np.sqrt(len(b[0])))
    ss_h = np.hypot(a[1].mean(1).std(ddof=1) / np.sqrt(len(a[1])), b[1].mean(1).std(ddof=1) / np.sqrt(len(b[1])))
    print(f"  {name:28s} single-member TO: val {dv.mean():+.5f} (feed SE {sv:.5f}, seed SE {ss_v:.5f}) | holdout {dh.mean():+.5f} "
          f"(feed SE {sh:.5f}, seed SE {ss_h:.5f}) | val-holdout gap {dv.mean() - dh.mean():+.5f}")
    return dv.mean(), dh.mean()
print("  per-member means val / holdout: F13l", np.round(v13.mean(1), 5), np.round(h13.mean(1), 5), "| F9", np.round(v9.mean(1), 5), np.round(h9.mean(1), 5))
cmp((v13, h13), (v9, h9), "F13l (scratch m02) - F9")
cmp((v7x, h7x), (v7, h7), "F7x (re-cool m02) - F7")
cmp((v7b, h7b), (v7, h7), "F7b (re-cool c0) - F7")
cmp((v7x, h7x), (v7b, h7b), "F7x - F7b (m02 increment)")

print("\n(c) CatBoost vs the G3 pool (val)")
to = np.mean([V_.load(m, "to") for m in FAM["F10"] + FAM["F7x"] + FAM["F7x70"]], 0)
ft = np.mean([V_.load(m, "ft") for m in FAM["F10"] + FAM["F7x"] + FAM["F7x70"]], 0)
pool = 0.5 * V_.z(ft) + 0.5 * V_.z(to)
pn = V_.nd(pool).mean()
for nm in ["cb_tr300k_none", "cb_tr500k_pw", "cb_tr300k_luo"]:
    zc = V_.z(np.load(P / f"{nm}_val.npy"))
    # mean within-feed Pearson of the z-scores
    fid = V_.fid; num = np.bincount(fid, zc * V_.z(pool)); den = np.sqrt(np.bincount(fid, zc ** 2) * np.bincount(fid, V_.z(pool) ** 2))
    rho = np.nanmean(num / np.where(den > 0, den, np.nan))
    print(f"  {nm:16s} val {V_.nd(zc).mean():.5f} (pool {pn:.5f}, gap {pn - V_.nd(zc).mean():.4f}) within-feed rho with pool {rho:.3f}")
    for w in (0.02, 0.05, 0.1):
        d = V_.nd(V_.z(pool) + w * zc) - V_.nd(pool)
        print(f"     pool + {w:.2f}*z: val {d.mean():+.5f} ± {np.sqrt((np.bincount(gv, d - d.mean()) ** 2).sum()) / len(d):.5f}")
