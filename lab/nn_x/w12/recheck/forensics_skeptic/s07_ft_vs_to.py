"""Does the hard-label val fine-tune (n2cpv3s3, 5-fold OOF by user) still help once the trunk is a KD student?
Member-level and pool-level FT - TO on val, by family, and split by user group (seen in train / new / guest). Read-only on preds/."""
from hx import *
V = load_val(); y, M, off, n, uid = V["y"], V["M"], V["off"], V["n"], V["uid"]
Z = np.load(OUT / "bank_val.npz"); nb_ = [str(s) for s in Z["names"]]; uf = Z["X"][off, nb_.index("u_feeds")]
seen = (uid != 0) & (uf > 0); new = (uid != 0) & (uf == 0); guest = uid == 0
FAM = {"m02 (hard, 7)": [f"r6_cd_m02_r{r}" for r in range(101, 108)], "m02_p70k (hard, 7)": [f"r6_cd_m02_p70k_r{r}" for r in range(101, 108)],
       "gru (hard, 6)": [f"v23_gru_pre2s_s{s}" for s in range(501, 507)], "pop1 (hard, 8)": [f"abl_e10_pop1_s{s}" for s in range(951, 959)],
       "kd05 d128 (KD, 4)": [f"w5_e10_kd05_s{s}" for s in range(951, 955)], "kd08 d128 (KD, 4)": [f"w5_e10_kd08_s{s}" for s in range(951, 955)],
       "kd05t25 d128 (KD, 4)": [f"w5_e10_kd05t25_s{s}" for s in range(951, 955)], "kd05w192 (KD, 4)": [f"w5_e10_kd05w192_s{s}" for s in range(951, 955)],
       "kd05t25w192 (KD, 4)": [f"w5_e10_kd05t25w192_s{s}" for s in range(951, 955)]}
ev = lambda s: ndcg_per_feed(s, y, off, M)
print("family | member TO | member FT-TO (paired, mean over members) | pool TO | pool FT-TO | 0.5/0.5 blend - pool TO | member FT-TO: seen / new")
allh, allk = [], []
for fam, mem in FAM.items():
    ft = [zfeed(np.load(P / f"nn_{m}_n2cpv3s3_val_oof.npy"), off, M) for m in mem]; to = [zfeed(np.load(P / f"nn_{m}_trainonly_val.npy"), off, M) for m in mem]
    dm = np.mean([ev(a) - ev(b) for a, b in zip(ft, to)], 0); mto = np.mean([ev(b).mean() for b in to])
    pf, pt = zfeed(np.mean(ft, 0), off, M), zfeed(np.mean(to, 0), off, M); ept = ev(pt); dp = ev(pf) - ept; db = ev(0.5 * pf + 0.5 * pt) - ept
    (allk if "KD" in fam else allh).append((dm, dp, db))
    print(f"{fam:22s} {mto:.5f} | {fmt(dm)} | {ept.mean():.5f} | {fmt(dp)} | {fmt(db)} | {fmt(dm[seen])} / {fmt(dm[new])}", flush=True)
for nm, L in (("hard-label families", allh), ("KD-student families", allk)):
    dm = np.mean([l[0] for l in L], 0); dp = np.mean([l[1] for l in L], 0); db = np.mean([l[2] for l in L], 0)
    print(f"MEAN over {nm}: member FT-TO {fmt(dm)} | pool FT-TO {fmt(dp)} | blend-TO {fmt(db)} | pool FT-TO seen {fmt(dp[seen])} new {fmt(dp[new])} guest {fmt(dp[guest])}")
dk = np.mean([l[0] for l in allk], 0) - np.mean([l[0] for l in allh], 0); print(f"KD minus hard, member FT-TO: {fmt(dk)}")
dk = np.mean([l[1] for l in allk], 0) - np.mean([l[1] for l in allh], 0); print(f"KD minus hard, pool FT-TO: {fmt(dk)}; seen {fmt(dk[seen])} new {fmt(dk[new])}")
for lo, hi in ((0, 30), (30, 100), (100, 1000)):
    mk = (M >= lo) & (M < hi); print(f"  M in [{lo},{hi}): pool FT-TO hard {fmt(np.mean([l[1] for l in allh], 0)[mk])} | KD {fmt(np.mean([l[1] for l in allk], 0)[mk])}")
