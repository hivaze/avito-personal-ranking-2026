import hk, clogit, numpy as np
c = hk.Ctx(); r = c.col("rank"); fid = c.col("fidx"); sc = c.score
for K in [10, 30]:
    inh = r <= K
    P, S, Soff = clogit.build_sets(fid, c.y, inh)
    for nm in ["it_hist_et1_luo", "it_hist_all_luo", "pre2_fe_z", "xp_imp_c50"]:
        x = c.col(nm).astype(float)
        q = np.quantile(x[inh], [.5, .9, .97, .99])
        for lab, xx in [("raw_z", (x - x[inh].mean()) / x[inh].std()), (">q90", (x > q[1]).astype(float)), (">q97", (x > q[2]).astype(float)), (">q99", (x > q[3]).astype(float))]:
            b, se, ll = clogit.fit(np.column_stack([sc, xx]), P, S, Soff)
            print(f"K={K} {nm:16s} {lab:6s} b={b[1]:+.3f} z={b[1]/se[1]:+.2f}  (q={q.round(2)})")
