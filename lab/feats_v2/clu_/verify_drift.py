"""Verifier: drift of KEEP features: val vs test-public vs test-private (boundary ts 1790497920), weekday vs weekend,
per MSK day, quantiles + PSI (val deciles, explicit bins for discrete values, NaN bin). Also per-feed-normalised
versions (within-feed rank is what the ranker sees; feed-level shifts in H move raw counts)."""
import numpy as np, polars as pl
FL = "<repo>/data/flat"; F = "<repo>/data/feats"
K = ["clu_hcnt1024", "clu_hmin1024", "clu_hrw4096", "clu_het14096", "clu_pmi1024_mean", "clu_pmi11024_mean"]
B = 1790497920
def meta(sp):
    fe = pl.read_parquet(f"{FL}/{sp}/feeds.parquet"); ts = fe["timestamp"].to_numpy().astype(np.int64)
    cf = np.load(f"{FL}/{sp}/cand_fidx.npy"); return ts[cf], ((ts + 3*3600)//86400 - 20702)[cf]
tv, dv = meta("val"); tt, dt = meta("test")
def psi(a, b, edges):
    def h(x):
        n = np.isnan(x); c = np.histogram(x[~n], edges)[0].astype(float); c = np.append(c, n.sum()); return (c + 1) / (c.sum() + len(c))
    p, q = h(a), h(b); return float(np.sum((p - q) * np.log(p / q)))
groups = lambda v, t: {"val": v, "pub": t[tt < B], "priv": t[tt >= B], "test_wkend(Sat+Sun)": t[(dt == 20) | (dt == 21)],
                       "test_wkday(Mon+Tue)": t[(dt == 22) | (dt == 23)], "val_Tue": v[dv == 16], "val_Thu": v[dv == 18]}
print("| feature | PSI val-pub | PSI val-priv | PSI pub-priv | PSI val-test_wkend | PSI val-test_wkday | PSI wkend-wkday | PSI valTue-valThu | mean val / pub / priv | median val / pub / priv |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in K:
    v = np.load(f"{F}/val/{k}.npy").astype(np.float64); t = np.load(f"{F}/test/{k}.npy").astype(np.float64)
    vv = v[~np.isnan(v)]
    e = np.unique(np.quantile(vv, np.linspace(0, 1, 11))); e[0] = -np.inf; e[-1] = np.inf
    if len(e) < 6:  # heavily discrete -> integer bins
        e = np.unique(np.concatenate([[-np.inf], np.arange(0, 20) + 0.5, [50.5, 100.5, 998.5, np.inf]]))
    G = groups(v, t)
    ps = [psi(G["val"], G["pub"], e), psi(G["val"], G["priv"], e), psi(G["pub"], G["priv"], e),
          psi(G["val"], G["test_wkend(Sat+Sun)"], e), psi(G["val"], G["test_wkday(Mon+Tue)"], e),
          psi(G["test_wkend(Sat+Sun)"], G["test_wkday(Mon+Tue)"], e), psi(G["val_Tue"], G["val_Thu"], e)]
    mn = [np.nanmean(G[g]) for g in ("val", "pub", "priv")]; md = [np.nanmedian(G[g]) for g in ("val", "pub", "priv")]
    print(f"| {k} | " + " | ".join(f"{x:.4f}" for x in ps) + f" | {mn[0]:.3f} / {mn[1]:.3f} / {mn[2]:.3f} | {md[0]:.3f} / {md[1]:.3f} / {md[2]:.3f} |")
print()
print("per MSK day mean (val 16=Tue,18=Thu; test 20=Sat,21=Sun,22=Mon,23=Tue):")
for k in K:
    v = np.load(f"{F}/val/{k}.npy").astype(np.float64); t = np.load(f"{F}/test/{k}.npy").astype(np.float64)
    s = [f"{d}:{np.nanmean(v[dv == d]):.3f}" for d in (16, 18)] + [f"{d}:{np.nanmean(t[dt == d]):.3f}" for d in (20, 21, 22, 23)]
    print(" ", k, " ".join(s), " share==0/999:", f"{np.nanmean(v == (999 if 'hmin' in k else 0)):.3f}/{np.nanmean(t == (999 if 'hmin' in k else 0)):.3f}")
