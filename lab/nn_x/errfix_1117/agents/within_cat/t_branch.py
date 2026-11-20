from wc import *
df = load(); d = features(df)
a = d["s_ft"].to_numpy().astype(np.float64); b = d["s_to"].to_numpy().astype(np.float64)
s = d["score"].to_numpy().astype(np.float64)
grid = [0.3, 0.4, 0.5, 0.6, 0.7]
r = cross_fit(d, lambda dd, lam: (1 - lam) * a + lam * b, grid)
print("global lam(s_to)", "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
# segment-specific: within-category rows of large categories
for name, mask in [("catn>=20", d["cat_n_in_feed"].to_numpy() >= 20), ("catn>=40", d["cat_n_in_feed"].to_numpy() >= 40),
                   ("dup>=1", d["n_cand_dup95"].to_numpy() >= 1), ("topcat", d["cat_user_rank"].to_numpy() == 1),
                   ("absentcat", d["cat_user_rank"].to_numpy() == 0), ("cold", d["xp_imp_c50"].to_numpy() == 0)]:
    r = cross_fit(d, lambda dd, lam: np.where(mask, (1 - lam) * a + lam * b, s), grid)
    print(name, "oof %.5f se %.5f z %.2f" % (r["delta"], r["se"], r["z"]), r["chosen"], r["full"])
