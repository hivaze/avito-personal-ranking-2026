"""C01 nested LIN residual: PRIMARY leafsvd_01..64 (z + pctrank) vs control [b].
INFO: [b, gbdt score] vs [b]; [b, score, leafSVD64] vs [b, score] (the spec's discriminator)."""
from h import *
H = Harness()
O = Path(HERE) / "out"
Sv = np.load(O / "c01_svd_val.npy").astype(np.float64); Sh = np.load(O / "c01_svd_hold.npy").astype(np.float64)
sv = np.load(O / "c01_score_val.npy").astype(np.float64); sh = np.load(O / "c01_score_hold.npy").astype(np.float64)
log(f"GBDT score alone: val NDCG {H.nd(sv).mean():.5f} holdout {H.ndh(sh).mean():.5f}; corr(zf score, b) val {np.corrcoef(H.zv(sv), H.b)[0,1]:.4f}")
xs_v = [Sv[:, j] for j in range(64)]; xs_h = [Sh[:, j] for j in range(64)]
res, d = H.run("C01_leafsvd64", xs_v, xs_h, out_jsonl=O / "results.jsonl")
H.run("C01_info_score_vs_b", [sv], [sh], primary=False, out_jsonl=O / "results.jsonl")
Cv, Ch = H.design([sv], [sh])
Gv, Gh = H.design([sv] + xs_v, [sh] + xs_h)
r3, _ = H.evaluate("C01_info_score+leafsvd_vs_score", Gv, Gh, Cv, Ch, primary=False)
with open(O / "results.jsonl", "a") as fh:
    fh.write(json.dumps(r3) + "\n")
