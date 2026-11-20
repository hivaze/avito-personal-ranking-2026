"""C09 test: one LIN with rp3b05, p3a on G3. Coverage printed (information)."""
import harness as Hn
from harness import np, log
v = np.load(Hn.CACHE / "c09_val.npz")
h = np.load(Hn.CACHE / "c09_hold.npz")
cov = dict(val_rows_rp3_pos=float((v["p3a"] > 0).mean()), hold_rows_rp3_pos=float((h["p3a"] > 0).mean()),
           val_median_nseq_pos=float(np.median(v["nseq"][v["nseq"] > 0])), hold_median_nseq_pos=float(np.median(h["nseq"][h["nseq"] > 0])))
log(f"coverage {cov}")
info = dict(coverage=cov)
info["rp3b05_alone"] = Hn.lin_info("rp3b05", [v["rp3b05"]], "rp3b05 alone (information)")
Hn.run_item("C09", ["rp3b05", "p3a"], [v["rp3b05"], v["p3a"]], [h["rp3b05"], h["p3a"]], extra=info)
