"""C13 nested LIN residual: [p1, p2, p3] each as z + pctrank (+ isnan) vs control [b]."""
from h import *
H = Harness()
O = Path(HERE) / "out"
xs_v = [np.load(O / f"c13_{n}_val.npy").astype(np.float64) for n in ("p1", "p2", "p3")]
xs_h = [np.load(O / f"c13_{n}_hold.npy").astype(np.float64) for n in ("p1", "p2", "p3")]
for n, x in zip(("p1", "p2", "p3"), xs_v):
    coverage(f"val {n}", x)
H.run("C13_pending_fav", xs_v, xs_h, out_jsonl=O / "results.jsonl")
