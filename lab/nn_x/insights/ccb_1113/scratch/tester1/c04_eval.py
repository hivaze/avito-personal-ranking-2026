"""C04 nested LIN residual: [seq_cv_w5, seq_cv_fwd5, seq_cv_t1, seq_cv_seqs, nb_loglift] (z + pctrank) vs control [b]."""
from h import *
H = Harness()
O = Path(HERE) / "out"
COLS = ["seq_cv_w5", "seq_cv_fwd5", "seq_cv_t1", "seq_cv_seqs", "nb_loglift"]
xs_v = [np.load(O / f"c04_{c}_val.npy").astype(np.float64) for c in COLS]
xs_h = [np.load(O / f"c04_{c}_hold.npy").astype(np.float64) for c in COLS]
for c, x, xh in zip(COLS, xs_v, xs_h):
    coverage(f"val {c}", x); coverage(f"hold {c}", xh)
H.run("C04_covis_variants_nb_loglift", xs_v, xs_h, out_jsonl=O / "results.jsonl")
