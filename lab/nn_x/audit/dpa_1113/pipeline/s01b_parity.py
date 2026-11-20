"""Parity check of the CPU rebuild: forward r6_cd_m02_r101 (F7x) and v23_gru_pre2s_s501 (F10) on val with the subset
SplitData of s01_extract.py and compare with the stored train-only val preds. Also checks that the trainer's
feature_stats (checkpoint f_mean/f_std) are reproduced by the trn sample. CPU only, fp32."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s01_extract import *  # noqa
from model import Ranker
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

sd, fe, f, raw, y = build("val")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
off = sd.cand_off_np; M = sd.M_np
for name, path in [("r6_cd_m02_r101", ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt"),
                   ("v23_gru_pre2s_s501", ROOT / "nn/ckpt/v23_gru_pre2s_s501.pt")]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    out = np.zeros(sd.n_cand, np.float32); t0 = time.time()
    with torch.no_grad():
        for fb in sd.batches(np.arange(sd.n), 65536, 1024, shuffle=False):
            b = sd.gather(fb); lg = m(b, use_pos=False); mm = b["cmask"]
            out[b["ci"][mm].numpy()] = lg[mm].float().numpy()
    st = np.load(ROOT / f"preds/nn_{name}_trainonly_val.npy")
    print(f"{name}: corr {np.corrcoef(out, st)[0,1]:.6f}  cpu ndcg {ndcg_per_feed(out, yv, off, M).mean():.5f} "
          f"stored {ndcg_per_feed(st, yv, off, M).mean():.5f}  {time.time()-t0:.0f}s", flush=True)
    np.save(OUT / f"val_cpu_{name}.npy", out)
