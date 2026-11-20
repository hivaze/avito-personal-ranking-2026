"""Skeptic P4 check 2 (inference-only, CPU): how much does a trained m02 member rely on the pre2 inputs on val?
Forward r6_cd_m02_r101 (F7x) and v23_gru_pre2s_s501 (F10) on val (a) as stored, (b) with the 6 pre2 xt columns put in
the exact Sep 6-8 format (NaN -> 0 after attach_xt, has:pre2_fe = 0; the within-feed rel columns follow inside
dense_features). Paired per-feed NDCG delta with SE, by slice. CPU only, fp32."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s01_extract_copy import *  # noqa  (copy of the proposer's parity-checked CPU rebuild; OUT redirected)
sys.path.insert(0, "<repo>/nn_x/w5")
from model import Ranker
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
sd, fe, f, raw, y = build("val")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
off = sd.cand_off_np; M = sd.M_np
P2 = [i for i, n in enumerate(XT) if "pre2" in n]
print("xt names:", XT); print("pre2 cols:", [XT[i] for i in P2])
xt0 = sd.xt.clone()
for name, path in [("r6_cd_m02_r101", ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt"),
                   ("v23_gru_pre2s_s501", ROOT / "nn/ckpt/v23_gru_pre2s_s501.pt")]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["cfg"]["xt_names"] == XT, "xt layout differs"
    m = Ranker(**ck["cfg"]); m.load_state_dict(ck["state_dict"]); m.eval()
    res = {}
    for mode in ("asis", "blank"):
        sd.xt = xt0.clone()
        if mode == "blank":
            sd.xt[:, P2] = 0
        out = np.zeros(sd.n_cand, np.float32); t0 = time.time()
        with torch.no_grad():
            for fb in sd.batches(np.arange(sd.n), 65536, 1024, shuffle=False):
                b = sd.gather(fb); lg = m(b, use_pos=False); mm = b["cmask"]
                out[b["ci"][mm].numpy()] = lg[mm].float().numpy()
        res[mode] = ndcg_per_feed(out, yv, off, M)
        print(f"{name} {mode}: val {res[mode].mean():.5f} ({time.time()-t0:.0f}s)", flush=True)
    d = res["blank"] - res["asis"]
    s = f"{name} blank - asis: all {d.mean():+.5f} ({d.std()/np.sqrt(len(d)):.5f})"
    for k, g in (("M<50", M < 50), ("M>=50", M >= 50)):
        s += f" | {k} {d[g].mean():+.5f} ({d[g].std()/np.sqrt(g.sum()):.5f})"
    print(s, flush=True)
