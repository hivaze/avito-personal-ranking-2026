"""INFORMATION ONLY (not pre-declared as an item; cannot pass or rescue): z-only LIN (within-feed z of the gap, NaN -> feed min;
no pct rank, no isnan) for A and ALL4, to see whether the non-stationary pct-rank channel (PSI 0.4) is what loses the holdout."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from harness import log_to, fit_lin, cv_lin, nd, ndh, stat, zv, zh, b, bh, yv, yh, SL, H_LATE, H_EARLY
import pmix

LOG = open(HERE / "run.log", "a"); log = log_to(LOG)
V = np.load(HERE / "feats/val.npz"); Hd = np.load(HERE / "feats/hold.npz")
W = pmix.weights()["private"]


def run(item, cols):
    G = np.stack([b] + [zv(np.asarray(V[c], np.float64)) for c in cols], 1)
    c_nd = nd(cv_lin(b[:, None])); p_nd = nd(cv_lin(G)); d = p_nd - c_nd
    dv, se, z = stat(d); pm, pse = pmix.wmean_se(d, W)
    th = fit_lin(G, np.ones(len(yv), bool))
    Gh = np.stack([bh] + [zh(np.asarray(Hd[c], np.float64)) for c in cols], 1)
    dh = ndh(Gh @ th) - ndh(bh); hd, hse, hz = stat(dh); ld = dh[H_LATE].mean(); ed = dh[H_EARLY].mean()
    log(f"INFO z-only {item} ({'+'.join(cols)}): nested val {dv:+.5f} (SE {se:.5f}) z {z:+.2f} | M<50 {d[SL['M<50']].mean():+.5f} M>=50 {d[SL['M>=50']].mean():+.5f} "
        f"coldpos {d[SL['coldpos']].mean():+.5f} warmpos {d[SL['warmpos']].mean():+.5f} | pmix {pm:+.5f} ({pse:.5f}) | theta {np.round(th, 4).tolist()} | "
        f"HOLDOUT {hd:+.5f} (SE {hse:.5f}) z {hz:+.2f} late {ld:+.5f} early {ed:+.5f}")
    with open(HERE / "results_info_zonly.jsonl", "a") as fh:
        fh.write(json.dumps(dict(id=f"zonly_{item}", cols=cols, val_delta=dv, val_se=se, val_z=z, pmix_delta=float(pm), pmix_se=float(pse),
                                 hold_delta=hd, hold_se=hse, late_delta=float(ld), early_delta=float(ed), theta=[float(t) for t in th])) + "\n")


run("A", ["gap_last"])
run("B", ["gap_last_any"])
run("ALL4", ["gap_last", "gap_last_any", "gap_first", "n_days_active"])
