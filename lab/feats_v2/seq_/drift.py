"""Drift check: PSI (10 val-quantile bins + NaN bin) val vs test-public vs test-private, plus medians and
the share of zeros / NaNs. Usage: python drift.py <prefix or names>"""
import sys
from seq_lib import *

PUBLIC_TS = 1790497920


def psi(a, b, edges):
    def hist(x):
        nan = np.isnan(x)
        h = np.histogram(np.clip(x[~nan], edges[0], edges[-1]), bins=edges)[0].astype(np.float64)
        h = np.append(h, nan.sum())
        return (h + 1) / (h.sum() + len(h))
    pa, pb = hist(a), hist(b)
    return float(((pa - pb) * np.log(pa / pb)).sum())


def main():
    names = sorted(p.stem for p in (FEATS / "val").glob("*.npy"))
    sel = [n for n in names if any(n.startswith(a) for a in sys.argv[1].split(","))]
    ft = feeds("test")
    pub = (ft["timestamp"].to_numpy() < PUBLIC_TS)[np.asarray(arr("test", "cand_fidx"))]
    rows = []
    for n in sel:
        v = np.load(FEATS / "val" / f"{n}.npy"); t = np.load(FEATS / "test" / f"{n}.npy")
        tp, tq = t[pub], t[~pub]
        vv = v[~np.isnan(v)]
        edges = np.unique(np.quantile(vv, np.linspace(0, 1, 11))) if len(vv) else np.array([0.0, 1.0])
        if len(edges) < 2:
            edges = np.array([edges[0] - 1, edges[0] + 1])
        rows.append((n, psi(v, tp, edges), psi(v, tq, edges), psi(tp, tq, edges),
                     np.nanmedian(v), np.nanmedian(tp), np.nanmedian(tq),
                     np.mean(v == 0), np.mean(tp == 0), np.mean(tq == 0),
                     np.isnan(v).mean(), np.isnan(tp).mean(), np.isnan(tq).mean()))
    print(f"{'feature':26s} {'PSI v-pub':>9s} {'v-priv':>7s} {'pub-priv':>8s} | {'med val':>8s} {'pub':>8s} {'priv':>8s} | "
          f"{'zero v/pub/priv':>17s} | nan v/pub/priv")
    for r in rows:
        print(f"{r[0]:26s} {r[1]:9.4f} {r[2]:7.4f} {r[3]:8.4f} | {r[4]:8.3f} {r[5]:8.3f} {r[6]:8.3f} | "
              f"{r[7]:.3f}/{r[8]:.3f}/{r[9]:.3f} | {r[10]:.3f}/{r[11]:.3f}/{r[12]:.3f}")


if __name__ == "__main__":
    main()
