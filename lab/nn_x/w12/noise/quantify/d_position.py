"""(D) position noise in train labels: click rate by display rank in production vs random-order train feeds.

Feeds: train, M >= 50 (where the stored detector works), p_random from artifacts/exploration/train_prandom.npz (the file
the trainer reads, configs/paths.toml [exploration] probability). Random-order pool: p > 0.9 (pure) and p > 0.5 (the
trainer's pool, 164,118 feeds); production: p < 0.05. The detector uses no clicks.
Display rank = rank of `pos` inside the logged feed (0 = top). The last logged item is a separate cell (the logged
feed ends at the last interaction, so it is positive 18-29% of the time in both regimes).

Attribution (examination hypothesis, same examination curve in both regimes): within each M bin,
  e(c) = rate_random(c) / mean rate_random over non-last rows       (examination factor of rank cell c)
  clicks a production feed would have had under a flat examination = sum_c clicks_prod(c) / e(c)
  share of production clicks attributable to position = 1 - that / actual clicks   (non-last rows)
SE by a Poisson bootstrap over feeds (200 replicates).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

nb.set_num_threads(12)
t0 = time.time()
F, M, off = layout("train")
pr = np.load(ROOT / "artifacts/exploration/train_prandom.npz")
assert (pr["fidx"] == np.arange(len(M))).all()
p = pr["p_random"].astype(np.float64)
pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
tgt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
print(f"train feeds {len(M)}, rows {len(pos)}; loaded {time.time() - t0:.0f}s", flush=True)

EDGES = np.array([0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 10**6])     # rank cells for non-last rows; cell 10 = last row
NC = len(EDGES)                                                  # 10 rank cells + last
CELL_NAMES = [f"{a}-{b - 1}" if b - a > 1 else f"{a}" for a, b in zip(EDGES[:-2], EDGES[1:-1])] + ["200+", "last"]
MBINS = [(50, 74), (75, 99), (100, 149), (150, 199), (200, 249), (250, 300)]


@nb.njit(parallel=True, cache=True)
def cell_counts(sel, off, M, pos, tgt, edges):
    n = len(sel)
    rows = np.zeros((n, len(edges)), np.int32)
    clk = np.zeros((n, len(edges)), np.int32)
    for q in nb.prange(n):
        f = sel[q]; o = off[f]; m = M[f]
        pp = np.empty(m, np.int64)
        for j in range(m):
            pp[j] = pos[o + j]
        order = np.argsort(pp, kind="mergesort")
        for r in range(m):
            i = o + order[r]
            if r == m - 1:
                c = len(edges) - 1
            else:
                c = 0
                while c < len(edges) - 2 and r >= edges[c + 1]:
                    c += 1
            rows[q, c] += 1
            clk[q, c] += tgt[i]
    return rows, clk


def collect(mask):
    sel = np.where(mask)[0].astype(np.int64)
    r, c = cell_counts(sel, off, M, pos, tgt, EDGES)
    return sel, r.astype(np.float64), c.astype(np.float64)


ok = (M >= 50) & ~np.isnan(p)
pools = {"random p>0.9": ok & (p > 0.9), "random p>0.5": ok & (p > 0.5), "production p<0.05": ok & (p < 0.05)}
data = {}
for k, m in pools.items():
    sel, r, c = collect(m)
    data[k] = (sel, r, c, np.searchsorted([b for _, b in MBINS], M[sel], side="left"))
    print(f"{k}: {len(sel)} feeds, {r.sum():.0f} rows, {c.sum():.0f} clicks, rate {c.sum() / r.sum():.5f}, "
          f"last item positive {c[:, -1].sum() / r[:, -1].sum():.4f}, share of clicks on the last item {c[:, -1].sum() / c.sum():.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


def stats(rand, prod, w_r=None, w_p=None):
    """Per-bin e(c) and relevance ratio; click-relocation metrics of production clicks (non-last rows)."""
    sr, rr, cr, br = rand
    sp, rp, cp, bp = prod
    if w_r is not None:
        rr = rr * w_r[:, None]; cr = cr * w_r[:, None]; rp = rp * w_p[:, None]; cp = cp * w_p[:, None]
    E = np.full((len(MBINS), NC), np.nan)
    REL = np.full((len(MBINS), NC), np.nan)
    A_, CF_, ROWS_, LOGE_ = [], [], [], []
    for b in range(len(MBINS)):
        Rr = rr[br == b].sum(0); Cr = cr[br == b].sum(0); Rp = rp[bp == b].sum(0); Cp = cp[bp == b].sum(0)
        ok_ = (Rr > 0) & (Rp > 0) & (Cr > 0)
        rate_r = np.where(ok_, Cr / np.maximum(Rr, 1), np.nan); rate_p = np.where(Rp > 0, Cp / np.maximum(Rp, 1), np.nan)
        bar_r = Cr[:-1].sum() / Rr[:-1].sum(); bar_p = Cp[:-1].sum() / Rp[:-1].sum()
        e = rate_r / bar_r
        E[b] = e
        REL[b] = (rate_p / rate_r) / (bar_p / bar_r)
        v = ok_[:-1]
        A_.append(Cp[:-1][v]); CF_.append(Cp[:-1][v] / e[:-1][v]); ROWS_.append(Rp[:-1][v]); LOGE_.append(np.log(e[:-1][v]))
    a = np.concatenate(A_); cf = np.concatenate(CF_); rows_ = np.concatenate(ROWS_); le = np.concatenate(LOGE_)
    ee = np.exp(le)
    out = dict(E=E, REL=REL)
    out["net"] = 1 - cf.sum() / a.sum()                                    # change in the click count under flat examination
    out["tv"] = 0.5 * np.abs(a / a.sum() - cf / cf.sum()).sum()            # click mass relocated across rank cells
    out["excess"] = (a * np.clip(1 - 1 / ee, 0, None)).sum() / a.sum()      # clicks owed to above-average examination
    out["missing"] = (a * np.clip(1 / ee - 1, 0, None)).sum() / a.sum()     # clicks lost to below-average examination
    w = (1 / ee) / ((a / ee).sum() / a.sum())
    out["w_min"], out["w_max"] = w.min(), w.max()
    out["ess"] = (a * w).sum() ** 2 / (a * w * w).sum() / a.sum()
    m_le = (rows_ * le).sum() / rows_.sum()
    out["sd_loge_rows"] = np.sqrt((rows_ * (le - m_le) ** 2).sum() / rows_.sum())
    Rr = rr.sum(0); Cr = cr.sum(0); Rp = rp.sum(0); Cp = cp.sum(0)
    out.update(rate_r=Cr / Rr, rate_p=Cp / Rp, top5_r=Cr[:4].sum() / Cr[:-1].sum(), top5_p=Cp[:4].sum() / Cp[:-1].sum(),
               rows5_r=Rr[:4].sum() / Rr[:-1].sum(), rows5_p=Rp[:4].sum() / Rp[:-1].sum(),
               base_r=Cr[:-1].sum() / Rr[:-1].sum(), base_p=Cp[:-1].sum() / Rp[:-1].sum(), share=out["excess"])
    return out


rng = np.random.default_rng(11)
prod = data["production p<0.05"]
for rk_name in ("random p>0.9", "random p>0.5"):
    rand = data[rk_name]
    S = stats(rand, prod)
    BS = [stats(rand, prod, rng.poisson(1.0, len(rand[0])).astype(np.float64), rng.poisson(1.0, len(prod[0])).astype(np.float64))
          for _ in range(200)]
    print(f"\n=== random-order reference: {rk_name} ===")
    for key, txt in [("excess", "share of production clicks owed to above-average examination (top slots), sum a(1-1/e)+ / A"),
                     ("missing", "clicks lost to below-average examination (deep slots), sum a(1/e-1)+ / A"),
                     ("tv", "click mass relocated across rank cells by examination (TV of actual vs flat-exam counterfactual)"),
                     ("net", "net change of the click count under a flat examination, 1 - CF/A"),
                     ("sd_loge_rows", "row-weighted sd of log e(rank) inside production feeds (logit-scale position offset)"),
                     ("ess", "ESS/n of IPS weights 1/e over production clicks")]:
        bs = np.array([b[key] for b in BS])
        print(f"  {txt}: {S[key]:.4f} ± {bs.std(ddof=1):.4f}")
    print(f"  IPS weights normalised to mean 1 over clicks: min {S['w_min']:.3f} max {S['w_max']:.3f}")
    print(f"share of non-last clicks on display ranks 0-4: random {S['top5_r']:.4f} | production {S['top5_p']:.4f} "
          f"(rows there: {S['rows5_r']:.4f} / {S['rows5_p']:.4f})")
    print(f"non-last base rate: random {S['base_r']:.5f} | production {S['base_p']:.5f}")
    print(f"{'rank':>8} {'rate rand':>10} {'± se':>8} {'rate prod':>10} {'± se':>8} {'exam e(c)':>10} {'± se':>7} {'prod/rand rel':>13} {'± se':>7}   (e and rel: mean over M bins with support)")
    rr_ = np.array([b["rate_r"] for b in BS]); rp_ = np.array([b["rate_p"] for b in BS])
    eE = np.array([np.nanmean(b["E"], 0) for b in BS]); eR = np.array([np.nanmean(b["REL"], 0) for b in BS])
    for c in range(NC):
        print(f"{CELL_NAMES[c]:>8} {S['rate_r'][c]:10.5f} {rr_[:, c].std(ddof=1):8.5f} {S['rate_p'][c]:10.5f} {rp_[:, c].std(ddof=1):8.5f} "
              f"{np.nanmean(S['E'][:, c]):10.3f} {eE[:, c].std(ddof=1):7.3f} {np.nanmean(S['REL'][:, c]):13.3f} {eR[:, c].std(ddof=1):7.3f}")
    print("per M bin: relocation metrics | e(rank 0) | rel(rank 0)")
    for b, (a, z) in enumerate(MBINS):
        Sb = stats((rand[0][rand[3] == b], rand[1][rand[3] == b], rand[2][rand[3] == b], rand[3][rand[3] == b]),
                   (prod[0][prod[3] == b], prod[1][prod[3] == b], prod[2][prod[3] == b], prod[3][prod[3] == b]))
        print(f"  M {a}-{z}: n_rand {int((rand[3] == b).sum())} n_prod {int((prod[3] == b).sum())} | excess {Sb['excess']:.4f} missing {Sb['missing']:.4f} tv {Sb['tv']:.4f} | "
              f"{S['E'][b, 0]:.3f} | {S['REL'][b, 0]:.3f}")
# relevance confound: production's top slots hold more relevant items; log split of the rank-0 lift
S = stats(data["random p>0.9"], prod)
lift_p = S["rate_p"][0] / S["base_p"]; lift_r = S["rate_r"][0] / S["base_r"]
print(f"\nrank-0 lift over the non-last base rate: production {lift_p:.2f}x = examination {lift_r:.2f}x (random-order curve) "
      f"x relevance {lift_p / lift_r:.2f}x; log share of the lift due to position {np.log(lift_r) / np.log(lift_p):.3f}")
print(f"done {time.time() - t0:.0f}s")
