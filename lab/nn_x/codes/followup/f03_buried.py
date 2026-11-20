"""Follow-up (3) BURIED-CELL DIAGNOSTIC (information, no gate; PREDECLARED.txt 05:47:47 UTC).
Cell = stage0_gate.py 'right category, buried' (zG3 = 0.5 rz(ft) + 0.5 rz(to); positive in G3's LSE-top category with >= 3
same-category candidates above it), leak-free feeds. Lost NDCG = exact per-positive split of 1-NDCG (tail.py §4) under zG3.
Decomposition of the cell loss and cue win rates on (cell positive, same-category beater) pairs by: delivery class (C1),
f2 class (home / other non-zero / zero), vertical (11), positive's f4 tercile (cuts over all val positives).
Flags: CONCENTRATION = class holds > 1.5x its row share of the cell loss and >= 3% of the cell loss;
       CUE = (class, cue) win rate > 0.55 with >= 500 pairs and lower 2-SE bound > 0.5.
Output: out/f03_buried.log, out/f03_buried.json."""
import sys, json, time
sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *
from qt import leak_mask

T0 = time.time()
LOGF = open(OUT / "f03_buried.log", "w")


def log(s=""):
    print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


V = Split("val")
keep = ~leak_mask()
B = buried_cell(V, keep)
zG3, cellpos, pi, ni, cat = B["zG3"], B["cellpos"], B["pi"], B["ni"], B["cat"]
fid, y, n, M, off = V.fid, V.y, V.n, V.M, V.off
nd = V.nd(zG3)
log(f"zG3 val NDCG {nd.mean():.5f}; leak-free feeds {keep.sum()}; NDCG leak-free {nd[keep].mean():.5f}")
log(f"cell positives: all feeds {int(cellpos.sum()):,}; leak-free {int(cellpos[keep[fid]].sum()):,}; pairs (leak-free) {len(pi):,}")

# exact per-positive loss split (tail.py §4) under zG3
order = np.lexsort((y, -zG3, fid))
rank = np.empty(V.N, np.int64); rank[order] = np.arange(V.N) - np.repeat(off, M)
disc = lambda r: 1.0 / np.log2(np.asarray(r, np.float64) + 2.0)
npos = np.bincount(fid, y, n).astype(np.int64)
cumd = np.concatenate([[0.0], np.cumsum(disc(np.arange(int(M.max()))))])
idcg = cumd[npos]
Pp = np.where(y > 0)[0]; po = Pp[np.lexsort((rank[Pp], fid[Pp]))]
starts = np.r_[0, np.cumsum(npos[npos > 0])[:-1]]
jj = np.arange(len(po)) - np.repeat(starts, npos[npos > 0])
contrib = np.zeros(V.N); contrib[po] = (disc(jj) - disc(rank[po])) / idcg[fid[po]]
assert np.allclose(np.bincount(fid, contrib, n), 1 - nd)
LK = keep[fid]
loss_all = contrib[LK].sum()
cmask = cellpos & LK
loss_cell = contrib[cmask].sum()
log(f"loss (sum 1-NDCG over leak-free feeds) {loss_all:.1f}; cell loss {loss_cell:.1f} = {loss_cell/loss_all:.4f} of all loss "
    f"(stage0 quoted ~0.37 via 05_mass buckets)")

# classes
HARD = [2, 9, 10, 11, 12, 25, 26, 30, 31, 48, 49, 50, 51, 52, 53]; NEAR = [27, 34, 35, 36, 38]; VLOW = [19, 37, 5, 47]; LOW = [4, 24]
dcls = np.full(64, 4)
for c in HARD: dcls[c] = 0
for c in NEAR: dcls[c] = 1
for c in VLOW: dcls[c] = 2
for c in LOW: dcls[c] = 3
DN = ["hard-zero", "near-zero", "very-low", "low", "goods"]
VERT = json.load(open(ROOT / "nn_x/codes/categories/out/code_vertical.json"))
vnames = sorted(set(VERT.values())); vid = np.full(64, len(vnames))
for c, v in VERT.items(): vid[int(c)] = vnames.index(v)
VN = vnames + ["unknown"]
f2 = np.load(FLAT / "val/cand_f2.npy").astype(np.float64)
home = np.load(OUT / "val_home_f2.npy").astype(np.float64)
f2c = np.where(f2 == 0, 2, np.where(f2 == home[fid], 0, 1))
F2N = ["home (f2 = feed mode)", "other non-zero", "zero"]
f4 = np.load(FLAT / "val/cand_f4.npy").astype(np.float64)
q4 = np.quantile(f4[y > 0], [1 / 3, 2 / 3])
f4c = np.searchsorted(q4, f4, side="right")
F4N = [f"f4 T1 (<= {q4[0]:.3f}, newest)", f"f4 T2 ({q4[0]:.3f}-{q4[1]:.3f})", f"f4 T3 (> {q4[1]:.3f}, oldest)"]
log(f"f4 tercile cuts over all val positives: {q4.round(4).tolist()} (f4 = listing age; higher = older is the report's reading)")
DIMS = {"delivery": (dcls[cat], DN), "f2 class": (f2c, F2N), "vertical": (vid[cat], VN), "f4 tercile": (f4c, F4N)}

# cues
pop = np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64), nan=0.0)
CUES = {"pop": pop}
for k in ["f1", "f2", "f3", "f4", "f5"]:
    CUES[k] = np.load(FLAT / f"val/cand_{k}.npy").astype(np.float64)
for k in ["maxsim32", "cos_hall", "cos_u"]:
    CUES[k] = np.load(FEATS / f"val/{k}.npy").astype(np.float64)
for k in ["s4_link", "s4_sym", "s4_link_any", "rp_lift", "rp_cnt"]:
    p_ = OUT / f"val_{k}.npy"
    if p_.exists():
        CUES[k] = np.load(p_).astype(np.float64)
CUES["-pop (anti)"] = -pop
CUES["-f4 (newer)"] = -CUES["f4"]
RES = {"cell_loss_share_of_all": float(loss_cell / loss_all), "n_cell_pos_leakfree": int(cmask.sum()), "n_pairs": int(len(pi)), "dims": {}}

log("\n=== cue win rates on ALL cell pairs (P(cue_pos > cue_beater), ties 0.5, user-clustered SE) ===")
allw = {}
for k, x in CUES.items():
    r, se, npair = pair_win(V, x, pi, ni)
    allw[k] = (r, se)
    log(f"  {k:12s} {r:.4f} +- {se:.4f} (pairs {npair:,})")
RES["cue_all"] = {k: list(map(float, v)) for k, v in allw.items()}

flags = []
for dim, (cls, names) in DIMS.items():
    log(f"\n=== by {dim} (of the POSITIVE) ===")
    log(f"  {'class':34s} | cell loss share | row share | pos share | loss/row | loss/pos-share | cell pos | loss per cell pos | pairs | "
        "top-3 cues by win rate (win +- SE)")
    rows = LK
    RES["dims"][dim] = {}
    for c in range(len(names)):
        lm = cmask & (cls == c)
        if lm.sum() == 0 and np.sum(rows & (cls == c)) == 0:
            continue
        ls = contrib[lm].sum() / loss_cell
        rs = np.mean(cls[rows] == c)
        ps = np.sum((cls == c) & (y > 0) & rows) / np.sum((y > 0) & rows)
        lpp = contrib[lm].sum() / max(lm.sum(), 1)
        pm = cls[pi] == c
        cw = {}
        for k, x in CUES.items():
            if pm.sum() == 0:
                continue
            r, se, npair = pair_win(V, x, pi, ni, pm)
            cw[k] = (r, se, npair)
            if r > 0.55 and npair >= 500 and r - 2 * se > 0.5:
                flags.append(f"CUE: {dim}={names[c]}: {k} win {r:.4f} +- {se:.4f} ({npair} pairs)")
        top = sorted(cw.items(), key=lambda kv: -kv[1][0])[:3]
        conc = ls > 1.5 * rs and ls >= 0.03
        if conc:
            flags.append(f"CONCENTRATION: {dim}={names[c]}: cell loss share {ls:.3f} vs row share {rs:.3f} (x{ls/max(rs,1e-9):.2f})")
        log(f"  {names[c]:34s} | {ls:.3f} | {rs:.3f} | {ps:.3f} | {ls/max(rs,1e-9):5.2f}{' *' if conc else '  '} | {ls/max(ps,1e-9):5.2f} | "
            f"{int(lm.sum()):5d} | {lpp:.3f} | {int(pm.sum()):6d} | " + "; ".join(f"{k} {v[0]:.3f}+-{v[1]:.3f}" for k, v in top))
        RES["dims"][dim][names[c]] = dict(loss_share=float(ls), row_share=float(rs), pos_share=float(ps), n_cell_pos=int(lm.sum()),
                                          loss_per_pos=float(lpp), pairs=int(pm.sum()),
                                          cues={k: [float(v[0]), float(v[1]), int(v[2])] for k, v in cw.items()})
# share of the cell vs the share of ALL loss per class (is the cell's composition special vs the rest of the loss?)
log("\n=== cell loss share vs share of ALL val loss (leak-free) per class ===")
for dim, (cls, names) in DIMS.items():
    s = "; ".join(f"{names[c]} {contrib[cmask & (cls == c)].sum()/loss_cell:.3f}/{contrib[LK & (cls == c)].sum()/loss_all:.3f}"
                  for c in range(len(names)) if np.any(LK & (cls == c)))
    log(f"  {dim}: cell/all-loss share: {s}")
log("\nFLAGS:")
for fl in flags or ["none"]:
    log("  " + fl)
RES["flags"] = flags
json.dump(RES, open(OUT / "f03_buried.json", "w"), indent=1, default=float)
log(f"done in {time.time()-T0:.0f}s")
