"""P2 read: nested value of the cross-user scored-period activity columns (p2_build.py) on top of the best file.

Arms (each column scaled by its global sd; cold = item never a train candidate; lm = log M centred):
  E    as-of exact-item activity of other users (earlier val/test histories and lists): log1p + indicator, x cold   (legal)
  F5B  backward f5 dynamics: f5 now minus f5 at the latest earlier appearance, its flag, recency                    (legal)
  L    later other-user histories and lists (future; legal under the same-user rule, not available equally on test)
  L4   the same within 96 h (horizon of public rows)
  F5F  forward f5 echo: f5 at the next later appearance minus f5 now                                           (LEAK CLASS)
Every arm is also read with its columns permuted jointly inside each feed (placebo, same per-feed distribution).
Then the label-free test coverage of each family, public vs private.
"""
import sys, json
sys.path.insert(0, "<repo>/lab/nn_x/w12/leader/data_channels")
from dc_lib import *  # noqa
from dc_nest import Nested, fmt

NAMES = json.load(open(f"{OUT}/out/p2_names.json"))
v = Split("val", hist=False)
s = v.best()
X = np.load(f"{OUT}/out/p2_val.npy"); col = {n: X[:, j].astype(np.float64) for j, n in enumerate(NAMES)}
f5 = np.load(f"{FLAT}/val/cand_f5.npy").astype(np.float64)
cold = (col["n_tr"] == 0).astype(float)
lm = np.log(v.M[v.fid]) - np.log(v.M[v.fid]).mean()


def cnt(n):
    c = col[n]; return [np.log1p(c), (c > 0).astype(float)]


def arm_E():
    out = []
    for n in ("HE_all", "HE_fc", "HE_rec", "HE_ct", "CE"):
        out += cnt(n)
    out += [np.log1p(col["HE_all"]) * cold, np.log1p(col["CE"]) * cold, np.log1p(col["HE_all"]) * lm]
    return out


def arm_F5B():
    hp = np.isfinite(col["f5_prev"]).astype(float)
    d = np.nan_to_num(f5 - col["f5_prev"]); rec = np.nan_to_num(col["dt_prev"], nan=1e4)
    return [d, hp, d * (rec < 24), np.log1p(np.minimum(rec, 1e4)) * hp, d * cold]


def arm_L(pref="HL", ex=("all", "fc", "rec", "ct")):
    out = []
    for e in ex:
        out += cnt(f"{pref}_{e}")
    out += [np.log1p(col[f"{pref}_all"]) * cold, np.log1p(col[f"{pref}_all"]) * lm]
    if pref == "HL":
        out += cnt("CL") + [np.log1p(col["CL"]) * cold]
    return out


def arm_F5F():
    hn = np.isfinite(col["f5_next"]).astype(float)
    d = np.nan_to_num(col["f5_next"] - f5); rec = np.nan_to_num(col["dt_next"], nan=1e4)
    return [d, hn, (d > 0).astype(float), (d < 0).astype(float), d * (rec < 24), d * cold]


def cover(cols):
    nz = np.zeros(v.N, bool)
    for c in cols:
        nz |= np.nan_to_num(c) != 0
    return nz


rng = np.random.default_rng(11)
perm = np.lexsort((rng.random(v.N), v.fid))           # one joint within-feed permutation (placebo)
N = Nested(v, s)
log("base", v.ndcg(s).mean())
arms = {
    "E (as-of, legal)": (arm_E(), ["HE_all", "CE"]),
    "F5B (backward f5, legal)": (arm_F5B(), None),
    "E+F5B (all legal as-of)": (arm_E() + arm_F5B(), ["HE_all", "CE"]),
    "L (later other users, future)": (arm_L(), ["HL_all", "CL"]),
    "L4 (later within 96 h)": (arm_L("HL4", ("all", "fc")), ["HL4_all"]),
    "F5F (forward f5 echo, LEAK CLASS)": (arm_F5F(), None),
    "L+F5F (all future)": (arm_L() + arm_F5F(), ["HL_all", "CL"]),
    "ALL": (arm_E() + arm_F5B() + arm_L() + arm_F5F(), ["HE_all", "CE", "HL_all", "CL"]),
}
res = []
for name, (cols, cov) in arms.items():
    covr = cover([col[c] for c in cov]) if cov else cover([cols[1]])
    covf = np.bincount(v.fid, weights=covr, minlength=v.n) > 0
    r = N.read(name, cols, mask_feeds=covf); r["row_cov"] = float(covr.mean())
    print(fmt(r), f"| row cov {r['row_cov']:.3f}", flush=True)
    rp = N.read(name + " [placebo]", [c[perm] for c in cols], insample=False)
    print(fmt(rp), flush=True)
    r["placebo"] = {k: rp[k] for k in ("dndcg", "se", "dnats", "dnats_se")}
    res.append(r)
json.dump(res, open(f"{OUT}/out/p2_read.json", "w"), indent=1, default=float)

# label-free coverage on test, public vs private
T = np.load(f"{OUT}/out/p2_test.npy"); tcol = {n: T[:, j] for j, n in enumerate(NAMES)}
tf = pl.read_parquet(f"{FLAT}/test/feeds.parquet"); tfid = np.load(f"{FLAT}/test/cand_fidx.npy")
pub = (tf["timestamp"].to_numpy() < BOUND)[tfid]
print("\nrow coverage (share of candidate rows with a non-zero value)       val    public  private")
for n in NAMES:
    if n in ("f5_prev", "dt_prev", "f5_next", "dt_next", "n_tr"):
        a, b_, c_ = np.isfinite(col[n]) & (col[n] != 0) if n != "n_tr" else col[n] > 0, None, None
        a = (np.nan_to_num(col[n], nan=0) != 0).mean(); b_ = (np.nan_to_num(tcol[n][pub], nan=0) != 0).mean(); c_ = (np.nan_to_num(tcol[n][~pub], nan=0) != 0).mean()
    else:
        a = (col[n] > 0).mean(); b_ = (tcol[n][pub] > 0).mean(); c_ = (tcol[n][~pub] > 0).mean()
    print(f"  {n:8s}  {a:.4f}  {b_:.4f}  {c_:.4f}")
log("done")
