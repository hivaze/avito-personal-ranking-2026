"""Part 4: winner's curse / selection optimism of (a) the m02 arm gate and (b) the val-best pool choice (decide3/decide4).
User-grouped bootstrap (user 0 = singleton groups) and split-half re-selection on val. Also the public-noise scale.
Writes p4_winners.txt."""
from lib import *
from scipy import stats as st
from scipy.optimize import minimize_scalar

C = OUT / "cache"
idx = {m: i for i, m in enumerate(MEMBERS)}
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))
V = Split("val")
Zv = {h: np.load(C / f"Z_val_{h}.npy", mmap_mode="r") for h in ("to", "ft")}
g = user_groups(V.users)
G = g.max() + 1
P_(f"val feeds {V.n}, user groups {G} (user 0 -> singletons)")

CAND = {  # every pool whose val number was looked at in decide3 (10:56) and decide4-interim (13:50)
    "Cd": ["F5cd", "F6", "F7"], "Pall": ["F5cd", "F6", "F7", "F9", "F10", "F12"],
    "Pall+F7x": ["F5cd", "F6", "F7", "F9", "F10", "F12", "F7x"], "PF7x": ["F5cd", "F6", "F7x", "F9", "F10", "F12"],
    "F7x": ["F7x"], "PF7x+F7x70": ["F5cd", "F6", "F7x", "F9", "F10", "F12", "F7x70"],
    "G3": ["F10", "F7x", "F7x70"], "G-70": ["F10", "F7x"], "M2": ["F7x", "F7x70"],
}
SETS = {"decide4 (val-best of 6)": ["PF7x", "Pall", "PF7x+F7x70", "G3", "G-70", "M2"],
        "decide3+decide4 union (8)": ["Pall", "Pall+F7x", "PF7x", "F7x", "PF7x+F7x70", "G3", "G-70", "M2"]}
ND = {}
for c, fams in CAND.items():
    ms = [m for f in fams for m in FAM[f]]; ii = [idx[m] for m in ms]
    to = V.z(np.asarray(Zv["to"][ii], np.float64).mean(0)); ft = V.z(np.asarray(Zv["ft"][ii], np.float64).mean(0))
    ND[c] = V.nd(0.5 * to + 0.5 * ft)
names = list(CAND)
X = np.stack([ND[c] for c in names])  # cand x feeds
P_("observed val (gain vs Cd +- paired SE): " + ", ".join(f"{c} {ND[c].mean():.5f} ({(ND[c] - ND['Cd']).mean():+.5f}+-{se(ND[c] - ND['Cd']):.5f})" for c in names))

# group sums for fast bootstrap
S = np.zeros((len(names), G)); np.add.at(S.T, g, X.T)
cnt = np.bincount(g, minlength=G).astype(np.float64)
rng = np.random.default_rng(7)
B = 4000
W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(np.float64)  # bootstrap group weights
XB = (W @ S.T) / (W @ cnt)[:, None]  # B x cand
X0 = X.mean(1)
ic = names.index("Cd")
for sname, cs in SETS.items():
    ii = [names.index(c) for c in cs]
    sel_full = cs[int(np.argmax(X0[ii]))]
    # (1) Efron bootstrap: treat full-sample means as truth
    selb = np.array(ii)[np.argmax(XB[:, ii], 1)]
    opt = ((XB[np.arange(B), selb] - X0[selb]) - (XB[:, ic] - X0[ic])).mean()
    # (2) null version: all candidates' true gains equal (centre each candidate) -> upper bound
    XBc = XB - X0[None, :]
    selc = np.array(ii)[np.argmax(XBc[:, ii], 1)]
    opt0 = (XBc[np.arange(B), selc] - XBc[:, ic]).mean()
    freq = {c: (selb == names.index(c)).mean() for c in cs}
    P_(f"\n== {sname}: full-sample val-best = {sel_full}")
    P_(f"  bootstrap selection frequencies: " + ", ".join(f"{c} {f:.2f}" for c, f in freq.items()))
    P_(f"  optimism of the selected pool's gain vs Cd: bootstrap (truth = observed) {opt:+.5f}; null (all equal) {opt0:+.5f}")
    # (3) split-half: select on half A, evaluate on half B (user-grouped); scale by 1/sqrt(2) for full-sample noise
    oa, ob = [], []
    for r in range(2000):
        half = rng.random(G) < 0.5
        mA, mB = half[g], ~half[g]
        xa, xb = X[:, mA].mean(1), X[:, mB].mean(1)
        k = ii[int(np.argmax(xa[ii]))]
        oa.append(xa[k] - xa[ic]); ob.append(xb[k] - xb[ic])
    oa, ob = np.array(oa), np.array(ob)
    P_(f"  split-half: gain of the half-A winner vs Cd on A {oa.mean():+.5f}, on B {ob.mean():+.5f}; optimism {oa.mean() - ob.mean():+.5f}"
       f" (half-sample) -> ~{(oa.mean() - ob.mean()) / np.sqrt(2):+.5f} at full-sample noise")
    if "M2" in cs:
        k = names.index("M2")
        P_(f"  M2 observed gain vs Cd {X0[k] - X0[ic]:+.5f}; minus optimism -> {X0[k] - X0[ic] - opt:+.5f} (bootstrap) .. {X0[k] - X0[ic] - opt0:+.5f} (null)")

# (a) arm-level: m02 passed stage 2 with member val +0.00117 +- 0.00039 vs c0 (gate: > 2 SE). Truncated-normal correction.
P_("\n== (a) m02 arm gate (CURRENT_STATUS 10:55: member val +0.00117 +- 0.00039 vs c0, gate val > 2 SE; holdout +0.00134 +- 0.00022)")
x, s = 0.00117, 0.00039; c = 2 * s
nll = lambda mu: -(st.norm.logpdf(x, mu, s) - st.norm.logsf(c, mu, s))
mle = minimize_scalar(nll, bounds=(-0.002, 0.004), method="bounded").x
for mu in (0.0005, 0.0008, 0.00117, 0.00134):
    a = (c - mu) / s
    P_(f"  if true val effect = {mu:.5f}: E[observed | passed gate] - true = {s * st.norm.pdf(a) / st.norm.sf(a):+.5f}")
P_(f"  conditional MLE of the true val effect given the gate: {mle:+.5f} (observed {x:+.5f}; shrink {x - mle:+.5f})")
P_("  The holdout effect of the same arm (+0.00134, z 6) was not gated on val, so the arm itself is real; the gate"
   " optimism on val is ~1e-4.")

# arms screened in the same program (val deltas from CURRENT_STATUS / R6/R7 reports) -> empirical-Bayes prior width
P_("\n== Public-noise scale of the shortfall (context only; nothing fitted on public)")
T = Split("test"); npub = int((T.ts < PUB_TS).sum())
for p in ("PF7x", "G3", "M2"):
    d = ND[p] - ND["Cd"]
    s_v, s_p = se(d), d.std() / np.sqrt(npub)
    gap = d.mean() - (PUBLIC[p] - PUBLIC["Cd"])
    P_(f"  {p:5s} val gain {d.mean():+.5f}+-{s_v:.5f} | public gain {PUBLIC[p] - PUBLIC['Cd']:+.5f}+-{s_p:.5f} (SE from the val per-feed SD,"
       f" n_pub {npub}) | val - public {gap:+.5f}+-{np.hypot(s_v, s_p):.5f} (z {gap / np.hypot(s_v, s_p):.2f})")
(OUT / "p4_winners.txt").write_text("\n".join(lines) + "\n")
