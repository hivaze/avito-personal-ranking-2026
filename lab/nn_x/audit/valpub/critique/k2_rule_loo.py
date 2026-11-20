"""Leave-one-out test of the pre-registered selection rule of reports/VAL_PUBLIC_1113.md section 3 on the 7 public-scored pools:
hold one pool out, apply each rule to the remaining pools, and ask whether it picks the public-best of them. Also pairwise sign
agreement of the lenses with public among the 5 pools that have a train-only (holdout) form, and the rule's own thresholds
evaluated on the scored contrasts. CPU only; own code; writes out/k2_rule_loo.log."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys, itertools
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; P = ROOT / "preds"
HERE = ROOT / "nn_x/audit/valpub/critique"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True, parents=True)
BOUND = 1790497920; N_PUB = 8562
PUB = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473, "TOonly": 0.392346}
ORDER = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
ELIG = ["Cd", "Pall", "PF7x", "G3", "M2"]            # pools with a train-only form (rule R4: stackers / w_ft variants ineligible)
S2_HOLD_REC = -0.00006                                # recorded OOS stacker holdout reading (CURRENT_STATUS 16:58)
FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84", "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
}
HPOOLS = {"Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
          "Pall": FAM["F5cd"] + FAM["F6"] + FAM["F7"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
          "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
          "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"], "M2": FAM["F7x"] + FAM["F7x70"]}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def zfeed(s, M):
    s = np.asarray(s, np.float64); n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


fv = pl.read_parquet(FLAT / "val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); offv = fv["cand_off"].to_numpy().astype(np.int64)
tsv = fv["timestamp"].to_numpy().astype(np.int64); yv = np.load(FLAT / "val/cand_target.npy")
ft = pl.read_parquet(FLAT / "test/feeds.parquet"); Mt = ft["M"].to_numpy().astype(np.int64); tst = ft["timestamp"].to_numpy().astype(np.int64)
pubmask = tst < BOUND; privmask = ~pubmask
dv = pl.read_parquet(ROOT / "nn_x/pubdiag/shift/desc_val.parquet"); dt = pl.read_parquet(ROOT / "nn_x/pubdiag/shift/desc_test.parquet")
cold_v = dv["cold_share"].to_numpy(); cold_t = dt["cold_share"].to_numpy()


def final_val(name):
    if name == "S2":
        return np.load(P / "nnstk_g3gate3r_val.npy").astype(np.float64)
    if name == "TOonly":
        return np.load(P / "ens_G3_toonly_val.npy").astype(np.float64)
    if name == "G3":
        return np.load(P / "ens_G3_final_val.npy").astype(np.float64)
    return 0.5 * zfeed(np.load(P / f"ens_{name}_ft_val.npy"), Mv) + 0.5 * zfeed(np.load(P / f"ens_{name}_to_val.npy"), Mv)


def pf_val(name):
    c = OUT / f"k2_pfval_{name}.npy"
    if c.exists():
        return np.load(c)
    x = ndcg_per_feed(final_val(name).astype(np.float32), yv, offv, Mv); np.save(c, x); return x


Mh, offh = np.load(P / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); offh = offh.astype(np.int64)
yh = np.load(P / "holdout/holdout_y.npy")


def pf_hold(name):
    c = OUT / f"k2_pfhold_{name}.npy"
    if c.exists():
        return np.load(c)
    acc = None
    for mm in HPOOLS[name]:
        z = zfeed(np.load(P / f"holdout/nn_{mm}_holdout.npy"), Mh); acc = z if acc is None else acc + z
    x = ndcg_per_feed((acc / len(HPOOLS[name])).astype(np.float32), yh, offh, Mh); np.save(c, x); return x


# ---- weights: public M-mix (calib bins) and private M x cold x hour5 (the rule's V')
EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
bv = np.searchsorted(EDGES, Mv, side="right"); bt = np.searchsorted(EDGES, Mt[pubmask], side="right")
w_pub = (np.bincount(bt, minlength=10) / len(bt) / np.maximum(np.bincount(bv, minlength=10) / len(bv), 1e-12))[bv]; w_pub /= w_pub.mean()


def cells(M, cold, ts):
    mseg = np.searchsorted(np.array([10, 25, 50, 100, 200]), M, side="left")
    cseg = np.searchsorted(np.array([.15, .3, .45]), cold, side="left")
    hr = ((ts + 10800) % 86400) // 3600; h5 = np.digitize(hr, [6, 10, 14, 18])
    return (mseg * 4 + cseg) * 5 + h5


cv, ct = cells(Mv, cold_v, tsv), cells(Mt, cold_t, tst)[privmask]
w_priv = np.zeros(len(cv)); lost = 0.0
for c in np.unique(np.r_[cv, ct]):
    a, b = (cv == c).mean(), (ct == c).mean()
    if a > 0:
        w_priv[cv == c] = b / a
    else:
        lost += b
w_priv /= w_priv.mean()
say(f"K2: RULE LOO. private M x cold x hour5 weights: n_eff {w_priv.sum() ** 2 / (w_priv ** 2).sum():.0f}, private mass without val support {lost:.4f}")

pf = {k: pf_val(k) for k in ORDER}; hp = {k: pf_hold(k) for k in HPOOLS}
V = {k: pf[k].mean() for k in ORDER}
Vp = {k: np.average(pf[k], weights=w_priv) for k in ORDER}
H = {k: hp[k].mean() for k in HPOOLS}; H["TOonly"] = H["G3"]; H["S2"] = H["G3"] + S2_HOLD_REC
SEv = {k: (pf[k] - pf["G3"]).std(ddof=1) / np.sqrt(len(Mv)) for k in ORDER}
SEh = {k: (hp[k] - hp["G3"]).std(ddof=1) / np.sqrt(len(Mh)) for k in HPOOLS}
say("\npool     val       V'(priv)   holdout    public   | d vs G3: val±SE          holdout±SE        public")
for k in ORDER:
    hs = f"{H[k] - H['G3']:+.5f}±{SEh.get(k, 0):.5f}" if k in HPOOLS else ("    0 (same TO)   " if k == "TOonly" else f"{S2_HOLD_REC:+.5f} (rec.)  ")
    say(f"{k:7s} {V[k]:.5f}  {Vp[k]:.5f}  {H[k]:.5f}  {PUB[k]:.5f} | {V[k] - V['G3']:+.5f}±{SEv[k]:.5f}  {hs}  {PUB[k] - PUB['G3']:+.5f}")

# ---- rules (argmax forms; a delta vs any common reference cancels in the argmax)
RULES = {
    "val-best (any pool)": lambda c: max(c, key=lambda k: V[k]),
    "val-best (eligible only)": lambda c: max([k for k in c if k in ELIG], key=lambda k: V[k]),
    "holdout-best (eligible)": lambda c: max([k for k in c if k in ELIG], key=lambda k: H[k]),
    "E = H + 0.15 V' (eligible) [R4]": lambda c: max([k for k in c if k in ELIG], key=lambda k: H[k] + 0.15 * Vp[k]),
    "E with 0.3 V (semantics R3)": lambda c: max([k for k in c if k in ELIG], key=lambda k: H[k] + 0.3 * V[k]),
    "0.5 H + 0.5 V (old decide S)": lambda c: max([k for k in c if k in ELIG], key=lambda k: 0.5 * H[k] + 0.5 * V[k]),
    "S = 0.5 H + 0.5 P (R5; uses public)": lambda c: max([k for k in c if k in ELIG], key=lambda k: 0.5 * H[k] + 0.5 * PUB[k]),
}
say("\n--- LOO: hold one pool out, apply each rule to the other 6, report the pick, its public rank among the 6 (1 = best) and the gap to the public-best of the 6 ---")
hdr = f"{'held out':9s} {'public-best of rest':20s} | " + " | ".join(f"{n[:30]:30s}" for n in RULES)
say(hdr)
summary = {n: [] for n in RULES}
for h in ["(none)"] + ORDER:
    cands = [k for k in ORDER if k != h]
    best = max(cands, key=lambda k: PUB[k])
    row = []
    for n, f in RULES.items():
        pick = f(cands); rank = sorted(cands, key=lambda k: -PUB[k]).index(pick) + 1
        gap = PUB[pick] - PUB[best]; summary[n].append((pick == best, rank, gap))
        row.append(f"{pick:6s} rank {rank}/{len(cands)} gap {gap:+.5f}")
    say(f"{h:9s} {best:20s} | " + " | ".join(f"{c:30s}" for c in row))
say("\nsummary over the 8 folds (none + 7): hits = picked the public-best; mean public rank; mean gap to public-best")
for n, v in summary.items():
    say(f"   {n:36s} hits {sum(x[0] for x in v)}/8  mean rank {np.mean([x[1] for x in v]):.2f}  mean gap {np.mean([x[2] for x in v]):+.5f}")
say("   NB: with G3 present every holdout-led rule picks G3 (holdout-best AND public-best): those 7 folds carry no information about")
say("   the rule. The only informative fold is 'G3 held out', where the rule must choose among Cd/Pall/PF7x/M2.")

# ---- the informative fold in detail
say("\n--- the G3-held-out fold in detail (eligible candidates Cd, Pall, PF7x, M2) ---")
cands = ["Cd", "Pall", "PF7x", "M2"]
for k in cands:
    say(f"   {k:5s} holdout {H[k]:.5f}  E(0.15V') {H[k] + 0.15 * Vp[k]:.5f}  val {V[k]:.5f}  public {PUB[k]:.5f}")
d = hp["M2"] - hp["PF7x"]; dv_ = pf["M2"] - pf["PF7x"]
say(f"   M2 - PF7x: holdout {d.mean():+.5f} ± {d.std(ddof=1) / np.sqrt(len(d)):.5f} (z {d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):+.2f}); val {dv_.mean():+.5f} ± {dv_.std(ddof=1) / np.sqrt(len(dv_)):.5f}; public {PUB['M2'] - PUB['PF7x']:+.5f}")
say("   -> the holdout cannot separate M2 from PF7x (z 0.0); E = H + 0.15 V' breaks the tie toward M2 (val), public went the other way (PF7x by 0.0003).")

# ---- pairwise sign agreement with public among the 5 eligible pools
say("\n--- pairwise sign agreement with public among the 5 pools with a holdout (10 pairs) ---")
Dp = {}
for a, b in itertools.combinations(ELIG, 2):
    dvv = pf[a] - pf[b]; dh = hp[a] - hp[b]
    se_pub = np.sqrt(np.average((dvv - np.average(dvv, weights=w_pub)) ** 2, weights=w_pub) / N_PUB)
    Dp[(a, b)] = dict(val=dvv.mean(), se_val=dvv.std(ddof=1) / np.sqrt(len(dvv)), hold=dh.mean(), se_hold=dh.std(ddof=1) / np.sqrt(len(dh)),
                      pub=PUB[a] - PUB[b], se_pub=se_pub, E=(H[a] + 0.15 * Vp[a]) - (H[b] + 0.15 * Vp[b]))
say(f"{'pair':12s} {'val d±SE':>20s} {'holdout d±SE':>20s} {'public d (SE)':>20s} {'E d':>9s} | sign agrees with public: val hold E | public |z|")
agree = {"val": 0, "hold": 0, "E": 0}; n_resolved = 0
for (a, b), q in Dp.items():
    s_val, s_h, s_E, s_p = np.sign(q["val"]), np.sign(q["hold"]), np.sign(q["E"]), np.sign(q["pub"])
    agree["val"] += s_val == s_p; agree["hold"] += s_h == s_p; agree["E"] += s_E == s_p
    zp = abs(q["pub"]) / q["se_pub"]; n_resolved += zp >= 1
    say(f"{a + '-' + b:12s} {q['val']:+.5f}±{q['se_val']:.5f}  {q['hold']:+.5f}±{q['se_hold']:.5f}  {q['pub']:+.5f} ({q['se_pub']:.5f})  {q['E']:+.5f} |   {'Y' if s_val == s_p else 'n'}   {'Y' if s_h == s_p else 'n'}  {'Y' if s_E == s_p else 'n'}  |  {zp:.1f}")
say(f"   agreements out of 10: val {agree['val']}  holdout {agree['hold']}  E {agree['E']};  pairs where public itself is >= 1 SE from 0: {n_resolved}/10")
say("   (10 correlated pairs of 5 pools; Kendall tau = (agree - disagree)/10; tau 0.6 vs 0.2 on n = 5 is not a significant difference, p ~0.2)")

# ---- the rule's own thresholds on the scored contrasts
say("\n--- the rule's thresholds evaluated on the scored contrasts ---")
for k in ["Cd", "Pall", "PF7x", "M2"]:
    dh = hp[k] - hp["G3"]; dvv = pf[k] - pf["G3"]
    se_pub = np.sqrt(np.average((dvv - np.average(dvv, weights=w_pub)) ** 2, weights=w_pub) / N_PUB); se_h = dh.std(ddof=1) / np.sqrt(len(dh))
    gap = (PUB[k] - PUB["G3"]) - dh.mean(); thr = 2.5 * np.sqrt(se_pub ** 2 + se_h ** 2)
    say(f"   {k:5s} R5 verification |P - H| = {abs(gap):.5f} vs threshold 2.5 sqrt(SE_pub^2+SE_h^2) = {thr:.5f}  ({abs(gap) / thr:.2f} of threshold)")
say("   -> the verification clause can only fire on a |P - H| >= 0.0013-0.0017, i.e. on a difference 2-3x larger than any near-copy")
say("      difference seen so far; it does not constrain the holdout lens at the scale that matters (0.0003-0.0007).")
say(f"   R4 switch bar E_C > +2 SE_E with SE_E ~ SE_h = {np.mean([SEh[k] for k in HPOOLS if k != 'G3']):.5f}: a near-copy needs H_C > ~{2 * np.mean([SEh[k] for k in HPOOLS if k != 'G3']):.5f} on the holdout;")
say("      the best challengers so far are U1 -0.00003, U2 +0.00002, U3 -0.00005 (decide8): the bar is ~10x the observed spread, so the")
say("      rule is 'G3 stays' for every near-copy that can be built from this member type; it is not testable by near-copies.")

(OUT / "k2_rule_loo.log").write_text("\n".join(LOG) + "\n")
print("written", OUT / "k2_rule_loo.log")
