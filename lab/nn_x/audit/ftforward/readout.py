"""Critique item 5 readout (CPU): forward / reverse transfer of the val fine-tune branch, pre-declared in plan.txt.

Forward: FT trained on Tue only, read on Thu feeds.   Reverse: FT trained on Thu only, read on Tue feeds.
Primary: D = NDCG[0.5 z(FT_day) + 0.5 z(TO)] - NDCG[0.5 z(FT_oof) + 0.5 z(TO)] on the target day (paired per-feed SE),
per member and for the 2-member pool (per-branch z-mean over members, then 0.5 z(S_ft) + 0.5 z(S_to)).
"""
import os
import sys

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
sys.path.insert(0, "<repo>/src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

HERE = "<repo>/nn_x/audit/ftforward"
P = "<repo>/preds"
FLAT = "<repo>/data/flat/val"
MEMBERS = ["r6_cd_m02_r101", "r6_cd_m02_r103"]
SEEDS = [0, 1, 2]

f = pl.read_parquet(f"{FLAT}/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{FLAT}/cand_target.npy")
n = len(M)
fid = np.repeat(np.arange(n), M)
ts = f["timestamp"].to_numpy().astype(np.int64)
TUE = (ts + 10800) // 86400 < 20720
THU = ~TUE
small = M < 50
out_lines = []


def say(s=""):
    print(s, flush=True)
    out_lines.append(s)


def z(s):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 1e-12))
    return (s - mu[fid]) / sd[fid]


def nd(s):
    return ndcg_per_feed(s.astype(np.float32), y, off, M)


def blend(ft, to):
    return 0.5 * z(ft) + 0.5 * z(to)


def fmt(d, mask):
    d = d[mask]
    return f"{d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}"


def row(label, d, mask):
    return (f"{label:46s} {fmt(d, mask):20s} | M<50 {fmt(d, mask & small):20s} | M>=50 {fmt(d, mask & ~small):20s}"
            f" | n {int(mask.sum())}")


def load(v, sname, s):
    return np.load(f"{HERE}/preds/nn_{v}_ft{sname}_s{s}_val.npy").astype(np.float64)


TO = {v: np.load(f"{P}/nn_{v}_trainonly_val.npy").astype(np.float64) for v in MEMBERS}
OOF = {v: np.load(f"{P}/nn_{v}_n2cpv3_val_oof.npy").astype(np.float64) for v in MEMBERS}
ND = {}  # cache of per-feed ndcg


def ndc(key, s=None):
    if key not in ND:
        ND[key] = nd(s)
    return ND[key]


def pool(d):
    return np.mean([z(d[v]) for v in MEMBERS], 0)


for direction, train_set, train_mask, target_mask, tname in (("FORWARD", "tue", TUE, THU, "Thu"),
                                                             ("REVERSE", "thu", THU, TUE, "Tue")):
    say("=" * 140)
    say(f"{direction}: FT trained on {train_set.upper()} feeds only (n={int(train_mask.sum())}), read on {tname} feeds "
        f"(n={int(target_mask.sum())}); paired per-feed NDCG deltas, mean ± SE")
    say("=" * 140)
    units = {v: dict(to=TO[v], oof=OOF[v], **{f"s{s}": load(v, train_set, s) for s in SEEDS}) for v in MEMBERS}
    units["POOL"] = {k: pool({v: units[v][k] for v in MEMBERS}) for k in ["to", "oof"] + [f"s{s}" for s in SEEDS]}
    for v in MEMBERS:
        units[v]["s_mean"] = np.mean([z(units[v][f"s{s}"]) for s in SEEDS], 0)
    units["POOL"]["s_mean"] = pool({v: units[v]["s_mean"] for v in MEMBERS})
    D_primary = {}
    for name, u in units.items():
        say(f"--- {name}")
        n_to, n_oof = ndc((name, "to"), u["to"]), ndc((name, "oof"), u["oof"])
        n_b_oof = ndc((name, "b_oof"), blend(u["oof"], u["to"]))
        say(f"{'levels on ' + tname + ': TO / FT_oof / blend_oof':46s} {n_to[target_mask].mean():.5f} / "
            f"{n_oof[target_mask].mean():.5f} / {n_b_oof[target_mask].mean():.5f}")
        say(row(f"FT_oof - TO  ({tname}, production same-day OOF)", n_oof - n_to, target_mask))
        say(row(f"blend(FT_oof) - TO  ({tname})", n_b_oof - n_to, target_mask))
        for s in SEEDS:
            ft = u[f"s{s}"]
            n_ft, n_b = ndc((name, train_set, s), ft), ndc((name, "b", train_set, s), blend(ft, u["to"]))
            tag = "PRIMARY seed 0" if s == 0 else f"seed {s}"
            say(row(f"[{tag}] D = blend(FT_{train_set}) - blend(FT_oof)  ({tname})", n_b - n_b_oof, target_mask))
            say(row(f"[{tag}] FT_{train_set} - TO alone  ({tname})", n_ft - n_to, target_mask))
            say(row(f"[{tag}] blend(FT_{train_set}) - TO  ({tname})", n_b - n_to, target_mask))
            say(row(f"[{tag}] in-sample FT_{train_set} - TO  ({train_set})", n_ft - n_to, train_mask))
            say(row(f"[{tag}] in-sample FT_{train_set} - FT_oof  ({train_set})", n_ft - n_oof, train_mask))
            if s == 0:
                D_primary[name] = (n_b - n_b_oof)
        ds = np.array([(ndc((name, "b", train_set, s)) - n_b_oof)[target_mask].mean() for s in SEEDS])
        say(f"{'D over seeds 0,1,2: mean / sd':46s} {ds.mean():+.5f} / {ds.std(ddof=1):.5f}   (per seed "
            + ", ".join(f"{d:+.5f}" for d in ds) + ")")
        n_bm = ndc((name, "b_mean", train_set), blend(u["s_mean"], u["to"]))
        say(row(f"3-seed z-mean FT_{train_set}: D vs blend(FT_oof)  ({tname})", n_bm - n_b_oof, target_mask))
        say(row(f"3-seed z-mean FT_{train_set}: blend - TO  ({tname})", n_bm - n_to, target_mask))
    d = D_primary["POOL"][target_mask]
    verdict = "KILL (D < -0.002: the FT branch does NOT transfer forward)" if d.mean() < -0.002 else \
        "PASS (D >= -0.002: no evidence that the FT branch fails to transfer)"
    say(f">>> {direction} primary readout, POOL, seed 0, {tname}: D = {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}"
        f" -> {verdict}" + ("  [kill rule applies to FORWARD only]" if direction == "REVERSE" else ""))
    # ---- secondary control: size-matched random user-grouped mix (seed 0), read on held-out target-day feeds
    try:
        mix = {v: load(v, f"mix_{train_set}", 0) for v in MEMBERS}
        tm = np.load(f"{HERE}/preds/nn_{MEMBERS[0]}_ftmix_{train_set}_s0_trainmask.npy")
        for v in MEMBERS[1:]:
            assert np.array_equal(tm, np.load(f"{HERE}/preds/nn_{v}_ftmix_{train_set}_s0_trainmask.npy"))
        ho = target_mask & ~tm
        say(f"--- CONTROL C1 ({direction}): FT on a random user-grouped mix of {int(tm.sum())} feeds "
            f"({int((tm & train_mask).sum())} {train_set}, {int((tm & target_mask).sum())} {tname.lower()}), seed 0; "
            f"read on the {int(ho.sum())} held-out {tname} feeds")
        mix["POOL"] = pool(mix)
        for name in MEMBERS + ["POOL"]:
            u = units[name]
            n_b_oof = ndc((name, "b_oof"))
            n_b_fwd = ndc((name, "b", train_set, 0))
            n_b_mix = nd(blend(mix[name], u["to"]))
            n_to = ndc((name, "to"))
            say(row(f"{name}: blend(C1 mix) - blend(FT_{train_set})  ({tname} held-out)", n_b_mix - n_b_fwd, ho))
            say(row(f"{name}: blend(C1 mix) - blend(FT_oof)  ({tname} held-out)", n_b_mix - n_b_oof, ho))
            say(row(f"{name}: blend(FT_{train_set}) - blend(FT_oof)  ({tname} held-out)", n_b_fwd - n_b_oof, ho))
            say(row(f"{name}: C1 mix - TO alone  ({tname} held-out)", nd(mix[name]) - n_to, ho))
    except FileNotFoundError as e:
        say(f"control C1 not available: {e}")
    say()

open(f"{HERE}/results.txt", "w").write("\n".join(out_lines) + "\n")
print(f"written {HERE}/results.txt")
