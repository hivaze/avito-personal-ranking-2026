"""Test the label-free edit rules proposed by the study agents, in code, on labelled validation feeds.

Sets: 'dev' = all validation feeds except the 722-feed eval slice and the 100 study feeds;
'eval' = the 722-feed slice. Each rule is a function of the card columns only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cards import Split  # noqa: E402

HERE = Path(__file__).parent
s = Split(sys.argv[1] if len(sys.argv) > 1 else "val")
F = s.feeds
disc = 1.0 / np.log2(np.arange(2, 302))

# per-feed card columns in model order
cache = HERE / f"{s.split}_cols.npz"
if cache.exists():
    z = np.load(cache, allow_pickle=True)
    cols = z["cols"].item()
else:
    cols = {k: [] for k in ["t", "hmax", "hidx", "h10", "ucos", "f4", "f5", "f3", "pop", "exp", "M"]}
    for i in range(F.n_feeds):
        rows = s.order(i)
        m = len(rows)
        t = s.table.row(i, named=True)
        ho, hn = int(t["hist_off"]), int(t["H"])
        he = s.hemb[ho:ho + hn]
        ok = he >= 0
        hidx = np.nonzero(ok)[0]
        cv_ok = s.cemb[rows] >= 0
        cvec = np.zeros((m, 128), np.float32)
        cvec[cv_ok] = s.items[s.cemb[rows][cv_ok]].astype(np.float32)
        if len(hidx):
            sim = (cvec @ s.items[he[ok]].astype(np.float32).T) / 65536.0
            b = sim.argmax(1)
            hm, hi = sim[np.arange(m), b], hidx[b]
            rec = hidx < 10
            h10 = sim[:, rec].max(1) if rec.any() else np.full(m, np.nan)
        else:
            hm, hi, h10 = np.full(m, np.nan), np.zeros(m, int), np.full(m, np.nan)
        hm = np.where(cv_ok, hm, np.nan)
        u = s.uemb[i]
        uc = (cvec @ u) / 65536.0 if u.any() else np.full(m, np.nan)
        uc = np.where(cv_ok, uc, np.nan)
        for k, v in [("t", s.target[rows] if s.target is not None else np.zeros(m)), ("hmax", hm), ("hidx", hi),
                     ("h10", h10), ("ucos", uc), ("f4", s.f["f4"][rows]), ("f5", s.f["f5"][rows]),
                     ("f3", s.f["f3"][rows]), ("pop", s.hist_cnt[rows]), ("exp", s.train_exp[rows]), ("M", m)]:
            cols[k].append(v)
    np.savez(cache, cols=np.array(cols, dtype=object))


def ndcg_t(t):
    ideal = disc[:int(t.sum())].sum()
    return float((t * disc[:len(t)]).sum() / ideal) if ideal > 0 else 1.0


def demote(m, flagged, to):
    """Move flagged model-order indices (0-based) down to position `to` (0-based), keeping their order."""
    if not len(flagged):
        return None
    keep = [x for x in range(m) if x not in set(flagged)]
    pos = min(to, len(keep))
    return keep[:pos] + list(flagged) + keep[pos:]


def run(rule, idx):
    d, fired, items, items_pos = [], 0, 0, 0
    for i in idx:
        c = {k: cols[k][i] for k in cols}
        m = c["M"]
        res = rule(c, m)
        if res is None:
            d.append(0.0); continue
        flagged, to = res
        new = demote(m, flagged, to)
        if new is None:
            d.append(0.0); continue
        fired += 1; items += len(flagged); items_pos += int(c["t"][flagged].sum())
        d.append(ndcg_t(c["t"][new]) - ndcg_t(c["t"]))
    d = np.array(d)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), fired, items, items_pos


def r1(th=0.98, kmin=25, h10max=0.60, lo=1, hi=10, to=15, minm=30, cap=3):
    def rule(c, m):
        if m < minm:
            return None
        r = np.arange(m)
        f = (r >= lo) & (r < hi) & (c["hmax"] >= th) & (c["hidx"] >= kmin) & (np.nan_to_num(c["h10"], nan=0) <= h10max)
        fl = np.nonzero(f)[0][:cap]
        return (fl, to) if len(fl) else None
    return rule


def cold_top(hi=20, to=30, minm=30):
    def rule(c, m):
        if m < minm:
            return None
        r = np.arange(m)
        f = (r >= 1) & (r < hi) & (c["f5"] <= 0.02) & (c["pop"] == 0) & (c["exp"] == 0)
        fl = np.nonzero(f)[0][:3]
        return (fl, to) if len(fl) else None
    return rule


def fresh_top(f4max=0.12, hi=10, to=20, minm=30):
    def rule(c, m):
        if m < minm:
            return None
        r = np.arange(m)
        f = (r >= 1) & (r < hi) & (c["f4"] <= f4max)
        fl = np.nonzero(f)[0][:3]
        return (fl, to) if len(fl) else None
    return rule


def lowucos_top(umax=0.40, hi=10, to=20, minm=30):
    def rule(c, m):
        if m < minm:
            return None
        r = np.arange(m)
        f = (r >= 1) & (r < hi) & (np.nan_to_num(c["ucos"], nan=1) <= umax)
        fl = np.nonzero(f)[0][:3]
        return (fl, to) if len(fl) else None
    return rule


if s.split == "val":
    key = json.load(open(HERE / "val/private/eval_key.json"))
    eval_idx = np.array(sorted(key.values()))
    study_txt = "".join(open(p).read() for p in sorted((HERE / "val/study").glob("*.txt")))
    # study feeds are identified by being in the study files: rebuild their fidx from build_val's sampling
    g = s.group_key()
    eval_users = set(g[eval_idx])
    dev_idx = np.array([i for i in range(F.n_feeds) if g[i] not in eval_users])
    print(f"dev feeds {len(dev_idx)} (includes the 100 study feeds, 0.7%), eval feeds {len(eval_idx)}")
    rules = {
        "R1 as written (.98, k>=25, h10<=.60, r2-10 -> 16, M>=30)": r1(),
        "R1 no size limit": r1(minm=0),
        "R1 h10 any": r1(h10max=9),
        "R1 k>=10": r1(kmin=10),
        "R1 k>=50": r1(kmin=50),
        "R1 th .95": r1(th=0.95),
        "R1 th .99": r1(th=0.99),
        "R1 incl. rank 1": r1(lo=0),
        "R1 -> rank 11": r1(to=10),
        "R1 -> rank 30": r1(to=29),
        "R1 ranks 2-20 -> 30": r1(hi=20, to=29),
        "cold (f5<=.02,pop0,exp0) r2-20 -> 31": cold_top(),
        "fresh (f4<=.12) r2-10 -> 21": fresh_top(),
        "low ucos (<=.40) r2-10 -> 21": lowucos_top(),
    }
    for name, rule in rules.items():
        out = []
        for lab, idx in [("dev", dev_idx), ("eval", eval_idx)]:
            mu, se, fired, items, pos = run(rule, idx)
            out.append(f"{lab}: {mu:+.5f} +- {se:.5f} fired {fired} ({fired / len(idx):.1%}) items {items} pos {pos} ({pos / max(items, 1):.1%})")
        print(f"{name:55s} | " + " | ".join(out))
