"""famN reads. usage: CUDA_VISIBLE_DEVICES="" python read.py [--n 30] [--pairs ARM:REF ARM:REF ...] [--md]
For every (arm, reference) pair at one scale: per-seed level and NATS (from logs/results.jsonl = nats.py output), the seed-mean, and the PAIRED per-feed
differences arm - reference on val (14,423 feeds) and on the holdout (30,000 feeds):
  d level = NDCG(arm) - NDCG(ref) per feed;   d nats = loss([z_W16, z_ref]) - loss([z_W16, z_arm]) per feed (conditional logit of nats.py, each fitted
  on the witness; the W16-only term cancels). Seeds are paired (arm seed s with reference seed s) and the per-feed differences are averaged over the
  seeds before the mean and SE over feeds are taken (SE = sd over feeds / sqrt(n feeds): it does not contain the seed-to-seed variance of a fit).
WINNER (protocol) = d nats >= 2 SE on val AND on the holdout."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn  # noqa: E402
from fn import cbx, np  # noqa: E402
sys.path.insert(0, str(fn.ROOT / "nn_x/w12/cb"))
import nats  # noqa: E402

REF = dict(Y="Yref", R="Rref", L="Lref")
NC = fn.CACHE / "nats"; NC.mkdir(parents=True, exist_ok=True)
_ev = None


def ev():
    global _ev
    if _ev is None:
        _ev = cbx.Evaluator()
    return _ev


def per_feed(tag):
    """dict sp -> (ndcg per feed, 2-parameter clogit loss per feed)."""
    out = {}
    m = nats._meta()
    for sp in ("val", "hold"):
        p = NC / f"{tag}_{sp}.npz"
        if not p.exists():
            x = np.load(fn.PRED / f"{tag}_{sp}.npy")
            zx = m.z(sp, x); zw = m.zw[sp]
            _, _, l2 = nats.clogit(np.stack([zw, zx], 1), m.y[sp].astype(np.float64), m.off[sp], m.M[sp])
            nd = ev().ndcg(sp, x)
            np.savez(p, nd=nd, l2=l2)
        z = np.load(p); out[sp] = (z["nd"], z["l2"])
    return out


def results():
    r = {}
    for line in open(fn.LOGS / "results.jsonl"):
        d = json.loads(line); r[d["tag"]] = d
    return r


def compare(arm, ref, nk, seeds, res):
    rows = {}
    for sp in ("val", "hold"):
        dl, dn = [], []
        for s in seeds:
            a, b = per_feed(f"{arm}_n{nk}k_s{s}")[sp], per_feed(f"{ref}_n{nk}k_s{s}")[sp]
            dl.append(a[0] - b[0]); dn.append(b[1] - a[1])
        dl = np.mean(dl, 0); dn = np.mean(dn, 0); nf = int((per_feed(f"{ref}_n{nk}k_s{seeds[0]}")[sp][1] != 0).sum()); n = len(dl)
        rows[sp] = dict(dlevel=float(dl.mean()), dlevel_se=float(dl.std() / np.sqrt(n)), dnats=float(dn.sum() / nf), dnats_se=float(dn.std() / np.sqrt(n) * n / nf), n=n)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--pairs", nargs="*", default=None); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    res = results(); nk = a.n
    arms = {}
    for t, d in res.items():
        if d["nfit"] // 1000 == nk and not d.get("dry"):
            arms.setdefault(d["arm"], []).append(d["seed"])
    pairs = [p.split(":") for p in a.pairs] if a.pairs else [(k, REF[k[0]]) for k in arms if k not in REF.values()]
    print(f"== scale n{nk}k; arms on disk: " + ", ".join(f"{k}{sorted(v)}" for k, v in sorted(arms.items())))
    for k in sorted(arms):
        for s in sorted(arms[k]):
            d = res[f"{k}_n{nk}k_s{s}"]
            print(f"  {d['tag']:28s} level {d['val']:.5f} / {d['hold']:.5f} | rho {d['rho_val']:.3f} / {d['rho_hold']:.3f} | NATS {d['nats_val']:+.5f}+-{d['nats_val_se']:.5f} / "
                  f"{d['nats_hold']:+.5f}+-{d['nats_hold_se']:.5f} | trees {d['trees']} fit {d['fit_min']:.1f} min ({d['s_per_iter']:.2f} s/it) model {d['model_mb']:.1f} MB "
                  f"| new-feature importance {d['new_feature_importance']}")
    out = []
    for arm, ref in pairs:
        if arm not in arms or ref not in arms:
            continue
        seeds = sorted(set(arms[arm]) & set(arms[ref]))
        if not seeds:
            continue
        c = compare(arm, ref, nk, seeds, res)
        v, h = c["val"], c["hold"]
        win = v["dnats"] >= 2 * v["dnats_se"] and h["dnats"] >= 2 * h["dnats_se"]
        mean = lambda key: float(np.mean([res[f"{arm}_n{nk}k_s{s}"][key] for s in arms[arm]]))
        line = (f"{arm} - {ref} (seeds {seeds}): d level val {v['dlevel']:+.5f}+-{v['dlevel_se']:.5f} (z {v['dlevel'] / v['dlevel_se']:+.1f}) hold {h['dlevel']:+.5f}+-{h['dlevel_se']:.5f} "
                f"(z {h['dlevel'] / h['dlevel_se']:+.1f}) | d NATS val {v['dnats']:+.5f}+-{v['dnats_se']:.5f} (z {v['dnats'] / v['dnats_se']:+.1f}) hold {h['dnats']:+.5f}+-{h['dnats_se']:.5f} "
                f"(z {h['dnats'] / h['dnats_se']:+.1f}) -> {'WINNER' if win else 'no'} | arm mean level {mean('val'):.5f} / {mean('hold'):.5f}, NATS {mean('nats_val'):+.5f} / {mean('nats_hold'):+.5f}")
        print(line); out.append(dict(arm=arm, ref=ref, seeds=seeds, nk=nk, **{f"{sp}_{k}": c[sp][k] for sp in c for k in c[sp]}, winner=bool(win)))
    with open(fn.LOGS / f"compare_n{nk}k.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
