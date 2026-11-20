"""Read finished arms: the nats.py line (SCREEN + member rule), the paired level difference to B0 on both witnesses, the paired NATS difference
to B0 (per-feed loss difference of the two 2-parameter conditional logits), the split of the NATS gain by "user has an earlier train feed"
(the support of block a), and bundles (member-equal z-mean, nats.py bundle form).
usage: python -u read_arms.py b1_mcc4_s0 b3_abc_s0 ...  [--bundle b0+b3_abc_s0 ...]     (tag "b0" = infra ts_lr020_s0)"""
import json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
import nats
from nats import clogit, screen

args = sys.argv[1:]
bundles = []
if "--bundle" in args:
    i = args.index("--bundle"); bundles = args[i + 1:]; args = args[:i]
m = nats._meta(); ev = cbx.Evaluator()


def P(tag, sp):
    if tag == "b0":
        return np.load(B0V if sp == "val" else B0H)
    return np.load(PREDS / f"{tag}_{sp}.npy")


cov = {}
for sp in ("val", "hold"):
    a = np.load(FEAT / f"a_{sp}.npy", mmap_mode="r")[:, 0]
    cov[sp] = np.isfinite(np.asarray(a[m.off[sp]]))            # per feed: the user has an earlier non-holdout train feed


def per_feed(sp, x):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; zw = m.zw[sp]
    zx = m.z(sp, np.asarray(x, np.float32)) if not isinstance(x, list) else np.mean([m.z(sp, np.asarray(a, np.float32)) for a in x], 0)
    _, _, lw = clogit(zw[:, None], y, off, M); _, _, l2 = clogit(np.stack([zw, zx], 1), y, off, M)
    return lw - l2, lw != 0


out = {}
base = {sp: per_feed(sp, P("b0", sp)) for sp in ("val", "hold")}
for tag in args:
    r = screen(P(tag, "val"), P(tag, "hold"), tag=tag); out[tag] = r
    if tag == "b0":
        continue
    c = ev.compare(P(tag, "val"), P(tag, "hold"), P("b0", "val"), P("b0", "hold"), label=f"{tag} - B0 level")
    line = []
    for sp in ("val", "hold"):
        d, ok = per_feed(sp, P(tag, sp)); d0, _ = base[sp]; nf = ok.sum(); dd = d - d0
        s = lambda v, msk: (v[msk].sum() / (ok & msk).sum(), np.std(v[msk]) / np.sqrt(msk.sum()) * msk.sum() / (ok & msk).sum())
        allm = np.ones(len(d), bool)
        g, gs = s(dd, allm); gc, gcs = s(dd, cov[sp]); gu, gus = s(dd, ~cov[sp]); nc, ncs = s(d, cov[sp]); nu, nus = s(d, ~cov[sp])
        line.append(f"{sp}: NATS - NATS(B0) {g:+.5f}+-{gs:.5f} | feeds with an earlier user feed ({cov[sp].mean():.3f}): NATS {nc:+.5f}+-{ncs:.5f} (vs B0 {gc:+.5f}+-{gcs:.5f}) | "
                    f"other feeds: NATS {nu:+.5f}+-{nus:.5f} (vs B0 {gu:+.5f}+-{gus:.5f})")
        out[tag][f"vsB0_{sp}"] = dict(nats_diff=float(g), se=float(gs), cov_share=float(cov[sp].mean()), nats_cov=float(nc), nats_cov_se=float(ncs), nats_uncov=float(nu),
                                      nats_uncov_se=float(nus), diff_cov=float(gc), diff_cov_se=float(gcs), diff_uncov=float(gu), diff_uncov_se=float(gus))
    out[tag]["level_vs_b0"] = c
    print(f"[{tag}] " + "\n      ".join(line), flush=True)
for b in bundles:
    tags = b.split("+")
    out["bundle:" + b] = screen([P(t, "val") for t in tags], [P(t, "hold") for t in tags], tag="BUNDLE(" + b + ")")
json.dump(out, open(HERE / f"out/read_{'_'.join(args)[:80]}.json", "w"), indent=1, default=float)
