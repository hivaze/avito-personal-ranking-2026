"""CPU copy of nn_x/cb26/cb_member.py (E3 XM: the hard-label CatBoost of record) for the round-2 track.

Same model definition as the record: CatBoostRanker YetiRank, depth 8, lr 0.08, l2_leaf_reg 5, border_count 254, seed 0,
2500 iterations, od Iter / od_wait 200, eval_metric NDCG:type=Base, 177 features (category_id native categorical),
train feeds = p_random > 0.5 and M >= 50 minus the 30K holdout (134,118 feeds).
Differences from the record's script (all forced by the round-2 rules):
  1. CPU: task_type="GPU", devices and gpu_ram_part are removed; thread_count=12; CUDA_VISIBLE_DEVICES must be "".
  2. data come from cbx (same matrices as cbdata.load_tab: checked in smoke.py); outputs go to nn_x/w12/cb/infra/out.
  3. --sel valA      : the record's rule: early stopping on val user-half A (val labels are used for MODEL SELECTION).
     --sel trainside : no val label anywhere: fit on the train-side fit split (about 90% of users), early stopping on the
                       train-side selection split (about 10% of users, cbx.trainside_split).
  4. use_best_model=False + predict(ntree_end=best_iter+1) (same predictions as use_best_model=True, but all trees are kept so
     that staged curves can be read); a callback stops training cleanly when --budget_min is reached or when the file
     out/<tag>.STOP appears (touch it to end a run early and still get its predictions and evaluation).
  5. evaluation by cbx.Evaluator (val, holdout, rho to W16, law, member rule); staged curve every --curve_every trees.
usage: CUDA_VISIBLE_DEVICES="" python cb_member_cpu.py --tag repro_valA --sel valA [--budget_min 110]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx  # noqa: E402  (asserts CPU-only env)
from cbx import log  # noqa: E402

OUT = cbx.HERE / "out"; OUT.mkdir(exist_ok=True)


class Budget:
    """CatBoost callback: stop when the wall-clock budget is used; log the eval curve."""

    def __init__(self, seconds, path):
        self.t0 = time.time(); self.seconds = seconds; self.path = path; self.rows = []; self.stopped = False

    def after_iteration(self, info):
        try:
            mets = {f"{ds}:{k}": float(v[-1]) for ds, d in info.metrics.items() for k, v in d.items()}
        except Exception:  # noqa: BLE001
            mets = {}
        self.rows.append((int(info.iteration), time.time() - self.t0, mets))
        if info.iteration % 50 == 0:
            with open(self.path, "w") as f:
                json.dump(self.rows, f)
        if time.time() - self.t0 > self.seconds or os.path.exists(str(self.path).replace(".curve.json", ".STOP")):
            self.stopped = True  # wall-clock budget used, or the STOP file out/<tag>.STOP was created (clean stop, model kept)
            return False
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="YetiRank")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--od_wait", type=int, default=200)
    ap.add_argument("--sel", choices=["valA", "trainside"], required=True)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--budget_min", type=float, default=1e9)
    ap.add_argument("--curve_every", type=int, default=100)
    ap.add_argument("--max_feeds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rsm", type=float, default=0.0, help="0 = CatBoost default (1.0)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from catboost import CatBoostRanker
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    tr = cbx.load("xm")
    names = tr.names
    log(f"{len(names)} features; {tr}")
    keep = np.ones(tr.n_feeds, bool)
    if a.max_feeds and tr.n_feeds > a.max_feeds:
        keep[:] = False
        keep[np.random.default_rng(0).choice(tr.n_feeds, a.max_feeds, replace=False)] = True
    va = cbx.load("val")
    # the record's val halves: GroupKFold(2) by user, user-0 feeds as singletons
    u = va.uid_feed; g = np.where(u == 0, -1 - np.arange(len(u)), u)
    hA, hB = [np.sort(x) for _, x in GroupKFold(n_splits=2).split(np.arange(va.n_feeds), groups=g)]
    assert (hB == np.load(cbx.ROOT / "nn_x/cb26/data/e3_xm_yeti_halfB_feeds.npy")).all(), "val halves differ from the record"
    mA = np.zeros(va.n_feeds, bool); mA[hA] = True
    if a.sel == "valA":
        cbx.ALLOW_EVAL_LABELS = True  # record-reproduction mode: val half A labels select the iteration count
        fit = keep
        ptr = cbx.make_pool(tr, feeds=fit, label=tr.y)
        pes = cbx.make_pool(va, feeds=mA, label=va.y)
        cbx.ALLOW_EVAL_LABELS = False
        n_sel = int(mA.sum())
    else:
        fit, sel = cbx.trainside_split(tr, a.frac)
        fit &= keep; sel &= keep
        assert not (fit & sel).any() and len(np.intersect1d(tr.uid_feed[fit][tr.uid_feed[fit] != 0], tr.uid_feed[sel])) == 0
        ptr = cbx.make_pool(tr, feeds=fit, label=tr.y)
        pes = cbx.make_pool(tr, feeds=sel, label=tr.y)
        n_sel = int(sel.sum())
    log(f"sel={a.sel}: fit feeds {int(fit.sum())} rows {ptr.num_row()} | selection feeds {n_sel} rows {pes.num_row()} | "
        f"pools ready {time.time() - t0:.0f}s rss {cbx.rss_gb()}")
    kw = dict(loss_function=a.loss, eval_metric="NDCG:type=Base", iterations=a.iters, learning_rate=a.lr, depth=a.depth,
              l2_leaf_reg=5, border_count=254, random_seed=a.seed, metric_period=50, od_type="Iter", od_wait=a.od_wait,
              use_best_model=False, allow_writing_files=False, thread_count=cbx.THREADS)
    if a.rsm:
        kw["rsm"] = a.rsm
    m = CatBoostRanker(**kw)
    cb = Budget(a.budget_min * 60, OUT / f"{a.tag}.curve.json")
    tf = time.time()
    m.fit(ptr, eval_set=pes, verbose=50, callbacks=[cb])
    fit_s = time.time() - tf
    json.dump(cb.rows, open(OUT / f"{a.tag}.curve.json", "w"))
    n_trees = m.tree_count_
    best = int(m.get_best_iteration())
    log(f"fit done: {n_trees} trees in {fit_s:.0f}s ({fit_s / max(n_trees, 1):.2f} s/iter), best_iter {best}, "
        f"budget stop {cb.stopped}, rss {cbx.rss_gb()}")
    del ptr, pes
    m.save_model(str(OUT / f"{a.tag}.cbm"))
    ev = cbx.Evaluator()
    te, ho = cbx.load("test"), cbx.load("hold")
    pv_pool, ph_pool, pt_pool = cbx.make_pool(va), cbx.make_pool(ho), cbx.make_pool(te)
    k = best + 1
    pv = m.predict(pv_pool, ntree_end=k, thread_count=cbx.THREADS).astype(np.float32)
    ph = m.predict(ph_pool, ntree_end=k, thread_count=cbx.THREADS).astype(np.float32)
    pt = m.predict(pt_pool, ntree_end=k, thread_count=cbx.THREADS).astype(np.float32)
    nv = ev.ndcg("val", pv)
    imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), names), reverse=True)[:25]
    extra = dict(args=vars(a), n_fit_feeds=int(fit.sum()), n_sel_feeds=n_sel, best_iter=best, trees_built=int(n_trees),
                 budget_stop=bool(cb.stopped), fit_seconds=fit_s, s_per_iter=fit_s / max(n_trees, 1),
                 val_halfA=float(nv[hA].mean()), val_halfB=float(nv[hB].mean()), time=time.time() - t0,
                 top=[(n, float(x)) for x, n in imp], peak_rss_gb=cbx.rss_gb()[1])
    # staged curve: val (full / half A / half B) and holdout NDCG every curve_every trees (evaluation only)
    curve = []
    sv = m.staged_predict(pv_pool, eval_period=a.curve_every, thread_count=cbx.THREADS)
    sh = m.staged_predict(ph_pool, eval_period=a.curve_every, thread_count=cbx.THREADS)
    for i, (xv, xh) in enumerate(zip(sv, sh)):
        nt = min((i + 1) * a.curve_every, n_trees)
        c = ev.ndcg("val", np.asarray(xv, np.float32)); h = ev.ndcg("hold", np.asarray(xh, np.float32))
        curve.append(dict(trees=nt, val=float(c.mean()), val_halfA=float(c[hA].mean()), val_halfB=float(c[hB].mean()), hold=float(h.mean())))
        log(f"   staged {nt:5d} trees: val {c.mean():.5f} (A {c[hA].mean():.5f} B {c[hB].mean():.5f}) holdout {h.mean():.5f}")
    extra["staged"] = curve
    r = ev.evaluate(a.tag, pv, ph, pt, out_dir=OUT, extra=extra)
    log("done", json.dumps({k_: v for k_, v in extra.items() if k_ not in ("top", "staged")}))
    log(f"RESULT {a.tag}: val_full {r['val']:.5f} halfA {extra['val_halfA']:.5f} halfB {extra['val_halfB']:.5f} holdout {r['hold']:.5f} "
        f"best_iter {best} trees {n_trees}")


if __name__ == "__main__":
    main()
