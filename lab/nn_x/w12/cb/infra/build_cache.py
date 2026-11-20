"""Build the cbx caches. usage: build_cache.py <stage> [...]
  stage core : X177 + ids for xm / hold / val / test, W16 pool z
  stage aux  : teacher targets, raw embeddings (cand int8, user, history means) for xm / hold / val / test
  stage big  : the larger training-set options xs (all short exploration-like feeds) and prod200k (X177, ids, teacher, embeddings)
  stage nn   : NN scores as inputs for val / hold / test (mean11, xf15, xf0..4), train positions
CPU only, 12 threads."""
import sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from cbx import log

t0 = time.time()
for stage in sys.argv[1:]:
    if stage == "core":
        for n in ("xm", "hold", "val", "test"):
            d = cbx.load(n); x = d.X; _ = d.item_id, d.cat, d.emb_idx
            log(f"{d} X {x.shape} {x.dtype}  [{time.time() - t0:.0f}s] rss {cbx.rss_gb()}")
        _ = cbx.load("xm").y
        ev = cbx.Evaluator()
        log(f"W16 pool: val {ev.nd16['val'].mean():.5f} holdout {ev.nd16['hold'].mean():.5f} [{time.time() - t0:.0f}s]")
    elif stage == "aux":
        for n in ("xm", "hold"):
            d = cbx.load(n)
            for w in ("mean11", "oof"):
                t = d.teacher(w); log(f"{n} teacher {w} {t.shape} sd {float(t[:1000000].std()):.3f}")
        for n in ("val", "test", "hold", "xm"):
            d = cbx.load(n)
            e = d.emb_cand_i8; u = d.emb_feed("user"); h = d.emb_feed("hmean")
            log(f"{n} emb cand {e.shape} user {u.shape} hist {h.shape} [{time.time() - t0:.0f}s]")
    elif stage == "big":
        for n in ("xs", "prod200k"):
            d = cbx.load(n); x = d.X; _ = d.item_id, d.cat, d.emb_idx, d.y
            log(f"{d} X {x.shape} [{time.time() - t0:.0f}s]")
            for w in ("mean11", "oof"):
                d.teacher(w)
            e = d.emb_cand_i8; u = d.emb_feed("user"); h = d.emb_feed("hmean")
            log(f"{n} teacher + emb done [{time.time() - t0:.0f}s]")
    elif stage == "nn":  # NN scores as inputs for the evaluation sets + remaining small columns, so that arms never need to write here
        for n in ("val", "hold", "test"):
            d = cbx.load(n)
            for w in ["mean11", "xf15"] + [f"xf{k}" for k in range(5)]:
                cbx.nn_input(d, w)
            for k in ("h5", "h20"):
                d.emb_feed(k)
            log(f"{n} nn inputs done [{time.time() - t0:.0f}s]")
        for n in ("xm", "xs", "prod200k", "hold"):
            d = cbx.load(n); _ = d.pos; d.emb_feed("h5"); d.emb_feed_n("h5")
log("DONE", sys.argv[1:], f"{time.time() - t0:.0f}s")
