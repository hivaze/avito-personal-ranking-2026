"""End-to-end checks of the CPU fine-tune smoke runs (run_ft_smoke.sh). CPU only, light.

  .venv/bin/python check_ft.py identity <tag orig> <tag copy flag-off>
  .venv/bin/python check_ft.py flagon   <tag --full_val>
Rules (written before the runs were read; they mirror PREDECLARED.txt R1 / R2 for the fine-tune copy):
  identity PASS iff logs (wall-clock fields removed) identical line for line, every training batch (feed ids, targets, loss)
     identical, every prediction file identical, summary json identical up to minutes and the full_val key.
  flagon PASS iff every training batch is a VAL batch; in every epoch of every seed each of the 14,423 val feeds appears
     exactly once; steps == seeds x epochs x batches(all val feeds); outputs: nn_<out>_valINSAMPLE.npy and nn_<out>_test.npy
     exist and are finite, NO nn_<out>_val_oof.npy; test_folds has 1 row; the json has insample_* and no oof_* keys; the
     log announces FULLVAL before the first "OOF" figure.
Appends to logs/check_ft.txt.
"""
import glob
import json
import os
import re
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = "<repo>"
LOG = open(f"{HERE}/logs/check_ft.txt", "a")
FAIL = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    P(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}")
    if not ok:
        FAIL.append(name)


def norm_log(path):
    out = []
    for l in open(path).read().splitlines():
        l = re.sub(r"^==== .*? args: ", "==== args: ", l)
        l = re.sub(r"\d+\.\d+ min", "X min", l)
        out.append(l)
    return out


def run(tag):
    d = f"{HERE}/runs/{tag}"
    (lg,) = glob.glob(f"{d}/logs/*_ft.log")
    return d, os.path.basename(lg)[:-7]


def identity(ta, tb):
    P(f"\n== FT IDENTITY: {ta} (original finetune.py) vs {tb} (finetune_full.py, flag off)   [{os.popen('date -u +%FT%TZ').read().strip()}]")
    (da, o), (db, ob) = run(ta), run(tb)
    assert o == ob
    P(f"   {open(da + '/cmd.txt').read().strip()[-260:]}")
    ra, rb = np.load(f"{da}/record.npz"), np.load(f"{db}/record.npz")
    check("both runs finished", bool(ra["ok"]) and bool(rb["ok"]), f"{float(ra['minutes']):.1f} / {float(rb['minutes']):.1f} min; eval cap {ra['cap']}")
    la, lb = norm_log(f"{da}/logs/{o}_ft.log"), norm_log(f"{db}/logs/{o}_ft.log")
    lb[0] = lb[0].replace(", 'full_val': False", "")
    check("logs identical line for line (copy's args line has the extra key full_val False)", la == lb, f"{len(la)} lines")
    for k in ("ll_fidx", "ll_off", "ll_split", "y", "yshapes", "loss", "lgsum", "calls"):
        check(f"record.{k} identical", np.array_equal(ra[k], rb[k]), f"{ra[k].shape}")
    P(f"       training steps {len(ra['loss'])}; loss first / last {ra['loss'][0]:.6f} / {ra['loss'][-1]:.6f}")
    fa = sorted(os.path.basename(p) for p in glob.glob(f"{da}/preds/*.npy"))
    fb = sorted(os.path.basename(p) for p in glob.glob(f"{db}/preds/*.npy"))
    check("same prediction files", fa == fb, fa)
    for p in fa:
        x, y = np.load(f"{da}/preds/{p}"), np.load(f"{db}/preds/{p}")
        check(f"{p} identical", np.array_equal(x, y), f"{x.shape} nonzero {int(np.count_nonzero(x))}")
    ja, jb = json.load(open(f"{da}/logs/{o}_ft.json")), json.load(open(f"{db}/logs/{o}_ft.json"))
    for j in (ja, jb):
        j.pop("minutes")
        j["args"].pop("full_val", None)
    check("summary json identical (minutes removed)", json.dumps(ja, sort_keys=True) == json.dumps(jb, sort_keys=True))


def flagon(tag):
    P(f"\n== FT FLAG ON (--full_val): {tag}   [{os.popen('date -u +%FT%TZ').read().strip()}]")
    d, o = run(tag)
    P(f"   {open(d + '/cmd.txt').read().strip()[-260:]}")
    rec = np.load(f"{d}/record.npz")
    check("run finished", bool(rec["ok"]), f"{float(rec['minutes']):.1f} min")
    j = json.load(open(f"{d}/logs/{o}_ft.json"))
    a = j["args"]
    f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet", columns=["M"])
    M = f["M"].to_numpy().astype(np.int64)
    n = len(M)
    Mp = (M + 15) // 16 * 16
    nb = sum(-(-int(c) // int(max(1, min(a["max_feeds"], a["cand_budget"] // mp)))) for mp, c in zip(*np.unique(Mp, return_counts=True)))
    off = rec["ll_off"]
    steps = len(off) - 1
    check("args: full_val True, folds forced to 1", a["full_val"] is True and a["folds"] == 1, f"seeds {a['seeds']} epochs {a['epochs']}")
    check("every training batch is a val batch", (rec["ll_split"] == "val").all(), f"{steps} steps")
    check(f"steps == seeds x epochs x batches(all {n} val feeds) = {a['seeds']} x {a['epochs']} x {nb}", steps == a["seeds"] * a["epochs"] * nb)
    ok = True
    for e in range(a["seeds"] * a["epochs"]):
        ids = rec["ll_fidx"][off[e * nb]:off[(e + 1) * nb]]
        ok &= len(ids) == n and np.array_equal(np.sort(ids), np.arange(n))
    check(f"in every epoch of every seed each of the {n} val feeds is trained on exactly once (no fold is held out)", ok)
    files = sorted(os.path.basename(p) for p in glob.glob(f"{d}/preds/*.npy"))
    check("outputs: valINSAMPLE + test + test_folds; NO ..._val_oof.npy",
          files == sorted([f"nn_{o}_valINSAMPLE.npy", f"nn_{o}_test.npy", f"nn_{o}_test_folds.npy"]), files)
    t = np.load(f"{d}/preds/nn_{o}_test.npy")
    tf = np.load(f"{d}/preds/nn_{o}_test_folds.npy")
    check("test predictions finite, 2,904,806 rows; test_folds has 1 row", np.isfinite(t).all() and t.shape == (2904806,) and tf.shape[0] == 1, f"{t.shape} {tf.shape}")
    check("json: insample_* keys, no oof_* key, note present", not [k for k in j if k.startswith("oof_")] and "insample_by_epoch" in j and j["full_val"] is True,
          f"train-only val NDCG (epoch 0, capped eval) {j['train_only_val_ndcg_epoch0']:.5f}; in-sample by epoch {np.round(j['insample_by_epoch'], 5).tolist()}")
    lines = open(f"{d}/logs/{o}_ft.log").read().splitlines()
    i_warn = [i for i, l in enumerate(lines) if l.startswith("FULLVAL: the 'OOF ndcg'")]
    i_oof = [i for i, l in enumerate(lines) if l.startswith("OOF ndcg by epoch")]
    check("log: FULLVAL announcement at start, IN-SAMPLE warning right before the first 'OOF' figure, summary line at the end",
          lines[1].startswith("FULLVAL --full_val") and i_warn and i_oof and i_warn[0] == i_oof[0] - 1 and lines[-1].startswith("FULLVAL: summary rewritten"))


if __name__ == "__main__":
    if sys.argv[1] == "identity":
        identity(sys.argv[2], sys.argv[3])
    else:
        flagon(sys.argv[2])
    P(f"   -> {'ALL PASS' if not FAIL else 'FAILED: ' + str(FAIL)}")
    sys.exit(1 if FAIL else 0)
