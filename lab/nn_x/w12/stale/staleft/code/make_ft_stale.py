"""Generate finetune_stale.py from the frozen copy finetune.py (md5 in MD5_SOURCE.txt) by exact, once-only replacements."""
src = open("finetune.py").read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


# 0. imports: the stale switch on data.feat_path (as nn_x/w5/stale_pred.py does)
rep("""from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, predict, save_json
from data import SplitData, feat_exists
""", """import sys

sys.path.insert(0, "<repo>/nn_x/w12/stale/staleft")
import data as Dm
from common import CKPT, LOGS, PREDS, ROOT, Logger, eval_ndcg, predict, save_json
from data import SplitData, feat_exists
from stale_map import STALE_XE, STALE_XT

# staleft: val files of the stale view (private-like scenario); read only while STALE["on"] is not None
STALE = {"on": None}
_feat_path = Dm.feat_path


def _stale_feat_path(split, name):
    if STALE["on"] is not None and split == "val" and name in STALE["on"]:
        return STALE["on"][name]
    return _feat_path(split, name)


Dm.feat_path = _stale_feat_path  # SplitData.attach_xt / attach_extra resolve feature files through this module-level name
""")
# 1. options
rep("""    return p.parse_args()
""", """    p.add_argument("--stale_train", type=float, default=0.0,
                   help="staleft: probability that a TRAINING val feed of a fine-tune step reads the stale (private-like) "
                        "inputs; 0 = finetune.py's training path, bit for bit")
    p.add_argument("--only_folds", default="", help="staleft test harness: comma list of folds to run (default all)")
    return p.parse_args(argv)
""")
rep("def parse():\n", "def parse(argv=None):\n")
rep("def main():\n    a = parse()\n", "def main(argv=None):\n    a = parse(argv)\n")
# 2. the stale val view: same feeds / tensors, xt and xe from the counterfactual files, extras with the FRESH val statistics
rep("""        log(f"extra features ({len(names)}): {names}")
""", """        log(f"extra features ({len(names)}): {names}")
    vs = copy.copy(va)  # staleft: stale val view (shares every tensor of va except xt / xe)
    xt_names = ck["cfg"].get("xt_names") or []
    STALE["on"] = STALE_XT
    if xt_names:
        vs.xt = None
        vs.attach_xt(xt_names)
    STALE["on"] = STALE_XE
    if names:
        vs.xe = None
        vs.attach_extra(names, st)
    STALE["on"] = None
    sw_xt = [n for n in xt_names if n in STALE_XT or (n.startswith("has:") and n[4:] in STALE_XT)]
    sw_xe = [n for n in names if n in STALE_XE]
    d_xt = [float((vs.xt[:, i].float() - va.xt[:, i].float()).abs().mean()) for i in range(len(xt_names))] if xt_names else []
    d_xe = [float((vs.xe[:, i] - va.xe[:, i]).abs().mean()) for i in range(len(names))] if names else []
    assert all((d > 0) == (n in sw_xt) or (n in sw_xt and d == 0) for n, d in zip(xt_names, d_xt)), "xt: an unswitched column changed"
    assert all((d > 0) == (n in sw_xe) or (n in sw_xe and d == 0) for n, d in zip(names, d_xe)), "xe: an unswitched column changed"
    log(f"staleft: stale_train {a.stale_train}; stale view switches {len(sw_xt)} xt columns {sw_xt} and {len(sw_xe)} extras {sw_xe}")
    log("staleft: mean |stale - fresh| on the model scale, xt: " + ", ".join(f"{n} {d:.4f}" for n, d in zip(xt_names, d_xt) if d > 0))
    log("staleft: mean |stale - fresh| on the model scale, extras: " + ", ".join(f"{n} {d:.4f}" for n, d in zip(names, d_xe) if d > 0))
    oof0_stale = np.zeros(va.n_cand, np.float32)  # starting (train-only) model on the stale view
    oof_fin_stale = np.zeros(va.n_cand, np.float32)  # final EMA weights on the stale view
    only = [int(x) for x in a.only_folds.split(",") if x != ""]
""")
rep("""    for k, (tr_f, ho_f) in enumerate(gkf.split(np.arange(va.n), groups=fold_groups)):
        curve = []
""", """    for k, (tr_f, ho_f) in enumerate(gkf.split(np.arange(va.n), groups=fold_groups)):
        if only and k not in only:
            fold_curves.append([])
            continue
        curve = []
""")
rep("""            rng = np.random.default_rng(1000 * k + s)
""", """            rng = np.random.default_rng(1000 * k + s)
            rng_s = np.random.default_rng(777000 + 1000 * k + s)  # staleft: its own stream, the batch stream is untouched
""")
rep("""                oof[0] += predict(model, va, ho_f)
""", """                oof[0] += predict(model, va, ho_f)
                oof0_stale += predict(model, vs, ho_f)
""")
rep("""                    b = src.gather(fb)  # train batches carry no "xe" -> the extra branch is skipped for them
""", """                    b = src.gather(fb)  # train batches carry no "xe" -> the extra branch is skipped for them
                    if a.stale_train > 0 and src is va:  # staleft: these feeds read the stale inputs in this step
                        sm = np.ones(len(fb), bool) if a.stale_train >= 1 else rng_s.random(len(fb)) < a.stale_train
                        if sm.any():
                            smt = torch.from_numpy(sm).to(dev)[:, None, None]
                            if "xt" in b:
                                b["xt"] = torch.where(smt, vs.xt[b["ci"]].float(), b["xt"])
                            if "xe" in b:
                                b["xe"] = torch.where(smt, vs.xe[b["ci"]], b["xe"])
""")
rep("""                oof_fin += predict(ema, va, ho_f) / a.seeds
""", """                oof_fin += predict(ema, va, ho_f) / a.seeds
                oof_fin_stale += predict(ema, vs, ho_f) / a.seeds
""")
rep("""    scores = [eval_ndcg(oof[e], va) for e in range(E + 1)]
""", """    if only:  # staleft test harness: partial run, raw arrays only
        np.savez(PREDS / f"nn_{out}_partial_folds{a.only_folds.replace(',', '_')}.npz", oof_fin=oof_fin, oof_fin_stale=oof_fin_stale,
                 oof0=oof[0], oof0_stale=oof0_stale, test_fin_folds=test_fin_folds.astype(np.float16), oof=oof.astype(np.float32))
        log(f"partial run (folds {only}) saved, {(time.time() - t0) / 60:.1f} min")
        return
    scores = [eval_ndcg(oof[e], va) for e in range(E + 1)]
""")
rep("""        np.save(PREDS / f"nn_{out}_test_folds.npy", test_fin_folds.astype(np.float16))
""", """        np.save(PREDS / f"nn_{out}_test_folds.npy", test_fin_folds.astype(np.float16))
        np.save(PREDS / f"nn_{out}_val_oof_stale.npy", oof_fin_stale)
        np.save(PREDS / f"nn_{out}_val_e0.npy", oof[0])
        np.save(PREDS / f"nn_{out}_val_e0_stale.npy", oof0_stale)
        log(f"staleft: final EMA weights on the stale view: OOF ndcg {eval_ndcg(oof_fin_stale, va):.5f}; "
            f"starting model fresh {eval_ndcg(oof[0], va):.5f} stale {eval_ndcg(oof0_stale, va):.5f}")
""")
rep("""if __name__ == "__main__":
    main()
""", """if __name__ == "__main__":
    main()
""")
open("finetune_stale.py", "w").write(src)
print("written finetune_stale.py")
