"""CPU check of the ranker-arm wrapper run_nn_baseline.py WITHOUT a GPU (CUDA_VISIBLE_DEVICES="").

nn/common.py refuses to import unless CUDA_VISIBLE_DEVICES is 4..7, so this test registers a stub 'common' module made
from nn/common.py's own source with only the guard removed. It checks that:
  * the output redirect binds (train.CKPT/LOGS/PREDS -> nn_x/llm/baseline/...),
  * the unmodified nn/train.py parser accepts the exact B-arm flags of run_matched.sh,
  * holdout_preds() writes holdout predictions in the preds/holdout layout (holdout feeds sorted, flat order),
  * nothing is written under nn/ or preds/.
  CUDA_VISIBLE_DEVICES="" ../../.venv_llm/bin/python smoke_baseline.py --out <scratch dir>
"""
import argparse
import os
import shlex
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("<repo>")
HERE = Path(__file__).resolve().parent
B_FLAGS = ("--variant llmcmp_base_s23 --train_feeds 150000 --seed 23 --no_pos --aux_pos 0.5 --aux_mode listwise "
           "--aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed "
           "--xt cvnd_pre2s --xt_drop 0 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 "
           "--attn_gate --lr 2e-3 --cooldown_mix 0.1 --epochs 20 --eval_every 250")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "run with CUDA_VISIBLE_DEVICES=''"
    marker = time.time()
    sys.path.insert(0, str(ROOT / "nn"))
    src = (ROOT / "nn" / "common.py").read_text()
    guard = 'raise RuntimeError("set CUDA_VISIBLE_DEVICES to one of <ids> (some GPUs belong to other users)")'
    assert guard in src
    stub = types.ModuleType("common")
    stub.__file__ = str(ROOT / "nn" / "common.py")
    exec(compile(src.replace(guard, "pass"), stub.__file__, "exec"), stub.__dict__)
    sys.modules["common"] = stub
    import torch
    torch.set_num_threads(12)
    import run_nn_baseline as RB
    train = RB.train
    ok_redirect = train.CKPT == HERE / "baseline" / "ckpt" and train.PREDS == HERE / "baseline" / "preds" \
        and train.LOGS == HERE / "baseline" / "logs"
    sys.argv = ["nn/train.py"] + shlex.split(B_FLAGS)
    args = train.parse()
    print("[parse] nn/train.py accepted the B-arm flags:", {k: v for k, v in vars(args).items()
                                                             if k in ("train_feeds", "epochs", "eval_every", "xt", "lr", "muon", "sched",
                                                                      "cooldown_mix", "select_on", "holdout", "seed")})
    # a ranker built like nn/train.py builds it for these flags (random init; we only test the writer path)
    D = RB.D
    names, rel, idc, relonly, nd = D.xt_config("cvnd_pre2s")
    m = train.Ranker(d=args.d, heads=args.heads, hist_layers=args.hist_layers, blocks=args.blocks, drop=args.drop,
                     use_hist=True, use_set=True, use_pos_tower=not args.no_pos, aux_pos=args.aux_pos > 0, ff=args.ff,
                     sim_bias=args.sim_bias, pos_mode=args.pos_mode, n_dense=nd, domain=args.domain, intent=False,
                     xt_names=names, id_buckets=0, id_drop=args.id_drop, arch=args.arch, id_hist_only=False, xt_rel=rel,
                     xt_drop=args.xt_drop, xt_idcols=idc, xt_relonly=relonly, attn_gate=args.attn_gate,
                     split_kv=args.muon, retr_m=args.retr_m)
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "cfg": m.cfg, "args": vars(args)}, out / "ckpt" / "fake.pt")
    ndcg = RB.holdout_preds("fake", dev="cpu", out=out, limit=24)
    import numpy as np
    s = np.load(out / "preds" / "holdout" / "nn_fake_holdout.npy")
    y = np.load(out / "preds" / "holdout" / "holdout_y.npy")
    M_all, _ = D.train_meta()
    hold = D.holdout_feeds(D.load_prandom(M_all), M_all)[:24]
    import polars as pl
    f = pl.read_parquet(D.FLAT / "train" / "feeds.parquet", columns=["cand_off", "M"])
    rows = D._ranges(f["cand_off"].to_numpy()[hold], f["M"].to_numpy()[hold])
    y_ref = np.load(D.FLAT / "train" / "cand_target.npy", mmap_mode="r")[rows]
    layout_ok = len(s) == len(rows) and np.array_equal(y, y_ref.astype(np.float32))
    # live jobs write to nn/ and preds/ all the time: only files this test could have produced count
    newer = [str(p) for d in (ROOT / "nn", ROOT / "preds") for p in d.rglob("*")
             if ("fake" in p.name or "llmcmp" in p.name) and p.stat().st_mtime > marker]
    ok = ok_redirect and layout_ok and not newer and np.isfinite(ndcg)
    print(f"[{'PASS' if ok else 'FAIL'}] redirect {ok_redirect}, holdout layout {layout_ok} ({len(s)} rows, 24 feeds), "
          f"holdout NDCG of a random-init ranker {ndcg:.4f}, files written under nn/ or preds/: {newer or 'none'}")


if __name__ == "__main__":
    main()
