"""T6: operational checks for the Dec-9 picks: submission validity of the current public-best (G3) and pick files,
score dtype / precision, presence of member checkpoints and prediction files, whether the pick-building scripts live
only in the volatile /tmp scratchpad, and backups of preds/."""
import os, sys, json, glob, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "12")
import numpy as np, polars as pl
R = "<repo>"
out = []
def say(s): print(s, flush=True); out.append(s)
ss = pl.read_parquet(f"{R}/data/competition/sample_submission.parquet")
say(f"sample_submission: {ss.height} rows, columns {ss.columns}, dtypes {ss.dtypes}")
for sub in ["cand_G_F10F7xF7x70_memeq", "cand_PF7x_memeq_6fam", "cand_Pall_memeq_6fam", "sub_v13_Cd_memeq_F5cdF6F7"]:
    p = f"{R}/subs/{sub}.parquet"
    s = pl.read_parquet(p)
    sc = s["score"].to_numpy()
    j = ss.select("feed_id", "item_id").join(s, on=["feed_id", "item_id"], how="left")
    dup = s.select("feed_id", "item_id").is_duplicated().sum()
    ties = s.group_by("feed_id").agg((pl.col("score").n_unique() != pl.len()).alias("t"))["t"].sum()
    # precision: distinct values per feed when cast to float32 / rounded to 6 decimals
    t32 = s.with_columns(pl.col("score").cast(pl.Float32)).group_by("feed_id").agg((pl.col("score").n_unique() != pl.len()).alias("t"))["t"].sum()
    say(f"{sub}: rows {s.height} (missing keys {j['score'].null_count()}, dup {dup}), dtype {s['score'].dtype}, finite {np.isfinite(sc).all()}, ties {ties}, ties after float32 cast {t32}, score range [{sc.min():.4g}, {sc.max():.4g}], mtime {os.path.getmtime(p):.0f}")
fam = json.load(open(f"{R}/nn_x/stack/hfit/cache/families.json"))
members = fam["F10"] + fam["F7x"] + fam["F7x70"]
miss = []
for v in members:
    for fp in [f"nn/ckpt/{v}.pt", f"preds/nn_{v}_trainonly_val.npy", f"preds/nn_{v}_trainonly_test.npy", f"preds/nn_{v}_n2cpv3_val_oof.npy", f"preds/nn_{v}_n2cpv3_test.npy", f"preds/holdout/nn_{v}_holdout.npy", f"nn/logs/{v}_n2cpv3_ft.json"]:
        if not os.path.exists(f"{R}/{fp}"): miss.append(fp)
say(f"G3 members {len(members)}: missing artifacts {len(miss)}: {miss[:10]}")
# where do the checkpoints of the r6 re-cools live?
for v in ["r6_cd_m02_r101", "r6_cd_m02_p70k_r101"]:
    hits = glob.glob(f"{R}/nn/ckpt/{v}*.pt") + glob.glob(f"{R}/nn_x/r6/ckpt/{v}*.pt")
    say(f"{v}: ckpt files {hits[:3]}")
# FT fold models saved?  (finetune.py saves predictions only)
say(f"FT fold-model checkpoints on disk (any *_ft*.pt under nn/ckpt): {len(glob.glob(f'{R}/nn/ckpt/*ft*.pt'))}")
# pick-building scripts: in repo or only in /tmp scratchpad?
for name in ["build_pool.py", "decide5.py", "build_pall.py", "build_pf7x.py", "decide3.py", "decide6.py", "build_gate5.sh"]:
    in_repo = subprocess.run(["bash", "-c", f"find {R} -name {name} -not -path '*/.venv/*' 2>/dev/null | head -3"], capture_output=True, text=True).stdout.split()
    in_tmp = glob.glob(f"/tmp/scratchpad")
    say(f"{name}: in repo {in_repo}; in /tmp scratchpad {len(in_tmp)}")
# disk and backups
du = subprocess.run(["bash", "-c", f"du -sh {R}/preds {R}/nn/ckpt {R}/subs 2>/dev/null; df -h {R} | tail -1"], capture_output=True, text=True).stdout
say("disk: " + du.replace("\n", " | "))
say(f"preds/ file count {len(os.listdir(f'{R}/preds'))}; any backup dir? {glob.glob(f'{R}/*backup*') + glob.glob(f'{R}/preds_bak*') + glob.glob('<mount>')}")
# git?
say("git repo: " + ("yes" if os.path.exists(f"{R}/.git") else "NO (no version control of nn/, src/, decision scripts)"))
# test/val key order vs flat order (grader joins on keys; confirm the sub rows are in flat order and feed ids unique across splits)
ft = pl.read_parquet(f"{R}/data/flat/test/feeds.parquet"); fv = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")
say(f"feed_id overlap val/test: {len(set(fv['feed_id'].to_list()) & set(ft['feed_id'].to_list()))}; test feeds {ft.height}; sample_submission distinct feeds {ss['feed_id'].n_unique()}")
open(f"{R}/nn_x/audit/weakspots/t6_ops.txt", "w").write("\n".join(out) + "\n")
