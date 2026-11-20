"""INVENTORY A, measurement M1 (CPU only, read-only): what a wide student reads, straight from its checkpoint cfg and logs.

 * loads the cfg of a held-out wide student, its full-data twin and a popularity-light student (torch.load on CPU;
   the state dict is used only for the shapes of the input layers and the stored normalisation buffers);
 * rebuilds the NAME of each of the n_dense columns in the order model.dense_features emits them
   (nn_x/w5/model.py l.52-132; names are mine, the order is the code's);
 * checks, for every train-stage input, that nn/feat_cache/<split>/<name>.npy exists for train / val / test,
   its dtype, row count against data/flat/<split>/cand_item_id.npy, the hard-link count and the twin path
   (data/feats or data/feats_new) with the same inode;
 * reads the 42 fine-tune extras from the FT log and makes the same file check for val / test.
Writes out/m01_ckpt_inputs.txt, out/dense_columns.tsv, out/xt_files.tsv, out/ft_extras.tsv.  Nothing else is touched.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
import re, sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(12)
R = Path("<repo>")
OUT = R / "reports/INPUTS_1116/model_inputs/out"
L = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


N_ET = 10


def dense_names(cfg):
    """Column names in the order of nn_x/w5/model.py dense_features (l.64-131)."""
    n = []
    for k in range(1, 6):
        n += [f"f{k}", f"f{k}_feedz", f"f{k}_feedpct"]                      # l.66-68
    n += ["cmiss", "log1p_embnorm"]                                          # l.69-70
    n += ["cos_u", "dot_u"]                                                  # l.71-73
    n += ["smax", "smax8", "smax32", "smean", "lse5", "top3"]                # l.81-88
    n += [f"smax_et{t}" for t in range(1, N_ET + 1)]                         # l.90-92
    n += ["log1p_catcnt", "catcnt_share"]                                    # l.98
    n += [f"log1p_catcnt_et{t}" for t in range(1, N_ET + 1)]                 # l.99-100
    n += ["cat_first_recency"]                                               # l.101-103
    n += ["feed_cat_share", "logM", "log1pH"]                                # l.104-105
    for x in ("cos_u", "smax", "lse5", "top3", "log1p_catcnt"):              # l.107-108
        n += [f"{x}_feedz", f"{x}_feedpct"]
    assert len(n) == 61
    n += ["is_user0", "is_user0_x_log1pH"]                                   # l.109-111
    xt = list(cfg.get("xt_names") or [])
    rel = cfg.get("xt_rel") or []
    relonly = set(cfg.get("xt_relonly") or [])
    reln = []
    for j in rel:                                                            # l.123-126
        reln += [f"{xt[j]}|feedpct", f"{xt[j]}|feedz"]
    kept = [f"xt:{xt[i]}" for i in range(len(xt)) if i not in relonly]       # l.127-129
    return n + kept + reln


def describe(tag, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, args, sd = ck["cfg"], ck["args"], ck["state_dict"]
    say(f"\n=== {tag}: {path.relative_to(R)}")
    say("cfg:", {k: v for k, v in cfg.items() if k != "xt_names"})
    xt = cfg.get("xt_names") or []
    say(f"xt_names ({len(xt)}):", xt)
    say("xt_rel ->", [xt[i] for i in (cfg.get("xt_rel") or [])])
    say("xt_relonly ->", [xt[i] for i in (cfg.get("xt_relonly") or [])])
    say("xt_idcols ->", [xt[i] for i in (cfg.get("xt_idcols") or [])])
    names = dense_names(cfg)
    say(f"n_dense cfg {cfg['n_dense']}  rebuilt names {len(names)}  c_mlp.0.weight {tuple(sd['c_mlp.0.weight'].shape)}  wide.weight {tuple(sd['wide.weight'].shape)}  c_lin.weight {tuple(sd['c_lin.weight'].shape)}")
    assert len(names) == cfg["n_dense"] == sd["c_mlp.0.weight"].shape[1]
    say("args (input-relevant):", {k: args.get(k) for k in ("xt", "xt_exclude", "xt_add", "xt_rel_extra", "xt_add_relonly", "d", "retr_m", "max_hist", "user0", "domain", "sim_bias", "kd_file", "kd_w", "kd_T", "holdout", "holdout_in_train", "epochs", "seed")})
    say("params:", sum(v.numel() for k, v in sd.items() if not k.startswith(("f_mean", "f_std"))))
    mods = sorted({k.split(".")[0] for k in sd})
    say("top-level modules / buffers:", mods)
    return cfg, names, sd


cfg_e, names_e, sd_e = describe("held-out wide student", R / "nn_x/w5/ckpt/w5_e10_kd05w192_s951.pt")
cfg_f, names_f, sd_f = describe("full-data wide student", R / "nn_x/w5/ckpt/w5_fd10_kd05w192_s951.pt")
cfg_p, names_p, sd_p = describe("full-data popularity-light student", R / "nn_x/abl/ckpt/abl_fd10_pop1kd05w192_s951.pt")
assert names_e == names_f and cfg_e["xt_names"] == cfg_f["xt_names"]
say("\nheld-out and full-data wide student: identical input lists (xt_names, dense column names)")
say("pop1 minus wide (xt):", [n for n in cfg_p["xt_names"] if n not in cfg_f["xt_names"]], " wide minus pop1:", [n for n in cfg_f["xt_names"] if n not in cfg_p["xt_names"]])

with open(OUT / "dense_columns.tsv", "w") as f:
    f.write("idx\tname\tf_mean\tf_std\twide_w\n")
    for i, n in enumerate(names_f):
        f.write(f"{i}\t{n}\t{float(sd_f['f_mean'][i]):.5f}\t{float(sd_f['f_std'][i]):.5f}\t{float(sd_f['wide.weight'][0, i]):+.5f}\n")

# ---- files behind the train-stage inputs
NROW = {s: len(np.load(R / f"data/flat/{s}/cand_item_id.npy", mmap_mode="r")) for s in ("train", "val", "test")}
say("\nrows per split:", NROW)


def twin(path):
    st = os.stat(path)
    out = []
    for d in ("data/feats", "data/feats_new"):
        q = R / d / path.parent.name / path.name
        if q.exists() and os.stat(q).st_ino == st.st_ino:
            out.append(d)
    return st.st_nlink, ",".join(out) or "-"


def file_rows(names, splits, fn):
    rows = []
    for n in names:
        g = n[4:] if n.startswith("has:") else n
        r = [n]
        for s in splits:
            p = R / "nn/feat_cache" / s / f"{g}.npy"
            if not p.exists():
                r += ["MISSING", "", "", ""]
                continue
            x = np.load(p, mmap_mode="r")
            nl, tw = twin(p)
            r += [str(x.dtype), "ok" if len(x) == NROW[s] and x.ndim == 1 else f"ROWS {x.shape}", str(nl), tw]
        rows.append(r)
    with open(OUT / fn, "w") as f:
        f.write("name\t" + "\t".join(f"{s}_dtype\t{s}_rows\t{s}_nlink\t{s}_twin" for s in splits) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    return rows


xr = file_rows(cfg_f["xt_names"] + [n for n in cfg_p["xt_names"] if n not in cfg_f["xt_names"]], ("train", "val", "test"), "xt_files.tsv")
bad = [r for r in xr if "MISSING" in r or any(c.startswith("ROWS") for c in r)]
say(f"train-stage input files: {len(xr)} names x 3 splits checked, problems: {bad}")
say("dtypes:", sorted({c for r in xr for c in r[1::4]}), " twins:", sorted({c for r in xr for c in r[4::4]}))

ftlog = (R / "nn_x/w5/logs/w5_fd10_kd05w192_s951_n2cpv3s3_ft.log").read_text()
ft = re.findall(r"'([^']+)'", [l for l in ftlog.splitlines() if l.startswith("extra features")][0].split(":", 1)[1])
say(f"\nfine-tune extras from the FT log ({len(ft)}):", ft)
fr = file_rows(ft, ("val", "test"), "ft_extras.tsv")
say("fine-tune extra files problems:", [r for r in fr if "MISSING" in r or any(c.startswith("ROWS") for c in r)])
say("fine-tune extras that exist for TRAIN rows in nn/feat_cache:", [n for n in ft if (R / "nn/feat_cache/train" / f"{n}.npy").exists()])
say("fine-tune extras that exist for TRAIN rows in data/feats:", [n for n in ft if (R / "data/feats/train" / f"{n}.npy").exists()])
say("fine-tune extras also among the 34 train-stage inputs:", [n for n in ft if n in cfg_f["xt_names"]])
say("extra branch of a fine-tuned model = Linear(42, 2d) GELU Linear(2d, d), zero-init last layer (model.add_extra); stats from val (attach_extra)")

# every file in nn/feat_cache that no current member reads (candidates that already have a train/val/test array)
used = {n[4:] if n.startswith("has:") else n for n in cfg_f["xt_names"] + cfg_p["xt_names"]}
allt = sorted(p.stem for p in (R / "nn/feat_cache/train").glob("*.npy"))
say("\nnn/feat_cache/train files not read by the wide or the pop1 student:", [n for n in allt if n not in used])
say("of these present for val and test too:", [n for n in allt if n not in used and all((R / "nn/feat_cache" / s / f"{n}.npy").exists() for s in ("val", "test"))])
(OUT / "m01_ckpt_inputs.txt").write_text("\n".join(L) + "\n")
