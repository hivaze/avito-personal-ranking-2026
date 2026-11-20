"""ENCODER-DOSE (A) harness tests (ledger nn_x/abl/ledger/ledger.md section 10.A): the --xt_add_relonly flag and the pre2noid arm.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/encdose/test_encdose.py      -> nn_x/abl/logs/test_encdose.log

  E1 helper: xt_add_relonly_idx flag-off identity (train.py calls it only behind `if a.xt_add_relonly:`; the test layout helper of
     tests/test_abl.py with add_relonly '' == without); refusals (no --xt_add, a name not in --xt_add, duplicates, an index already in
     xt_rel / xt_relonly).
  E2 layout: pre2noid (exclude <6 pre2> + add <6 pre2n> + add_relonly <3>) == the BASE layout with the 6 names swapped at indices 28-33:
     identical xt_rel, xt_relonly, xt_idcols, n_dense (104 = N_DENSE 61 + 2 + 34 + 2 x 5 - 3); dense_features column labels identical up to the pre2 -> pre2n renaming.
  E3 lane: abl_lane.sh DRY pre2noid:23 == base:23 + exactly {xt_exclude, xt_add, xt_add_relonly}; the lane refuses GPUs other than
     ABL_GPUS; test_abl's AST / guard bookkeeping covers the flag (run separately).
  E4 files (after build_pre2noid.py): the 7 pre2n_* files per split are hard links data/feats <-> nn/feat_cache (same inode, nlink 2),
     float32, NaN mask == pre2_fe's, _s == (raw + 20) / 6, visible through nn_x/abl/feat_cache.
  E5 data: MiniSplit xt of the pre2noid layout == the base's xt with columns 28-33 replaced by the pre2n transforms (fp16(clip)) on real
     train candidates; dense_features of the two layouts have identical shape and identical columns outside the 6 raw + 6 rank/z of
     the pre2 block.
  E6 checkpoint round trip: Ranker(**cfg) from a saved cfg reproduces the logits bit-identically; cfg carries the pre2noid layout.
"""
import argparse
import io
import os
import subprocess
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "<repo>"
ABL = f"{ROOT}/nn_x/abl"
sys.path.insert(0, f"{ABL}/tests")
sys.path.insert(1, ABL)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
import abl_arms as AA  # noqa: E402
import model as MO  # noqa: E402
import test_abl as TA  # noqa: E402  (its helpers: xt_layout, dense_cols, dry_cmds, load_parse, _ranker; importing runs no test)
from mini_split import MiniSplit, pick_train_feeds  # noqa: E402

PRE2 = "pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe"
PRE2N = "pre2n_fe_z,pre2n_new_z,pre2n_fe_s,pre2n_new_s,pre2n_hmax32,has:pre2n_fe"
PRE2N_RO = "pre2n_fe_s,pre2n_new_s,pre2n_hmax32"
LOG = open(f"{ABL}/logs/test_encdose.log", "w")
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


def main():
    parse = TA.load_parse(TA.TRAIN_PY)
    P("== E1 helper xt_add_relonly_idx")
    base = TA.xt_layout("cvnd_pre2s")
    check("flag off: xt_layout(add_relonly='') == xt_layout()", TA.xt_layout("cvnd_pre2s", add_relonly="") == base)
    n = base[0]
    errs = {}
    for nm, fn in (("no_add", lambda: AA.xt_add_relonly_idx([0], [1], n, "", "pre2_fe_s", lambda *a: None)),
                   ("not_in_add", lambda: AA.xt_add_relonly_idx([0], [1], n + ["x1", "x2"], "x1,x2", "ctx_nd_max", lambda *a: None)),
                   ("dup", lambda: AA.xt_add_relonly_idx([0], [1], n + ["x1"], "x1", "x1,x1", lambda *a: None)),
                   ("already_rel", lambda: AA.xt_add_relonly_idx([len(n)], [1], n + ["x1"], "x1", "x1", lambda *a: None)),
                   ("already_relonly", lambda: AA.xt_add_relonly_idx([0], [len(n)], n + ["x1"], "x1", "x1", lambda *a: None))):
        try:
            fn()
            errs[nm] = "no error"
        except AssertionError as e:
            errs[nm] = f"AssertionError: {str(e)[:40]}"
    check("refuses: no --xt_add, a name not in --xt_add, duplicates, an index already in xt_rel / xt_relonly",
          all(v.startswith("AssertionError") for v in errs.values()), errs)
    r, ro = AA.xt_add_relonly_idx([0, 1], None, n + ["x1", "x2"], "x1,x2", "x2,x1", lambda *a: None)
    check("appends the named indices in the given order to xt_rel and xt_relonly", r == [0, 1, len(n) + 1, len(n)] and ro == [len(n) + 1, len(n)], (r, ro))

    P("\n== E2 layout: pre2noid == base with the 6 pre2 names swapped at indices 28-33")
    lay = TA.xt_layout("cvnd_pre2s", exclude=PRE2, add=PRE2N, add_relonly=PRE2N_RO)
    swapped = [x.replace("pre2_", "pre2n_") for x in base[0]]
    check(f"xt_names: 34 names, == base names with pre2_ -> pre2n_ (positions 28-33: {lay[0][28:]})", lay[0] == swapped and len(lay[0]) == 34
          and lay[0][28:] == PRE2N.split(",") and base[0][28:] == PRE2.split(","), lay[0][26:])
    check(f"xt_rel / xt_relonly / xt_idcols / n_dense identical to the base: {base[1]} / {base[3]} / {base[2]} / {base[4]}",
          lay[1] == base[1] and lay[3] == base[3] and lay[2] == base[2] and lay[4] == base[4] == MO.N_DENSE + 2 + 34 + 2 * 5 - 3, (lay[1], lay[3], lay[2], lay[4]))
    c_b, c_n = TA.dense_cols(base[0], base[1], base[3]), TA.dense_cols(lay[0], lay[1], lay[3])
    check("dense_features column labels identical up to the renaming (raw block minus the 3 rel-only, then [pct, z] x 5)",
          [c.replace("pre2n_", "pre2_") for c in c_n] == c_b and len(c_n) == 34 - 3 + 10, c_n[-12:])
    xo = TA.xt_layout("cvnd_pre2s", exclude=PRE2)
    check("without --xt_add_relonly the added columns would be raw-only (xt_rel 2, xt_relonly None, n_dense 92): the flag is what restores the base layout",
          TA.xt_layout("cvnd_pre2s", exclude=PRE2, add=PRE2N)[1] == xo[1] and TA.xt_layout("cvnd_pre2s", exclude=PRE2, add=PRE2N)[3] is None
          and TA.xt_layout("cvnd_pre2s", exclude=PRE2, add=PRE2N)[4] == xo[4] + 6 == MO.N_DENSE + 2 + 34 + 4, xo[1:])

    P("\n== E3 lane")
    rc, cmds = TA.dry_cmds(TA.LANE, 4, "pre2noid:23 base:23", r"abl_\w+_s23") if False else (None, None)
    out = subprocess.run(["bash", TA.LANE, "4", "pre2noid:23 base:23"], capture_output=True, text=True, env={**os.environ, "DRY": "1", "ABL_GPUS": "4 5"})
    cmds = {}
    for line in out.stdout.splitlines():
        if " train.py " in line:
            import re
            m = re.search(r"abl_\w+_s23", line)
            cmds[m.group(0)] = line.split(" train.py ", 1)[1].split()
    nb, nn_ = vars(parse(cmds["abl_base_s23"])), vars(parse(cmds["abl_pre2noid_s23"]))
    d = TA.diff_ns(nb, nn_)
    want = {"xt_exclude": ("", PRE2), "xt_add": ("", PRE2N), "xt_add_relonly": ("", PRE2N_RO)}
    check("abl_lane.sh DRY (ABL_GPUS='4 5'): abl_pre2noid_s23 differs from abl_base_s23 ONLY by xt_exclude / xt_add / xt_add_relonly", out.returncode == 0 and d == want, d)
    refused = subprocess.run(["bash", TA.LANE, "4", "pre2noid:23"], capture_output=True, text=True, env={**os.environ, "DRY": "1"}).returncode != 0
    check("abl_lane.sh with the default allow-list refuses GPU N (ABL_GPUS='4 5' must be passed explicitly)", refused)
    lay_cmd = TA.xt_layout("cvnd_pre2s", exclude=nn_["xt_exclude"], rel_extra=nn_["xt_rel_extra"], add=nn_["xt_add"], add_relonly=nn_["xt_add_relonly"])
    check("layout from the lane command == E2 layout", lay_cmd == lay)

    P("\n== E4 files")
    have = all(os.path.exists(f"{ROOT}/nn/feat_cache/{sp}/pre2n_{k}.npy") for sp in ("train", "val", "test") for k in ("fe", "new", "fe_z", "new_z", "fe_s", "new_s", "hmax32"))
    if not have:
        P("  pre2n_* files not built yet: E4 / E5 / E6 skipped (run build_pre2noid.py first)")
    else:
        ok, info = True, {}
        for sp in ("train", "val", "test"):
            sl = slice(None, None, 997) if sp == "train" else slice(None)
            fe = np.load(f"{ROOT}/nn/feat_cache/{sp}/pre2n_fe.npy", mmap_mode="r")[sl]
            ref = np.load(f"{ROOT}/data/feats/{sp}/pre2_fe.npy", mmap_mode="r")[sl]
            same_nan = np.array_equal(np.isnan(fe), np.isnan(ref))
            links = all(os.stat(f"{ROOT}/data/feats/{sp}/pre2n_{k}.npy").st_ino == os.stat(f"{ROOT}/nn/feat_cache/{sp}/pre2n_{k}.npy").st_ino
                        and os.stat(f"{ROOT}/data/feats/{sp}/pre2n_{k}.npy").st_nlink == 2 and os.path.exists(f"{ABL}/feat_cache/{sp}/pre2n_{k}.npy")
                        for k in ("fe", "new", "fe_z", "new_z", "fe_s", "new_s", "hmax32"))
            fs_ = np.load(f"{ROOT}/nn/feat_cache/{sp}/pre2n_fe_s.npy", mmap_mode="r")[sl]
            ns_ = np.load(f"{ROOT}/nn/feat_cache/{sp}/pre2n_new_s.npy", mmap_mode="r")[sl]
            nw = np.load(f"{ROOT}/nn/feat_cache/{sp}/pre2n_new.npy", mmap_mode="r")[sl]
            s_ok = np.array_equal(np.asarray(fs_), ((np.asarray(fe) + 20) / 6).astype(np.float32), equal_nan=True) and \
                np.array_equal(np.asarray(ns_), ((np.asarray(nw) + 20) / 6).astype(np.float32), equal_nan=True)
            f32 = all(np.load(f"{ROOT}/nn/feat_cache/{sp}/pre2n_{k}.npy", mmap_mode="r").dtype == np.float32 for k in ("fe", "fe_z", "fe_s", "hmax32"))
            ok = ok and same_nan and links and s_ok and f32
            info[sp] = (same_nan, links, s_ok, f32)
        check("pre2n_*: hard links data/feats <-> nn/feat_cache (inode, nlink 2; visible via nn_x/abl/feat_cache), float32, NaN mask == pre2_fe's, "
              "_s == (raw + 20) / 6 (train every 997th row)", ok, info)

        P("\n== E5 data: MiniSplit xt and dense_features of the pre2noid layout vs the base layout on real train batches")
        feeds = pick_train_feeds(n_each=24, seed=7)
        ms_b, ms_n = MiniSplit("train", feeds, xt_names=base[0]), MiniSplit("train", feeds, xt_names=lay[0])
        same_outside = torch.equal(ms_b.xt[:, :28], ms_n.xt[:, :28])
        differ_inside = not torch.equal(ms_b.xt[:, 28:], ms_n.xt[:, 28:])
        has_same = torch.equal(ms_b.xt[:, 33], ms_n.xt[:, 33])  # has: indicator identical (same NaN mask)
        import polars as pl
        f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
        fs2 = np.sort(np.asarray(feeds))
        crow = np.concatenate([np.arange(o, o + m) for o, m in zip(f["cand_off"].to_numpy().astype(np.int64)[fs2], f["M"].to_numpy().astype(np.int64)[fs2])])
        want_z = np.nan_to_num(np.clip(np.asarray(np.load(f"{ROOT}/nn/feat_cache/train/pre2n_fe_z.npy", mmap_mode="r")[crow]), -5, 5), nan=0.0).astype(np.float32)
        want_s = np.nan_to_num(np.clip(np.asarray(np.load(f"{ROOT}/nn/feat_cache/train/pre2n_fe_s.npy", mmap_mode="r")[crow]), -10, 10), nan=0.0).astype(np.float32)
        check(f"MiniSplit xt: columns 0-27 identical to the base, columns 28-33 differ, has: (33) identical; col 28 == fp16(clip(pre2n_fe_z, +-5)), "
              f"col 30 == fp16(clip(pre2n_fe_s, +-10)) on {len(crow)} candidates",
              same_outside and differ_inside and has_same and torch.equal(ms_n.xt[:, 28], torch.from_numpy(want_z).half())
              and torch.equal(ms_n.xt[:, 30], torch.from_numpy(want_s).half()), (same_outside, differ_inside, has_same))
        rng = np.random.default_rng(0)
        fb = rng.choice(ms_b.n, 48, replace=False)
        b0, b1 = ms_b.gather(fb, with_pos=True), ms_n.gather(fb, with_pos=True)
        X0 = MO.dense_features(b0, base[4], xt_rel=base[1], xt_drop=0.0, train=False, xt_idcols=base[2], xt_relonly=base[3])
        X1 = MO.dense_features(b1, lay[4], xt_rel=lay[1], xt_drop=0.0, train=False, xt_idcols=lay[2], xt_relonly=lay[3])
        nb_ = MO.N_DENSE + 2
        pre2_cols = [i for i, c in enumerate(c_b) if "pre2" in c]
        other = [i for i in range(len(c_b)) if i not in pre2_cols]
        check(f"dense_features: same shape ({tuple(X0.shape)}); the {len(other)} non-pre2 columns bit-identical; the {len(pre2_cols)} pre2 columns differ",
              X0.shape == X1.shape and torch.equal(X0[..., :nb_], X1[..., :nb_])
              and all(torch.equal(X0[..., nb_ + i], X1[..., nb_ + i]) for i in other) and sum(not torch.equal(X0[..., nb_ + i], X1[..., nb_ + i]) for i in pre2_cols) >= 5,
              (len(other), len(pre2_cols)))

        P("\n== E6 checkpoint round trip")
        a = parse(cmds["abl_pre2noid_s23"])
        m = TA._ranker(a, lay)
        Xd = m.dense(b1)
        m.set_norm(Xd[b1["cmask"]].mean(0), Xd[b1["cmask"]].std(0).clamp_min(1e-3))
        m.eval()
        l1 = m(b1, m.dense(b1))
        buf = io.BytesIO()
        torch.save({"cfg": dict(m.cfg), "state_dict": m.state_dict()}, buf)
        buf.seek(0)
        ck = torch.load(buf, map_location="cpu", weights_only=False)
        m2 = MO.Ranker(**ck["cfg"])
        m2.load_state_dict(ck["state_dict"])
        m2.eval()
        l2 = m2(b1, m2.dense(b1))
        cfg = ck["cfg"]
        check(f"Ranker(**cfg) round trip: logits bit-identical; cfg carries the pre2noid layout (xt_names with pre2n_*, xt_rel / xt_relonly of the base, n_dense {base[4]} = the base's)",
              torch.equal(l1, l2) and list(cfg["xt_names"]) == lay[0] and (cfg.get("xt_rel") or None) == lay[1] and (cfg.get("xt_relonly") or None) == lay[3]
              and cfg["n_dense"] == lay[4] == base[4], (float((l1 - l2).detach().abs().max()), cfg["n_dense"]))
    P(f"\n== {'ALL PASS' if not FAIL else 'FAILURES: ' + str(FAIL)} ({len(FAIL)} failed)")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
