"""Reviewer's comparison of two rv_cpu.py runs.
  rv_cmp.py identity <runA> <runB>                       original vs copy without the flag
  rv_cmp.py flagon <runC> <runB> <seed> <teacher.npy> [recool]   copy with --holdout_in_train vs copy without"""
import sys, os, re, glob, json, numpy as np, polars as pl, torch
R = "<repo>"; RV = f"{R}/nn_x/w12/stale/fulldata_review/runs"
mode, ta, tb = sys.argv[1], sys.argv[2], sys.argv[3]
A, B = f"{RV}/{ta}", f"{RV}/{tb}"
FAIL = []
def chk(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok: FAIL.append(name)
def logl(d):
    f = glob.glob(f"{d}/logs/*.log"); assert len(f) == 1
    L = open(f[0]).read().splitlines()
    return [re.sub(r"\d+\.\d+ min", "T min", re.sub(r"; \d+s$", "; Ts", l)) for l in L]
ra = np.load(f"{A}/rec.npz"); rb = np.load(f"{B}/rec.npz") if tb != "-" else None
assert bool(ra["ok"]) and (rb is None or bool(rb["ok"]))
def ck(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/ckpt/*.pt")):
        out[os.path.basename(f)] = torch.load(f, map_location="cpu", weights_only=False)
    return out
def files(d):
    return sorted(os.path.relpath(f, d) for f in glob.glob(f"{d}/preds/**/*.npy", recursive=True))
if mode == "identity":
    la, lb = logl(A), logl(B)
    a0 = la[0].replace(", 'holdout_in_train': False", ""); b0 = lb[0].replace(", 'holdout_in_train': False", "")
    a0, b0 = re.sub(r"==== .*? args", "==== args", a0), re.sub(r"==== .*? args", "==== args", b0)
    chk("log line 0 (args; the copy has the extra key holdout_in_train False)", a0 == b0 and "'holdout_in_train': False" in lb[0] and "holdout_in_train" not in la[0])
    chk(f"log lines 1..{len(la) - 1} identical", la[1:] == lb[1:], f"({len(la)} vs {len(lb)} lines)")
    for k in ("fidx", "flen", "fsplit", "fhz", "y", "yshape", "loss", "gn", "ev"):
        chk(f"record {k} identical", ra[k].shape == rb[k].shape and bool((ra[k] == rb[k]).all()), f"(n {ra[k].shape})")
    print(f"     training gathers {len(ra['flen'])}, loss first {ra['loss'][0]:.6f} last {ra['loss'][-1]:.6f}; eval calls {len(ra['ev'])}")
    ca, cb = ck(A), ck(B)
    chk("same checkpoint files", list(ca) == list(cb), str(list(ca)))
    for n in ca:
        sa, sb = ca[n]["state_dict"], cb[n]["state_dict"]
        same = list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) and sa[k].dtype == sb[k].dtype for k in sa)
        aa, ab = dict(ca[n]["args"]), dict(cb[n]["args"]); extra = ab.pop("holdout_in_train", "missing")
        chk(f"ckpt {n}: {len(sa)} tensors bit-identical, cfg equal, args equal up to holdout_in_train", same and ca[n]["cfg"] == cb[n]["cfg"] and aa == ab and extra is False)
    fa, fb = files(A), files(B)
    chk("same prediction file names", fa == fb, str(fa))
    for f in fa:
        x, y = np.load(f"{A}/{f}"), np.load(f"{B}/{f}")
        chk(f"{f} identical", x.shape == y.shape and x.dtype == y.dtype and bool((x == y).all()), f"({x.shape}, nonzero {int((x != 0).sum())})")
    ja = sorted(os.path.basename(f) for f in glob.glob(f"{A}/logs/*")); jb = sorted(os.path.basename(f) for f in glob.glob(f"{B}/logs/*"))
    chk("same log-dir file names", ja == jb, str(ja))
else:
    seed, teacher = int(sys.argv[4]), sys.argv[5]; recool = len(sys.argv) > 6
    f = pl.read_parquet(f"{R}/data/flat/train/feeds.parquet"); Mt = f["M"].to_numpy().astype(np.int64); offt = f["cand_off"].to_numpy().astype(np.int64); n = len(Mt)
    p = np.nan_to_num(np.load(f"{R}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0); p[Mt < 50] = 0.0
    cand = np.where((p > 0.9) & (Mt >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    def batches(ids, rng):   # my re-implementation of the bucketed batching
        Mp = (Mt[ids] + 15) // 16 * 16; out = []
        for mp in np.unique(Mp):
            x = rng.permutation(ids[Mp == mp]); Bn = int(max(1, min(512, 32768 // mp))); out += [x[i:i + Bn] for i in range(0, len(x), Bn)]
        return [out[k] for k in rng.permutation(len(out))]
    def expected(with_hold):
        rng = np.random.default_rng(seed); feeds = np.sort(rng.choice(n, 20000, replace=False))
        if not with_hold: feeds = np.setdiff1d(feeds, hold)
        return feeds, batches(feeds, rng)
    def steps(r):
        o = np.concatenate([[0], np.cumsum(r["flen"])]); return [r["fidx"][o[i]:o[i + 1]] for i in range(len(r["flen"]))]
    sc = steps(ra); sb = steps(rb) if rb is not None else None          # C (flag on), B (flag off)
    fc, bc = expected(True); fb_, bb = expected(False)
    nh = int(np.isin(fc, hold).sum())
    if rb is not None:
      chk("flag-off run: 0 holdout feeds in any training batch", not np.isin(rb["fidx"], hold).any())
      chk("flag-off run: every step == my recomputed regular batch or a pool batch", all(np.array_equal(sb[i], bb[i]) or (p[sb[i]] > 0.5).all() for i in range(len(sb))) and len(sb) == len(bb), f"(steps {len(sb)}, expected {len(bb)})")
    chk(f"(c) steps: flag-on {len(sc)} == my recomputed count with the holdout {len(bc)}; without {len(bb)}", len(sc) == len(bc))
    t_cool = 0 if recool else int(round(0.8 * len(bc)))
    reg = [i for i in range(len(sc)) if np.array_equal(sc[i], bc[i])]; poolst = [i for i in range(len(sc)) if i not in reg]
    chk("flag-on: every non-regular step is at/after the cooldown start and holds only p_random > 0.5 feeds of the feed list", all(i >= t_cool and (p[sc[i]] > 0.5).all() and np.isin(sc[i], fc).all() for i in poolst), f"(regular {len(reg)}, pool {len(poolst)}, t_cool {t_cool})")
    inreg = np.concatenate([sc[i] for i in reg]); hreg = np.intersect1d(inreg, hold)
    chk(f"(a) holdout feeds of the 20,000 subsample: {nh}; in the regular batches of the flag-on run: {len(hreg)}", len(hreg) > 0 and (len(poolst) > 0 or len(hreg) == nh))
    allc = np.unique(ra["fidx"])
    chk("(d) the non-holdout feeds of the flag-on feed list == the flag-off feed list", np.array_equal(np.setdiff1d(fc, hold), fb_) and np.isin(allc, fc).all())
    if poolst:
        pf = np.concatenate([sc[i] for i in poolst]); print(f"     pool batches {len(poolst)}: {len(pf)} feed-occurrences, {int(np.isin(pf, hold).sum())} of them holdout feeds ({100 * np.isin(pf, hold).mean():.1f}%); pool share of holdout {100 * nh / int((p[fc] > 0.5).sum()):.1f}%")
        chk("(e) holdout feeds are drawn in pool batches", np.isin(pf, hold).any())
    # (b) distillation target on every feed-occurrence, recomputed from the label and teacher files
    T = np.load(teacher, mmap_mode="r"); Y = np.load(f"{R}/data/flat/train/cand_target.npy", mmap_mode="r")
    yo = np.concatenate([[0], np.cumsum(ra["yshape"][:, 0] * ra["yshape"][:, 1])]); mh = mo = 0.0; nho = 0
    assert len(ra["yshape"]) == len(sc)
    HOLDSET = set(hold.tolist())
    for i, fb in enumerate(sc):
        yy = ra["y"][yo[i]:yo[i + 1]].reshape(ra["yshape"][i])
        for j, fe in enumerate(fb):
            r = np.arange(offt[fe], offt[fe] + Mt[fe]); t = np.asarray(T[r], np.float64); lab = np.asarray(Y[r], np.float64)
            q = np.exp(t - t.max()); q /= q.sum(); exp_ = 0.5 * lab / max(lab.sum(), 1) + 0.5 * q * (lab.sum() > 0)
            d = float(np.abs(yy[j, :Mt[fe]] - exp_).max()); pad = float(np.abs(yy[j, Mt[fe]:]).max()) if yy.shape[1] > Mt[fe] else 0.0
            if int(fe) in HOLDSET:
                mh = max(mh, d, pad); nho += 1
            else:
                mo = max(mo, d, pad)
    chk(f"(b) target of {nho} holdout feed-occurrences == 0.5 y/sum(y) + 0.5 softmax(teacher file rows)", nho > 0 and mh < 1e-6, f"max abs diff {mh:.1e} (other feeds {mo:.1e})")
    lc = logl(A); txt = "\n".join(lc)
    ev = [l for l in lc if re.match(r"step \d+/\d+ ", l)]; ins = [l for l in lc if l.startswith("FULLDATA step")]
    chk("(d) every eval line shows holdout_ndcg nan and has its own IN-SAMPLE line", len(ev) == len(ins) > 0 and all("holdout_ndcg nan" in l for l in ev), f"({len(ev)} evals)")
    best = re.search(r"BEST step (\d+)", txt); chk("(d) BEST step == total steps", best is not None and int(best.group(1)) == len(sc), best.group(0) if best else "")
    c = ck(A); names = list(c); kept = [x for x in names if "@" not in x]; last = [x for x in names if "@last" in x]
    same = len(kept) == 1 and len(last) == 1 and all(torch.equal(c[kept[0]]["state_dict"][k], c[last[0]]["state_dict"][k]) for k in c[kept[0]]["state_dict"])
    chk("(d) kept checkpoint bit-identical to @last; args keep == 'last' and holdout_in_train True", same and c[kept[0]]["args"]["keep"] == "last" and c[kept[0]]["args"]["holdout_in_train"] is True, str(names))
    hf = sorted(os.path.basename(x) for x in glob.glob(f"{A}/preds/holdout/*"))
    chk("(d) holdout dir holds only *holdoutINSAMPLE* files (no out-of-sample name, no holdout_meta / holdout_y)", len(hf) >= 2 and all("holdoutINSAMPLE" in x for x in hf), str(hf))
    pfiles = sorted(os.path.basename(x) for x in glob.glob(f"{A}/preds/*.npy")); chk("val / test prediction files of the kept and @last checkpoints", len(pfiles) == 4, str(pfiles))
    js = glob.glob(f"{A}/logs/*_fulldata.json"); chk("fulldata json present", len(js) == 1)
    if js:
        j = json.load(open(js[0])); print("     fulldata.json:", {k: j[k] for k in ("kept_step", "total_steps", "n_holdout_feeds_in_training", "n_training_feeds", "pool_holdout_feeds", "holdout_feeds_per_epoch", "steps")})
    for l in lc:
        if l.startswith("FULLDATA") or "holdout:" in l or "cooldown mix" in l or "train feeds" in l or "BEST" in l: print("     LOG:", l[:230])
print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
