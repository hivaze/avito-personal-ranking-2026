"""Frozen evaluation of FT-study trials on the CONFIRM half (set A) or on all val (set B).

Usage:
  python confirm.py A <trial> [<trial> ...]      CONFIRM-feed pool delta vs n2cpv3 (+ slices), set A
  python confirm.py B <dir_tag> <outdir>         all-val pool delta vs n2cpv3 for set B (files nn_<m>_<tag>_val_oof.npy)
  python confirm.py Aall                         CONFIRM numbers of every finished trial (report only, after freeze)
"""
import json
import sys

import numpy as np

from ftlib import FT_DIR, SET_A, SET_B, Val, load_base_ft, load_to


def evaluate(v, members, ft, mask, name):
    to_z = v.zmean([load_to(m) for m in members])
    base = [load_base_ft(m) for m in members]
    nb = v.nd(v.pool(base, to_z))
    res = {}
    for w in (0.5, 0.3, 0.7):
        d = v.nd(v.pool(ft, to_z, w)) - nb
        res[f"w{w}"] = v.stat(d, mask)
    d = v.nd(v.pool(ft, to_z)) - nb
    res["slices"] = {k: v.stat(d, mask & s) for k, s in v.slices.items() if k != "all"}
    res["pool_vs_TOpool"] = v.stat(v.nd(v.pool(ft, to_z)) - v.nd(to_z), mask)
    res["base_vs_TOpool"] = v.stat(nb - v.nd(to_z), mask)
    res["single_mix"] = {}
    for m, f, b in zip(members, ft, base):
        t = load_to(m)
        dm = v.nd(0.5 * v.z(f) + 0.5 * v.z(t)) - v.nd(0.5 * v.z(b) + 0.5 * v.z(t))
        res["single_mix"][m] = v.stat(dm, mask)
    mu, se, sec, n = res["w0.5"]
    print(f"{name}: pool delta {mu:+.5f} +- {se:.5f} (clustered {sec:.5f}, n {n}, z {mu / se:+.2f})")
    for k, (a, b, c, nn) in res["slices"].items():
        print(f"   slice {k:6s} {a:+.5f} +- {b:.5f} (z {a / b:+.2f}, n {nn})")
    print(f"   w_ft 0.3 {res['w0.3'][0]:+.5f} +- {res['w0.3'][1]:.5f} | w_ft 0.7 {res['w0.7'][0]:+.5f} +- {res['w0.7'][1]:.5f}")
    print(f"   pool vs TO-only pool {res['pool_vs_TOpool'][0]:+.5f} (n2cpv3 pool vs TO-only {res['base_vs_TOpool'][0]:+.5f})")
    for m, s in res["single_mix"].items():
        print(f"   single mix {m}: {s[0]:+.5f} +- {s[1]:.5f}")
    return res


def main():
    v = Val()
    mode = sys.argv[1]
    if mode in ("A", "Aall"):
        if mode == "A":
            trials = [int(x) for x in sys.argv[2:]]
        else:
            trials = sorted(int(json.loads(l)["trial"]) for l in open(FT_DIR / "trials.jsonl"))
        out = {}
        for t in trials:
            ft = [np.load(FT_DIR / "out" / f"t{t}" / f"nn_{m}_t{t}_val_oof.npy").astype(np.float64) for m in SET_A]
            out[t] = dict(confirm=evaluate(v, SET_A, ft, v.confirm, f"trial {t} CONFIRM"),
                          tune=evaluate(v, SET_A, ft, v.tune, f"trial {t} TUNE (check)"))
        json.dump(out, open(FT_DIR / f"confirm_{mode}_{'_'.join(map(str, trials)) if mode == 'A' else 'all'}.json",
                            "w"), indent=1)
    elif mode == "B":
        tag, od = sys.argv[2], sys.argv[3]
        from pathlib import Path
        ft = [np.load(Path(od) / f"nn_{m}_{tag}_val_oof.npy").astype(np.float64) for m in SET_B]
        res = dict(all=evaluate(v, SET_B, ft, np.ones(v.n, bool), f"set B {tag} ALL val"),
                   tune=evaluate(v, SET_B, ft, v.tune, f"set B {tag} TUNE half"),
                   confirm=evaluate(v, SET_B, ft, v.confirm, f"set B {tag} CONFIRM half"))
        json.dump(res, open(FT_DIR / f"confirm_B_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
