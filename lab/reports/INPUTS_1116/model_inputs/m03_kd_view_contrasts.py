"""INVENTORY A, measurement M3 (CPU only, read-only): what the records can say about a student whose input VIEW differs
from its teacher's, from stored predictions.

Part A  held-out 10-epoch members (val 14,423 feeds + in-period holdout 30,000 feeds; selected and last-step checkpoints),
        4-pools of seeds 951-954 (per-feed z-mean), paired per feed:
          hard labels:   base (34 inputs) | xdrop_pop (26) | pop1 (26 + xp_imp_c50) | pop2 (26 + xp_ctr_c)
          distillation:  kd05 (d128, teacher_mean11) | kd05t25 (d128, teacher_mean25) | kd05w192 (d192, mean11)
                         pop1kd05t25 (d128, mean25, pop1 view) | pop1kd05w192 (d192, mean11, pop1 view)
        -> the gain from distillation in the full view and in the pop1 view; the view gap under hard labels and under
           distillation (teacher_mean11 never saw xp_imp_c50 and DID see the 8 popularity inputs the pop1 student lacks).
Part B  full-data wide students (val only; their holdout is in-sample), teacher_mean11, d192:
          pf (26 inputs) | pop1 (26 + xp_imp_c50) | pop2 (26 + xp_ctr_c)
        -> the trained-input value of ONE added column for a distilled student whose teacher never saw that column.
Member-level means are over seeds; pool-level deltas carry the per-feed paired SE (seed noise is NOT in that SE).
Writes out/m03_kd_view_contrasts.txt.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/nn_x/w12/recheck/forensics_skeptic")
from hx import *  # noqa: E402,F401

OUTD = R / "reports/INPUTS_1116/model_inputs/out"
A = R / "nn_x/abl/preds"
LOG = open(OUTD / "m03_kd_view_contrasts.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


ARMS = {  # tag -> (harness, variant stem, description)
    "HL base":        ("w5", "w5_e10_base", "hard labels, d128, 34 inputs"),
    "HL xdrop_pop":   ("abl", "abl_e10_xdrop_pop", "hard labels, d128, 26 inputs (8 popularity inputs removed)"),
    "HL pop1":        ("abl", "abl_e10_pop1", "hard labels, d128, 26 + xp_imp_c50"),
    "HL pop2":        ("abl", "abl_e10_pop2", "hard labels, d128, 26 + xp_ctr_c"),
    "KD kd05":        ("w5", "w5_e10_kd05", "distilled d128, teacher_mean11, 34 inputs"),
    "KD kd05t25":     ("w5", "w5_e10_kd05t25", "distilled d128, teacher_mean25, 34 inputs"),
    "KD kd05w192":    ("w5", "w5_e10_kd05w192", "distilled d192, teacher_mean11, 34 inputs"),
    "KD pop1 t25":    ("abl", "abl_e10_pop1kd05t25", "distilled d128, teacher_mean25, pop1 view"),
    "KD pop1 w192":   ("abl", "abl_e10_pop1kd05w192", "distilled d192, teacher_mean11, pop1 view"),
}
SEEDS = (951, 952, 953, 954)


def load(split, D, last):
    y, M, off = D["y"], D["M"], D["off"]
    P = {}
    for tag, (h, v, _) in ARMS.items():
        d = A if h == "abl" else W
        fs = []
        for s in SEEDS:
            f = (d / f"nn_{v}_s{s}{'@last' if last else ''}_trainonly_val.npy") if split == "val" else (d / f"holdout/nn_{v}_s{s}{'@last' if last else ''}_holdout.npy")
            if not f.exists():
                fs = None
                break
            fs.append(f)
        if fs is None:
            continue
        Z = [zfeed(np.load(f), off, M) for f in fs]
        mem = np.stack([ndcg_per_feed(z, y, off, M) for z in Z])
        pool = ndcg_per_feed(zfeed(np.mean(Z, 0), off, M), y, off, M)
        P[tag] = (mem, pool)
    return P


CONTR = [
    ("distillation gain, full view d128 (mean11)", "KD kd05", "HL base"),
    ("distillation gain, full view d128 (mean25)", "KD kd05t25", "HL base"),
    ("distillation gain, pop1 view d128 (mean25)", "KD pop1 t25", "HL pop1"),
    ("view gap full - pop1, hard labels d128", "HL base", "HL pop1"),
    ("view gap full - pop1, distilled d128 (mean25)", "KD kd05t25", "KD pop1 t25"),
    ("view gap full - pop1, distilled d192 (mean11)", "KD kd05w192", "KD pop1 w192"),
    ("ONE added column xp_imp_c50, hard labels (pop1 - xdrop_pop)", "HL pop1", "HL xdrop_pop"),
    ("ONE added column xp_ctr_c, hard labels (pop2 - xdrop_pop)", "HL pop2", "HL xdrop_pop"),
    ("8 popularity inputs, hard labels (base - xdrop_pop)", "HL base", "HL xdrop_pop"),
]
say("== Part A: held-out 10-epoch members, 4-pools of seeds 951-954")
for tag, (h, v, desc) in ARMS.items():
    say(f"   {tag:14s} = {v}_s951-954  ({desc})")
for split in ("val", "hold"):
    D = load_val() if split == "val" else load_hold()
    for last in (False, True):
        P = load(split, D, last)
        say(f"\n-- {split}, {'LAST-step' if last else 'SELECTED'} checkpoints ({len(D['M'])} feeds); arms found: {len(P)} of {len(ARMS)}")
        for tag in P:
            mem, pool = P[tag]
            say(f"   level {tag:14s} member mean {mem.mean():.5f} (seed sd {mem.mean(1).std(ddof=1):.5f})  4-pool {pool.mean():.5f}")
        for name, a, b in CONTR:
            if a not in P or b not in P:
                say(f"   {name}: missing arm")
                continue
            dm = P[a][0].mean(0) - P[b][0].mean(0)
            dp = P[a][1] - P[b][1]
            say(f"   {name:62s} member {fmt(dm)}   4-pool {fmt(dp)} (z {dp.mean() / se(dp):+.1f})")

# ---- Part B: full-data wide students of three popularity-light views (val only)
say("\n== Part B: full-data wide students (d192, teacher_mean11), val only")
D = load_val(); y, M, off = D["y"], D["M"], D["off"]


def fd(view, seeds, last=False):
    Z = [zfeed(np.load(A / f"nn_abl_fd10_{view}kd05w192_s{s}{'@last' if last else ''}_trainonly_val.npy"), off, M) for s in seeds]
    return np.stack([ndcg_per_feed(z, y, off, M) for z in Z]), ndcg_per_feed(zfeed(np.mean(Z, 0), off, M), y, off, M)


G1, G2 = (951, 952, 953, 954), (955, 956, 957, 958)
for name, va, sa, vb, sb in [
    ("xp_imp_c50 added (pop1 - pf), seeds 951-954", "pop1", G1, "pf", G1),
    ("xp_imp_c50 added (pop1 - pf), seeds 955-958", "pop1", G2, "pf", G2),
    ("xp_imp_c50 added (pop1 - pf), seeds 951-958 (8-pools)", "pop1", G1 + G2, "pf", G1 + G2),
    ("xp_ctr_c added (pop2 - pf), seeds 951-954", "pop2", G1, "pf", G1),
    ("xp_ctr_c vs xp_imp_c50 (pop2 - pop1), seeds 951-954", "pop2", G1, "pop1", G1),
    ("A/A of the pf view: pf 955-958 - pf 951-954 (seed-draw noise)", "pf", G2, "pf", G1),
    ("A/A of the pop1 view: pop1 955-958 - pop1 951-954", "pop1", G2, "pop1", G1),
    ("A/A of the pop1 view: pop1 959-962 - pop1 951-954", "pop1", (959, 960, 961, 962), "pop1", G1),
]:
    ma, pa = fd(va, sa); mb, pb = fd(vb, sb)
    dm = ma.mean(0) - mb.mean(0); dp = pa - pb
    say(f"   {name:62s} levels {pa.mean():.5f} / {pb.mean():.5f}  member {fmt(dm)}  pool {fmt(dp)} (z {dp.mean() / se(dp):+.1f})")
say("\n   per-seed val NDCG, full-data wide students:")
for view, seeds in (("pf", G1 + G2), ("pop1", G1 + G2 + (959, 960, 961, 962)), ("pop2", G1)):
    m, _ = fd(view, seeds)
    say(f"   {view:5s}", " ".join(f"{s}:{v:.5f}" for s, v in zip(seeds, m.mean(1))), f" mean {m.mean():.5f} sd {m.mean(1).std(ddof=1):.5f}")
