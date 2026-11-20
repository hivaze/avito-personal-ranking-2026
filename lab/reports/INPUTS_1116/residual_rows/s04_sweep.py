"""INVENTORY B, step 4: the uniform lens (lens.py) on every candidate column that has arrays on disk.
Sources:
  store  = data/feats + data/feats_new columns that exist for train AND val (holdout rows are sliced out of the train array)
  gate   = per-candidate arrays left by the residual / proxy gates (val + holdout, or val only)
Usage: s04_sweep.py val|hold TO16|FILE [store|gate|all]     Output: out/sweep_<split>_<base>.jsonl (one json line per column; resumable)
CPU only, 12 threads, read-only."""
import sys, json, time, os
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
from lens import *

sp, base = sys.argv[1], sys.argv[2]; what = sys.argv[3] if len(sys.argv) > 3 else "all"; t0 = time.time()
CCB = ROOT / "nn_x/insights/ccb_1113/scratch"; CB = ROOT / "nn_x/w12/cb/famB/feat"


def specs():
    S = []
    if what in ("store", "all"):
        for d in ("data/feats", "data/feats_new"):
            tr = {f[:-4] for f in os.listdir(ROOT / d / "train") if f.endswith(".npy")}; va = {f[:-4] for f in os.listdir(ROOT / d / "val") if f.endswith(".npy")}
            for n in sorted(tr & va):
                S.append(("store", n, dict(val=str(ROOT / d / "val" / f"{n}.npy"), hold="TRAIN:" + str(ROOT / d / "train" / f"{n}.npy"))))
    if what in ("gate", "all"):
        g = lambda grp, name, v, h=None: S.append((grp, name, dict(val=str(v), **({"hold": str(h)} if h else {}))))
        for n in "c01_score c04_nb_loglift c04_seq_cv_fwd5 c04_seq_cv_seqs c04_seq_cv_t1 c04_seq_cv_w5 c07_exhaust c07_it_n7 c07_it_q95gap c07_it_slope7 c07_it_trend c13_p1 c13_p2 c13_p3".split():
            g("ccb", n, CCB / f"tester1/out/{n}_val.npy", CCB / f"tester1/out/{n}_hold.npy")
        for item, keys in (("c03", "g_cmatch g_f2res g_shift loc16 mode6 mode96"), ("c06", "theta_l3 seen_l3"), ("c09", "p3a rp3b05 nseq")):
            for k in keys.split():
                g("ccb", f"{item}_{k}", f"{CCB}/tester3/cache/{item}_val.npz::{k}", f"{CCB}/tester3/cache/{item}_hold.npz::{k}")
        g("ccb", "c12_kappaB", f"{CCB}/tester3/cache/c12_cols.npz::kB_v", f"{CCB}/tester3/cache/c12_cols.npz::kB_h")
        for item, k in (("c02", 15), ("c08", 6), ("c14", 2), ("covis", 11)):
            for j in range(k):
                g("ccb", f"{item}_col{j}", f"{CCB}/tester2/feats/{item}_val.npy#{j}", f"{CCB}/tester2/feats/{item}_hold.npy#{j}")
        TG = ROOT / "nn_x/insights/timegap_1113/feats"
        for k in "gap_first gap_first_any gap_last gap_last_any n_days_active n_days_active_any never_seen".split():
            g("timegap", "tg_" + k, f"{TG}/val.npz::{k}", f"{TG}/hold.npz::{k}")
        EN = ROOT / "nn_x/insights/enrich_1113/feats"
        for k in "s2 s2r s4 comp minh home age".split():
            g("enrich", "en_" + k, f"{EN}/val.npz::{k}", f"{EN}/hold.npz::{k}")
        FU = ROOT / "nn_x/codes/followup/out"
        for k in "rp_cnt rp_lift s4_link s4_link_any s4_sum s4_sym".split():
            g("codes", "fu_" + k, FU / f"val_{k}.npy", FU / f"hold_{k}.npy")
        EX = ROOT / "nn_x/codes/exploit"
        g("codes", "ex_Y1_feedappeal", EX / "03_feedappeal/val_Y1.npy", EX / "03_feedappeal/hold_Y1.npy")
        g("codes", "ex_X5_novelty", EX / "04_novelty/val_X5.npy", EX / "04_novelty/hold_X5.npy"); g("codes", "ex_Y3_novgate_cos", EX / "04_novelty/val_Y3.npy", EX / "04_novelty/hold_Y3.npy")
        for k in "X2 X3 n_s1 n_s2".split():
            g("codes", f"ex_{k}_s1s2_lf", f"{EX}/02_s1s2contrast/feats_val.npz::val_lf_{k}", f"{EX}/02_s1s2contrast/feats_hold.npz::{k}")
        P2 = ROOT / "nn_x/pretrain2"
        g("pretrain2", "p2_coscr", P2 / "P2/out/val_coscr.npy", P2 / "P2/out/hold_coscr.npy")
        for k in "s_pos_item s_neg_item contrast_item s_pos_cl s_neg_cl contrast_cl".split():
            g("pretrain2", "p4_" + k, f"{P2}/P4/signals_val.npz::{k}", f"{P2}/P4/signals_holdout.npz::{k}")
        for blk in "abcd":
            for j, n in enumerate(json.load(open(CB / f"{blk}_names.json"))):
                g("cbB_" + blk, n, f"{CB}/{blk}_val.npy#{j}", f"{CB}/{blk}_hold.npy#{j}")
        g("itemprior", "ip_x_val_T", ROOT / "nn_x/audit/itemprior/x_val_T.npy")
        EU = ROOT / "nn_x/emb/usability/out/sig_val.npz"
        for k in "proj_mu proj_pc1 chg_d chg_flag in_train nd_act nd_view_only maxcos_act maxcos_view zca_maxcos cen_maxcos abtt3_maxcos zca_new5 cen_new5 abtt3_new5 cos_tv".split():
            g("emb", "emb_" + k, f"{EU}::{k}")
        LG = ROOT / "nn_x/w12/lit/int_gnn"
        for j in range(3):
            g("lit_gnn", f"walk_all_c{j}", f"{LG}/walk_all_val.npy#{j}"); g("lit_gnn", f"walk_train_c{j}", f"{LG}/walk_train_val.npy#{j}")
    return S


_cache = {}
def load(src, D):
    tr = src.startswith("TRAIN:")
    if tr: src = src[6:]
    j = None
    if "#" in src: src, j = src.split("#"); j = int(j)
    key = None
    if "::" in src: src, key = src.split("::")
    if key is not None:
        if src not in _cache:
            _cache.clear(); _cache[src] = np.load(src)
        a = _cache[src][key]
    else:
        a = np.load(src, mmap_mode="r")
    if tr: a = a[D["rows"]]
    if j is not None: a = a[:, j]
    a = np.asarray(a, np.float64)
    assert len(a) == len(D["y"]), (src, a.shape)
    return a


D = load_split(sp, base)
print(f"[{time.time() - t0:.0f}s] {sp} / base {base}: {D['n']} feeds, {len(D['y'])} rows, base NDCG {D['ndcg0'].mean():.5f}", flush=True)
if sp == "val":
    import pmix
    D["wpriv"] = pmix.weights()["private"]
outp = OUT / f"sweep_{sp}_{base}.jsonl"
done = {j["key"] for j in map(json.loads, open(outp)) if "error" not in j} if outp.exists() else set()  # errored columns are retried
S = [s for s in specs() if sp in s[2]]
fo = open(outp, "a"); n = 0
for grp, name, src in S:
    key = f"{grp}:{name}"
    if key in done: continue
    try:
        x = load(src[sp], D); r = read_column(D, x=x)
    except Exception as e:
        r = dict(error=repr(e)[:200])
    r = {k: (float(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in r.items()}
    r.update(key=key, grp=grp, name=name, src=src[sp])
    fo.write(json.dumps(r) + "\n"); fo.flush(); n += 1
    if n % 20 == 0: print(f"   [{time.time() - t0:.0f}s] {n} columns; last {key} LR_z {r.get('LR_z', float('nan')):.1f}", flush=True)
print(f"[{time.time() - t0:.0f}s] done: {n} new columns, {len(S)} specs", flush=True)
