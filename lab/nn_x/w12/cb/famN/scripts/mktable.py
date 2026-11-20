"""Markdown tables for reports/CATBOOST_1115/famN.md from logs/results.jsonl + the paired reads of read.py. usage: python mktable.py > logs/tables.md"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fn  # noqa: E402
from fn import np  # noqa: E402
import read  # noqa: E402

WHAT = {
    "Yref": "reference: YetiRank, 177 features, has_time=True (time-ordered feeds, rows shuffled in the feed)",
    "Yref_nt": "control: YetiRank with CatBoost's default random permutations (has_time=False, file row order)",
    "Rref": "reference: RMSE on the centred out-of-fold teacher logit",
    "Lref": "reference: Logloss on clicks",
    "L_lda_pair": "N1 native: embedding column u*e (128-d), LDA:components=1, Logloss",
    "L_lda_ehm": "N1 native: embedding column e - mean history (128-d), LDA, Logloss",
    "L_lda_ue": "N1 native: embedding column [u ; e] (256-d), LDA, Logloss",
    "L_lda_all": "N1 native: the three embedding columns together, LDA, Logloss",
    "Y_def_pair32": "N1 native: embedding column u*e (32-d PCA), DEFAULT calcers (LDA + KNN:k=5), YetiRank",
    "Y_def_ehm32": "N1 native: e - mean history (32-d PCA), default calcers, YetiRank",
    "Y_def_ue32": "N1 native: [u ; e] (32-d PCA), default calcers, YetiRank",
    "Y_def_all32": "N1 native: the three vectors (32-d PCA each) as three embedding columns, default calcers (LDA + KNN:k=5), YetiRank",
    "Y_def_pair": "N1 native: u*e (128-d), default calcers, YetiRank",
    "Y_lda_all": "N1 native: three embedding columns, LDA only, YetiRank",
    "R_knn5_pair": "N1 native: u*e (32-d PCA), KNN:k=5 on the teacher target, RMSE",
    "R_knn50_pair": "N1 native: u*e (32-d PCA), KNN:k=50 on the teacher target, RMSE",
    "R_knn200_pair": "N1 native: u*e (32-d PCA), KNN:k=200 on the teacher target, RMSE",
    "R_knn50_ehm": "N1 native: e - mean history (32-d PCA), KNN:k=50, RMSE", "R_knn50_ue": "N1 native: [u ; e] (32-d PCA), KNN:k=50, RMSE",
    "Y_txt_s": "N2 native: text columns t_cat + t_clu (token sets), BoW, dictionary 500 tokens / column (lower bound 50), YetiRank",
    "Y_txt_l": "N2 native: same, dictionary 2000 tokens / column (lower bound 20), YetiRank",
    "R_txt_s": "N2 native: text columns, 500 tokens / column, RMSE on the teacher",
    "R_txt_l": "N2 native: text columns, 2000 tokens / column, RMSE on the teacher",
    "Y_q1024": "N3: per_float_feature_quantization border_count=1024 on it_hist_et1_luo and pre2_fe_z, YetiRank",
    "Y_oh64": "N3: one_hot_max_size=64 (category_id one-hot instead of CTR), YetiRank",
    "Y_sel": "N3: the 90 features kept by select_features (RecursiveByShapValues), YetiRank",
    "Y_hlda": "N1x hand-made: 3 numeric columns = out-of-fold Fisher direction on u*e, e - hmean, [u ; e], YetiRank",
    "Y_hknn": "N1x hand-made: 2 numeric columns = mean teacher_oof of the 50 / 200 nearest fit rows (u*e, 32-d), YetiRank",
    "Y_hknn3": "N1x hand-made: 6 numeric columns = KNN teacher mean (k 50 / 200) for the three vectors, YetiRank",
    "Y_hall": "N1x hand-made: 3 LDA columns + 6 KNN teacher-mean columns, YetiRank",
    "Y_hx": "N1x hand-made: 3 LDA columns + 2 KNN columns (u*e), YetiRank",
    "R_hknn": "N1x hand-made: 2 KNN teacher-mean columns (u*e), RMSE on the teacher",
    "R_hknn3": "N1x hand-made: 6 KNN teacher-mean columns, RMSE on the teacher",
}
nk = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"
res = read.results()
arms = {}
for t, d in res.items():
    if d["nfit"] // 1000 == nk and not d.get("dry"):
        arms.setdefault(d["arm"], []).append(d["seed"])
cmp_ = {}
p = fn.LOGS / f"compare_n{nk}k.json"
if p.exists():
    for c in json.load(open(p)):
        cmp_[c["arm"]] = c
order = [a for a in WHAT if a in arms] + [a for a in arms if a not in WHAT]
if MODE in ("arms", "both"):
    print(f"### Scale n{nk}k: one row per arm (mean of its seeds); d = paired per-feed difference to the same-loss reference (seed-paired)\n")
    print("| arm | what | seeds | trees | fit min / seed | model MB | level val / holdout | rho | NATS val / holdout (seed mean; single-seed SE) | d level val / holdout | d NATS val / holdout | winner (>= 2 SE on both) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in order:
        ds = [res[f"{a}_n{nk}k_s{s}"] for s in sorted(arms[a])]
        m = lambda k: float(np.mean([d[k] for d in ds]))
        c = cmp_.get(a)
        dl = dn = win = "reference" if a in ("Yref", "Rref", "Lref") else "n/a"
        if c:
            dl = f"{c['val_dlevel']:+.5f} ± {c['val_dlevel_se']:.5f} / {c['hold_dlevel']:+.5f} ± {c['hold_dlevel_se']:.5f}"
            dn = f"{c['val_dnats']:+.5f} ± {c['val_dnats_se']:.5f} / {c['hold_dnats']:+.5f} ± {c['hold_dnats_se']:.5f}"
            win = ("**YES**" if c["winner"] else "no") + f" (vs {c['ref']}, seeds {c['seeds']})"
        print(f"| {a} | {WHAT.get(a, '')} | {sorted(arms[a])} | {'/'.join(str(d['trees']) for d in ds)} | {m('fit_min'):.1f} | {m('model_mb'):.1f} | {m('val'):.5f} / {m('hold'):.5f} | "
              f"{m('rho_val'):.3f} / {m('rho_hold'):.3f} | {m('nats_val'):+.5f} ± {m('nats_val_se'):.5f} / {m('nats_hold'):+.5f} ± {m('nats_hold_se'):.5f} | {dl} | {dn} | {win} |")
if MODE in ("seeds", "both"):
    print(f"\n### Scale n{nk}k: the nats.py line of every (arm, seed)\n")
    print("| tag | trees | fit min | s / iter | model MB | level val / holdout | rho | NATS val | NATS holdout | PROMISING (>= 0.0016 both) | member rule 0.9 / 0.1: val, holdout, pooled z | importance of the new columns (PredictionValuesChange, %) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in order:
        for s in sorted(arms[a]):
            d = res[f"{a}_n{nk}k_s{s}"]
            print(f"| {d['tag']} | {d['trees']} | {d['fit_min']:.1f} | {d['s_per_iter']:.2f} | {d['model_mb']:.1f} | {d['val']:.5f} / {d['hold']:.5f} | {d['rho_val']:.3f} / {d['rho_hold']:.3f} | "
                  f"{d['nats_val']:+.5f} ± {d['nats_val_se']:.5f} | {d['nats_hold']:+.5f} ± {d['nats_hold_se']:.5f} | {'PROMISING' if d['promising'] else 'no'} | "
                  f"{d['mix01_val']:+.5f} ± {d['mix01_val_se']:.5f}, {d['mix01_hold']:+.5f} ± {d['mix01_hold_se']:.5f}, z {d['pooled_z']:+.2f} {'PASS' if d['member_pass'] else 'no pass'} | "
                  f"{', '.join(f'{n} {v}' for n, v in d['new_feature_importance']) or '-'} |")
