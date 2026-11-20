"""Information: 2-epoch training trajectories (val / holdout NDCG per logged eval) of w10_<arm>_s23/24 vs the reference w9_base_s23/24 (nn_x/w9/dw3)."""
import json, sys
arms = sys.argv[1:] or ["r3", "r3w20"]
B = {s: {h["step"]: h for h in json.load(open(f"<repo>/nn_x/w9/dw3/logs/w9_base_s{s}.json"))["history"]} for s in (23, 24)}
for arm in arms:
    for s in (23, 24):
        try:
            j = json.load(open(f"<repo>/nn_x/w10/r3/logs/w10_{arm}_s{s}.json"))
        except FileNotFoundError:
            print(arm, s, "missing"); continue
        H = {h["step"]: h for h in j["history"]}
        print(f"{arm} s{s} ({j['train_minutes']:.1f} min, best {j['best_step']}): step: d_val / d_hold (arm - base)")
        print("   " + " ".join(f"{k}:{H[k]['val_ndcg'] - B[s][k]['val_ndcg']:+.4f}/{H[k]['holdout_ndcg'] - B[s][k]['holdout_ndcg']:+.4f}" for k in sorted(H) if k in B[s]))
        print("   loss (arm vs base): " + " ".join(f"{k}:{H[k]['loss']:.3f}/{B[s][k]['loss']:.3f}" for k in sorted(H) if k in B[s] and k % 4000 == 0 or k == 19670))
