"""Segment definitions shared by segment_tables.py and cv_tests.py."""
import numpy as np


def cut(x, edges, labels):
    x = np.asarray(x, dtype=np.float64)
    i = np.searchsorted(np.asarray(edges, dtype=np.float64), x, side="left")  # bins (e[i-1], e[i]]
    return np.array(labels, dtype=object)[i]


def seg_defs(t, test=False):
    S = {}
    S["M"] = cut(t["M"], [10, 25, 50, 100, 200], ["M5-10", "M11-25", "M26-50", "M51-100", "M101-200", "M201-300"])
    S["H"] = cut(t["H"], [0, 19, 63, 191, 255], ["H0", "H1-19", "H20-63", "H64-191", "H192-255", "H256"])
    u0 = t["uid0"].to_numpy(); ntr = t["n_train_feeds"].to_numpy()
    S["user"] = np.where(u0, "uid0", cut(ntr, [0, 1, 4], ["tr0", "tr1", "tr2-4", "tr5+"]))
    S["cold_share"] = cut(t["cold_share"], [0.15, 0.3, 0.45], ["cold<15%", "cold15-30%", "cold30-45%", "cold>=45%"])
    S["hour"] = cut(t["hour"], [5, 9, 13, 17], ["h00-05", "h06-09", "h10-13", "h14-17", "h18-23"])
    S["n_cat"] = cut(t["n_cat"], [4, 8, 12, 18], ["ncat1-4", "ncat5-8", "ncat9-12", "ncat13-18", "ncat19+"])
    S["hist_top_cat_share"] = cut(t["hist_top_cat_share"], [0.0, 0.33, 0.5, 0.7],
                                  ["htop=0(H0)", "htop<=.33", "htop.33-.5", "htop.5-.7", "htop>.7"])
    S["cand_cat_in_hist"] = cut(t["cand_cat_in_hist_share"], [0.5, 0.8, 0.95],
                                ["cih<=.5", "cih.5-.8", "cih.8-.95", "cih>.95"])
    if not test:
        S["day"] = np.where(t["day"].to_numpy() == 0, "Tue", "Thu")
        S["rho_nn_cb"] = cut(t["rho_nn_cb"], [0.7, 0.8, 0.9], ["rho<=.7", "rho.7-.8", "rho.8-.9", "rho>.9"])
        S["npos"] = cut(t["npos"], [1, 2, 3, 5], ["npos1", "npos2", "npos3", "npos4-5", "npos6+"])
        S["n_pos_cat"] = cut(t["n_pos_cat"], [1, 2], ["1poscat", "2poscat", "3+poscat"])
        pc = t["pos_cat"].to_numpy()
        top = [13, 2, 3, 50, 6, 7, 10, 15, 11, 48]
        S["pos_cat"] = np.where(pc < 0, "multi", np.where(np.isin(pc, top), np.char.add("cat", pc.astype(str)), "cat_other"))
        S["pos_cat_in_hist"] = np.where(t["pos_cat_in_hist"].to_numpy(), "poscat_in_hist", "poscat_NOT_in_hist")
        S["pos_cat_hist_share"] = cut(t["pos_cat_hist_share"], [0.0, 0.05, 0.2, 0.5],
                                      ["phs=0", "phs<=.05", "phs.05-.2", "phs.2-.5", "phs>.5"])
        S["pos_f3_catrank"] = cut(t["pos_f3_catrank"], [0, 1, 2, 5], ["f3cat#1", "f3cat#2", "f3cat#3", "f3cat#4-6", "f3cat#7+"])
        S["pos_f3_pct"] = cut(t["pos_f3_pct"], [0.0, 0.1, 0.3, 0.6], ["f3top", "f3<=10%", "f3 10-30%", "f3 30-60%", "f3>60%"])
        S["pos_f1_pct"] = cut(t["pos_f1_pct"], [0.0, 0.1, 0.3, 0.6], ["f1top", "f1<=10%", "f1 10-30%", "f1 30-60%", "f1>60%"])
        S["pos_cold"] = np.where(t["pos_cold_all"].to_numpy(), "pos_all_cold",
                                 np.where(t["pos_cold_any"].to_numpy(), "pos_some_cold", "pos_warm"))
        S["pos_cat_share_feed"] = cut(t["pos_cat_share_feed"], [0.05, 0.15, 0.3, 0.6],
                                      ["pcs<=5%", "pcs5-15%", "pcs15-30%", "pcs30-60%", "pcs>60%"])
    else:
        S["day"] = np.full(len(t), "test")
    return S
