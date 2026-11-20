"""Label-free cluster transition counts from history sequences (all splits, transductive, identical for val and test).
For every pair of consecutive valid history events (older a -> newer b) count T[a, b]; separately T1 where the newer event
has event_type 1 (the event type closest to the target action). Train feeds are weighted 1 / (#train feeds of the user)
so heavy users with many near-identical histories do not dominate (user_id 0 rows weight 1).
Also per-day cluster demand (share among history events with hpos < 10) and supply (share among candidates) for 'dsr'.
Outputs: data/clu_work/pmi_{K}.npz (T, T1), dsr_{K}.npz (dem [24, K], sup [24, K]) indexed by day_index (0..23)."""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *

NDAY = 24


@nb.njit(parallel=True, cache=True)
def trans_accum(ho, H, hcl, het, fw, K, nth):
    T = np.zeros((nth, K * K), np.float64)
    T1 = np.zeros((nth, K * K), np.float64)
    nf = len(H)
    for t in nb.prange(nth):
        for f in range(t * nf // nth, (t + 1) * nf // nth):
            o = ho[f]
            w = fw[f]
            prev = -1  # newer valid event cluster
            prev_e1 = False
            for j in range(H[f]):
                c = hcl[o + j]
                if c < 0:
                    continue
                if prev >= 0:
                    T[t, c * K + prev] += w
                    if prev_e1:
                        T1[t, c * K + prev] += w
                prev = c
                prev_e1 = het[o + j] == 1
    Ts = np.zeros(K * K); T1s = np.zeros(K * K)
    for t in range(nth):
        Ts += T[t]; T1s += T1[t]
    return Ts, T1s


def main():
    nb.set_num_threads(24)
    for K in (256, 1024):
        T = np.zeros(K * K); T1 = np.zeros(K * K)
        dem = np.zeros((NDAY, K)); sup = np.zeros((NDAY, K))
        for split in ["train", "val", "test"]:
            fe = feeds(split)
            if split == "train":
                u = fe["uid"]
                cnt = np.bincount(u)
                fw = np.where(u > 0, 1.0 / cnt[u], 1.0)
            else:
                fw = np.ones(len(u := fe["uid"]))
            hcl = np.asarray(hist_cl(split, K))
            het = np.load(FLAT / split / "hist_event_type.npy")
            t, t1 = trans_accum(fe["ho"], fe["H"], hcl, het, fw, K, 24)
            T += t; T1 += t1
            # demand / supply per day
            dayf = day_index(fe["ts"])
            hf = np.load(FLAT / split / "hist_fidx.npy")
            hp = np.load(FLAT / split / "hist_hpos.npy")
            ok = (hp < 10) & (hcl >= 0)
            dem += np.bincount(dayf[hf[ok]] * K + hcl[ok], minlength=NDAY * K).reshape(NDAY, K)
            ccl = np.asarray(cand_cl(split, K))
            cf = np.load(FLAT / split / "cand_fidx.npy")
            ok = ccl >= 0
            sup += np.bincount(dayf[cf[ok]] * K + ccl[ok], minlength=NDAY * K).reshape(NDAY, K)
            log("pmi/dsr", K, split)
        np.savez(WORK / f"pmi_{K}.npz", T=T.reshape(K, K), T1=T1.reshape(K, K))
        np.savez(WORK / f"dsr_{K}.npz", dem=dem, sup=sup)


if __name__ == "__main__":
    main()
