"""05b: CE explained by the head-of-feed examination term g(shown rank) alone (no end-of-feed term), and how a g-offset
redistributes the CE gradient on the holdout (random-like) feeds. Reuses 05's reconstruction."""
import sys, numpy as np
sys.path.insert(0, "<repo>/src")
exec(open("05_holdout_labels.py").read().split("# ---- holdout ----")[0].split("# ---- fit examination offsets")[0])
o5 = np.load(f"{OUT}/pool_pos_offset.npz"); gs, h = o5["g_rank"], o5["h_rkend"]
rh = rows(hold); yh = np.asarray(ctgt[rh]).astype(np.float64); posh = np.asarray(cpos[rh]).astype(np.int64)
Mh = M[hold]; offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); fid = np.repeat(np.arange(len(Mh)), Mh)
rkh, rkeh = within_rank(posh, Mh)
og = gs[np.minimum(rkh, 299)]
def sm(s):
    mx = np.maximum.reduceat(s, offh); e = np.exp(s - np.repeat(mx, Mh)); Z = np.add.reduceat(e, offh)
    return s - np.repeat(mx + np.log(Z), Mh)
K = np.add.reduceat(yh, offh)
def ce(s): return (-np.bincount(fid, weights=sm(s) * yh, minlength=len(Mh)) / K).mean()
MEM = [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"v28_lr1_m02_s41{i}" for i in range(1, 8)] + [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)]
c0 = cg = 0; negshift = []; posw = []
for mname in MEM:
    s = np.load(f"{PR}/holdout/nn_{mname}_holdout.npy").astype(np.float64)
    c0 += ce(s) / len(MEM); cg += ce(s + og) / len(MEM)
    p0 = np.exp(sm(s)); p1 = np.exp(sm(s + og))
    neg = yh == 0
    # share of negative CE-gradient mass on negatives shown in the top 10 slots, before/after the offset
    top = neg & (rkh < 10)
    negshift.append(((p0[top].sum() / p0[neg].sum()), (p1[top].sum() / p1[neg].sum())))
    # positive gradient (1/K - p_i) by shown rank bucket, before/after
    pos = yh == 1
    g0 = (1 / np.repeat(K, Mh) - p0)[pos]; g1 = (1 / np.repeat(K, Mh) - p1)[pos]
    posw.append([ (g1[(rkh[pos] >= a) & (rkh[pos] < b)].sum() / g1.sum()) / (g0[(rkh[pos] >= a) & (rkh[pos] < b)].sum() / g0.sum())
                 for a, b in [(0, 3), (3, 10), (10, 30), (30, 100), (100, 300)]])
print(f"holdout CE: uniform {np.log(Mh).mean():.4f}; g-only offset {ce(og):.4f}; member {c0:.4f}; member + g {cg:.4f}; "
      f"g removes {c0-cg:.4f} nats ({(c0-cg)/(np.log(Mh).mean()-c0)*100:.1f}% of the member's gain over uniform)")
ns = np.array(negshift).mean(0)
print(f"negative CE-gradient mass on negatives shown in the top 10 slots: {ns[0]*100:.1f}% -> {ns[1]*100:.1f}% with the g offset")
pw = np.array(posw).mean(0)
print("relative change of positive-gradient share by shown rank [0,3) [3,10) [10,30) [30,100) [100,300): " + " ".join(f"{x:.3f}" for x in pw))
