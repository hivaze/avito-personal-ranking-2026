"""Branch combination: weight, uncertainty bonus |s_ft - s_to|, power-mean of exp scores."""
from util import *

df = load()
c = Ctx(df)
sft = df["s_ft"].to_numpy().astype(np.float64)
sto = df["s_to"].to_numpy().astype(np.float64)
s = c.base
d = np.abs(sft - sto)


def show(name, r):
    print(f"{name:28s} cf={r['cf']:+.5f} se={r['se']:.5f} z={r['z']:+.2f} chosen={r['chosen']} ins_best={r['ins_best']} ins={r['ins_delta']:+.5f}", flush=True)


show("weight a*s_ft+(1-a)*s_to", c.crossfit(lambda a: a * sft + (1 - a) * sto, [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]))
show("score + w|sft-sto|", c.crossfit(lambda w: s + w * d, [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5]))
show("max/min mix", c.crossfit(lambda w: s + w * (np.maximum(sft, sto) - s), [-0.5, -0.3, 0.3, 0.5, 1.0]))


def powmean(k):
    m = np.maximum(sft, sto)
    return m + np.log(0.5 * np.exp(k * (sft - m)) + 0.5 * np.exp(k * (sto - m))) / k


show("power mean k", c.crossfit(powmean, [-4, -2, -1, 0.5, 1, 2, 4, 8]))
# per-feed: rank-based disagreement
rft = c.ranks(sft); rto = c.ranks(sto)
lr = np.abs(np.log(rft) - np.log(rto))
show("score + w*|log rank diff|", c.crossfit(lambda w: s + w * lr, [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]))
