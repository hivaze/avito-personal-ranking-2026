"""(C, follow-up) Is the leader's +0.00305 on public inside board noise? Paired SE as a function of the per-feed
correlation rho between two files' NDCG (both with the best file's per-feed sd under the public / private mix):
paired sd = sd * sqrt(2 (1 - rho)). Observed rho on val: near copies 0.99, W8P20 0.976, G3 0.956, a CatBoost model 0.668
(out/c_evalnoise.txt). The leader's file is unknown; a strong model of another family is assumed to sit at rho 0.80-0.95.
"""
import sys
from pathlib import Path

from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

sd_pub = 0.00268 * np.sqrt(N_PUB)      # out/c_evalnoise.txt: SE of one file at the public size and mix
sd_priv = 0.00177 * np.sqrt(N_PRIV)
gap = 0.3982556 - 0.3952092
print(f"per-feed sd: public mix {sd_pub:.4f}, private mix {sd_priv:.4f}; leader gap on public {gap:+.5f}")
for rho in (0.99, 0.976, 0.956, 0.95, 0.90, 0.85, 0.80, 0.70):
    sp = sd_pub * np.sqrt(2 * (1 - rho)) / np.sqrt(N_PUB)
    sq = sd_priv * np.sqrt(2 * (1 - rho)) / np.sqrt(N_PRIV)
    print(f"  rho {rho:.3f}: paired SE public {sp:.5f}, private {sq:.5f}; gap z on public {gap / sp:4.1f}; "
          f"P(private order flips) if the true gap equals the public gap {norm.cdf(-gap / sq):.4f}, "
          f"if the public gap is one noisy draw (flat prior) {norm.cdf(-gap / np.hypot(sp, sq)):.4f}")
print("(ignores the leader's own selection among many public submissions, which inflates its public score)")
