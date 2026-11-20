"""Step 1: rebuild per-feed val NDCG, holdout NDCG (train-only pool) and test final scores for every file; verify vs records.
Writes out/nd_val.npz (names, nd [n_files x 14423]), out/nd_hold.npz, out/test_z.npy (float16 per file), out/s1_build.log."""
from lib import *
import io, time
buf = io.StringIO()


def say(*a):
    print(*a, flush=True); print(*a, file=buf)


t0 = time.time()
names = MODERN + OLDARR + UNSCORED
fv, Mv, ov = meta("val"); ft, Mt, ot = meta("test")
ND = {}; TZ = {}; HALF = {}
for n in names:
    sv = final_score(n, "val"); ND[n] = nd_val(sv)
    rec = VAL_REC.get(n)
    say(f"{n:7s} val {ND[n].mean():.5f}" + (f"  recorded {rec:.5f}  diff {ND[n].mean() - rec:+.5f}" if rec else "  (unscored)") + (f"  public {PUBLIC[n]:.6f}" if n in PUBLIC else ""))
    st = final_score(n, "test")
    # per-feed standardise the test score (final is a sum of two z's; old pools arbitrary scale) for correlation work
    TZ[n] = zf("test", st).astype(np.float32)
    if n in ENS:
        HALF[n] = (nd_val(half_score(n, "ft", "val")), nd_val(half_score(n, "to", "val")))
        say(f"        halves: FT {HALF[n][0].mean():.5f}  TO {HALF[n][1].mean():.5f}")
np.savez(OUT / "nd_val.npz", names=np.array(names), nd=np.stack([ND[n] for n in names]))
np.savez(OUT / "nd_val_halves.npz", names=np.array(list(HALF)), ft=np.stack([HALF[n][0] for n in HALF]), to=np.stack([HALF[n][1] for n in HALF]))
np.save(OUT / "test_z.npy", np.stack([TZ[n] for n in names]))
# holdout
HN = {}
for n, ms in HOLD_MEMBERS.items():
    HN[n], _ = hold_pool_nd(ms)
    say(f"holdout {n:7s} {HN[n].mean():.5f}  ({len(ms)} members)")
np.savez(OUT / "nd_hold.npz", names=np.array(list(HN)), nd=np.stack([HN[n] for n in HN]))
# pairwise corr on test: flat corr of per-feed-standardised final scores == candidate-weighted mean within-feed Pearson
T = np.stack([TZ[n] for n in names]).astype(np.float64)
pub, day, _ = test_parts()
rows_pub = np.repeat(pub, Mt)
C = np.corrcoef(T); Cp = np.corrcoef(T[:, rows_pub]); Cq = np.corrcoef(T[:, ~rows_pub])
np.savez(OUT / "corr_test.npz", names=np.array(names), all=C, public=Cp, private=Cq)
say("\ncorr to C3 on test (all / public / private):")
i3 = names.index("C3")
for i, n in enumerate(names):
    say(f"  {n:7s} {C[i, i3]:.4f} {Cp[i, i3]:.4f} {Cq[i, i3]:.4f}   to G3 {C[i, names.index('G3')]:.4f}  to W8 {C[i, names.index('W8')]:.4f}")
say(f"done {time.time() - t0:.0f}s")
(OUT / "s1_build.log").write_text(buf.getvalue())
