"""Label-free compatibility check of a merge partner's submission file against one of ours (prepared 2026-11-14; RECHECK_1114
skeptic_duplicates N4, long_bets bet 1). No labels, no leaderboard access: only the two test files and the two public scores.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python -I src/merge_check.py \
      --partner <dir>/partner.parquet --partner_public 0.3951 [--ours subs/cand_W8kd_r2.parquet --ours_public 0.394486] \
      [--blend subs/cand_MERGE_<tag>.parquet --w 0.5]

The partner file is UNTRUSTED data: keep it in its own directory outside src/, run with -I, never execute anything from it.
Checks: same (feed_id, item_id) keys as the sample submission, finite scores, ties.
Prints: within-feed correlation of the two files (per-feed rank-z; all feeds, M<50, M>=50), the frontier law on record
  gain(equal mix over the BETTER file) = 0.0186 (1 - rho) - 0.378 |public gap|     (long_bets F13: predicts 10 on-disk partners within 0.0002)
and the decision line of long_bets bet 1: blend only if law - 0.0005 >= 0; one public slot; kill if blend < better solo - 0.0018.
With --blend it writes (1 - w) z(ours) + w z(partner) through make_sub (w default 0.5; the weight is fixed by the law, never scanned on public);
the blended file still has to pass src/presub_gate.py as far as its checks apply (file validity; member audit covers only our half)."""
import sys, argparse
sys.path.insert(0, '<repo>/src')
from common import *
from stack_val import feed_rank_z
from make_sub import make_sub

p = argparse.ArgumentParser()
p.add_argument('--partner', required=True); p.add_argument('--partner_public', type=float, required=True)
p.add_argument('--ours', default='<repo>/subs/cand_W8kd_r2.parquet'); p.add_argument('--ours_public', type=float, default=0.394486)
p.add_argument('--blend', default=''); p.add_argument('--w', type=float, default=0.5)
a = p.parse_args()

f, M, off = feed_meta('test'); fidx = np.load(FLAT / 'test/cand_fidx.npy')
key = pl.DataFrame({'feed_id': f['feed_id'].to_numpy()[fidx].astype(np.int64), 'item_id': np.load(FLAT / 'test/cand_item_id.npy').astype(np.int64),
                    'row': np.arange(len(fidx), dtype=np.int64)})


def load(path, who):
    df = pl.read_parquet(path)
    assert {'feed_id', 'item_id', 'score'} <= set(df.columns), f'{who}: columns {df.columns}'
    df = df.select(pl.col('feed_id').cast(pl.Int64), pl.col('item_id').cast(pl.Int64), pl.col('score').cast(pl.Float64))
    assert len(df) == len(key), f'{who}: {len(df)} rows, expected {len(key)}'
    assert df.select('feed_id', 'item_id').is_unique().all(), f'{who}: duplicate (feed_id, item_id)'
    j = key.join(df, on=['feed_id', 'item_id'], how='left').sort('row')
    assert j['score'].null_count() == 0, f'{who}: {j["score"].null_count()} keys of the test set are missing'
    s = j['score'].to_numpy(); assert np.isfinite(s).all(), f'{who}: non-finite scores'
    ties = int(pl.DataFrame({'f': fidx, 's': s}).select(pl.struct('f', 's').is_duplicated().sum()).item())
    print(f'{who}: {path}: {len(s)} rows OK, tied rows within a feed {ties} ({ties / len(s):.2%}; ties are scored pessimistically by the metric)')
    return s


so, sp = load(a.ours, 'ours'), load(a.partner, 'partner')
zo = feed_rank_z(off, M, so.astype(np.float32))[:, 1]; zp = feed_rank_z(off, M, sp.astype(np.float32))[:, 1]
big = np.repeat(M >= 50, M); c = lambda m_: float(np.corrcoef(zo[m_], zp[m_])[0, 1])
rho = float(np.corrcoef(zo, zp)[0, 1]); gap = abs(a.ours_public - a.partner_public); law = 0.0186 * (1 - rho) - 0.378 * gap
# per-feed Spearman-like agreement of the top of the list: share of feeds with the same top-1 item
st = np.asarray(off, np.int64); top = lambda z: np.array([o + int(np.argmax(z[o:o + m])) for o, m in zip(st, M)])
t1 = float((top(zo) == top(zp)).mean())
print(f'within-feed correlation (rank-z): all {rho:.4f} | M>=50 {c(big):.4f} | M<50 {c(~big):.4f} | same top-1 item in {t1:.1%} of feeds')
print(f'reference points: our hard-label NN pools vs W8 0.984-0.994; CatBoost vs W8 about 0.90; another distilled pool of ours 0.999')
print(f'public: ours {a.ours_public:.6f}, partner {a.partner_public:.6f}, |gap| {gap:.6f}')
print(f'LAW: equal mix over the better file = 0.0186 * (1 - {rho:.4f}) - 0.378 * {gap:.6f} = {law:+.5f}'
      f'  -> predicted public of the mix {max(a.ours_public, a.partner_public) + law:.5f}')
print(f'DECISION (long_bets bet 1): {"BLEND (law - 0.0005 >= 0)" if law - 0.0005 >= 0 else "do NOT blend (law - 0.0005 < 0): keep the better solo file"}'
      f'; one public slot; kill if the blend scores below the better solo - 0.0018')
for g in (0.0, 0.001, 0.002): print(f'   sensitivity: at |gap| {g:.3f} the law gives {0.0186 * (1 - rho) - 0.378 * g:+.5f}')
if a.blend:
    make_sub(((1 - a.w) * zo + a.w * zp).astype(np.float32), a.blend); print(f'written {a.blend} = {1 - a.w:g} z(ours) + {a.w:g} z(partner)')
