"""2) Is history a snapshot at feed time? Consecutive feeds of the same user (train).
For each consecutive pair (A earlier, B later):
  - find shift k such that hist_B[k:k+L] == hist_A[0:L] (item_id & event_type), i.e. B = k new events + A
  - where do A's candidates (positives vs negatives) show up in B: in B's history (which event type, source,
    inside the new part?) or again as B candidates.
"""
import numpy as np
import polars as pl
import numba as nb
from common import feeds, arr

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
nb.set_num_threads(40)


@nb.njit(parallel=True, cache=True)
def pair_kernel(a_idx, b_idx, hoff, H, coff, M, hi, ht, hs, ci, ct,
                out_k, out_match, cand_start, c_in_bhist_type, c_in_bhist_src, c_in_bhist_pos, c_in_bnew,
                c_in_bcand, c_target, c_item_in_ahist):
    n = len(a_idx)
    for p in nb.prange(n):
        a = a_idx[p]; b = b_idx[p]
        ha = hoff[a]; la = H[a]; hb = hoff[b]; lb = H[b]
        # shift detection
        k = -1; best = 0.0
        if la > 0:
            for j in range(lb):
                if hi[hb + j] == hi[ha] and ht[hb + j] == ht[ha]:
                    L = min(la, lb - j)
                    ok = 0
                    for t in range(L):
                        if hi[hb + j + t] == hi[ha + t] and ht[hb + j + t] == ht[ha + t]:
                            ok += 1
                    r = ok / L
                    if r > best:
                        best = r; k = j
                    if r == 1.0:
                        break
        out_k[p] = k; out_match[p] = best
        # candidate-level
        ca = coff[a]; ma = M[a]; cb = coff[b]; mb = M[b]
        s = cand_start[p]
        for i in range(ma):
            it = ci[ca + i]
            c_target[s + i] = ct[ca + i]
            c_in_bhist_type[s + i] = 0; c_in_bhist_pos[s + i] = -1; c_in_bnew[s + i] = 0; c_in_bhist_src[s + i] = -1
            # newest occurrence in B history; record type of the OLDEST occurrence (first interaction) too
            for j in range(lb - 1, -1, -1):   # oldest -> newest, keep oldest
                if hi[hb + j] == it:
                    c_in_bhist_type[s + i] = ht[hb + j]
                    c_in_bhist_src[s + i] = hs[hb + j]
                    c_in_bhist_pos[s + i] = j
                    break
            if c_in_bhist_pos[s + i] >= 0 and k >= 0 and c_in_bhist_pos[s + i] < k:
                c_in_bnew[s + i] = 1
            c_in_bcand[s + i] = 0
            for j in range(mb):
                if ci[cb + j] == it:
                    c_in_bcand[s + i] = 1
                    break
            c_item_in_ahist[s + i] = 0
            for j in range(la):
                if hi[ha + j] == it:
                    c_item_in_ahist[s + i] = 1
                    break


def main():
    f = feeds('train')
    f = f.filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp', 'fidx'])
    u = f['user_id'].to_numpy(); fi = f['fidx'].to_numpy(); ts = f['timestamp'].to_numpy()
    same = u[1:] == u[:-1]
    a_idx = fi[:-1][same].astype(np.int64); b_idx = fi[1:][same].astype(np.int64)
    gap = (ts[1:] - ts[:-1])[same]
    print('consecutive pairs (same user, train):', len(a_idx))
    F = feeds('train').sort('fidx')
    hoff = F['hist_off'].to_numpy(); H = F['H'].to_numpy(); coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    print('loading arrays...', flush=True)
    hi = np.asarray(arr('train', 'hist_item_id')); ht = np.asarray(arr('train', 'hist_event_type'))
    hs = np.asarray(arr('train', 'hist_event_source'))
    ci = np.asarray(arr('train', 'cand_item_id')); ct = np.asarray(arr('train', 'cand_target'))
    ma = M[a_idx]
    cand_start = np.concatenate([[0], np.cumsum(ma)[:-1]]).astype(np.int64)
    N = ma.sum()
    out_k = np.empty(len(a_idx), np.int32); out_match = np.empty(len(a_idx), np.float32)
    c_type = np.empty(N, np.int8); c_src = np.empty(N, np.int8); c_pos = np.empty(N, np.int16)
    c_new = np.empty(N, np.int8); c_bc = np.empty(N, np.int8); c_t = np.empty(N, np.int8); c_ah = np.empty(N, np.int8)
    print('running kernel on', N, 'A-candidates', flush=True)
    pair_kernel(a_idx, b_idx, hoff, H, coff, M, hi, ht, hs, ci, ct, out_k, out_match, cand_start,
                c_type, c_src, c_pos, c_new, c_bc, c_t, c_ah)
    np.savez('<repo>/eda/history_leak/out_pairs_train.npz', a=a_idx, b=b_idx, gap=gap, k=out_k,
             match=out_match)
    P = pl.DataFrame({'gap': gap, 'k': out_k, 'match': out_match, 'HA': H[a_idx], 'HB': H[b_idx]})
    print('\n--- shift alignment ---')
    print('pairs with gap==0:', (gap == 0).mean())
    print('fraction perfect alignment (B = k new events + A prefix):', (out_match == 1.0).mean())
    print('fraction alignment >=0.9:', (out_match >= 0.9).mean(), ' no anchor found:', (out_k < 0).mean())
    P = P.with_columns(pl.col('gap').cut([0, 128, 600, 3600, 6 * 3600, 86400, 3 * 86400, 7 * 86400],
                                         labels=['<=0', '<=128s', '<=10m', '<=1h', '<=6h', '<=1d', '<=3d', '<=7d', '>7d']).alias('gb'))
    print(P.group_by('gb').agg(pl.len(), (pl.col('match') == 1).mean().alias('perfect'),
                               pl.col('k').filter(pl.col('k') >= 0).median().alias('median_new_events'),
                               pl.col('k').filter(pl.col('k') >= 0).mean().alias('mean_new_events'),
                               (pl.col('k') == 0).mean().alias('k==0')).sort('gb'))
    # candidate level
    pair_of_c = np.repeat(np.arange(len(a_idx)), ma)
    C = pl.DataFrame({'t': c_t, 'type': c_type, 'src': c_src, 'bpos': c_pos, 'new': c_new, 'bcand': c_bc,
                      'ahist': c_ah, 'gap': gap[pair_of_c], 'k': out_k[pair_of_c]})
    print('\nA candidates in A own history (should be 0):', C['ahist'].mean())
    print('\n--- A candidates vs later feed B ---')
    print(C.group_by('t').agg(pl.len(), (pl.col('type') > 0).mean().alias('in_B_hist'),
                              (pl.col('new') == 1).mean().alias('in_B_new_part'),
                              (pl.col('bcand') == 1).mean().alias('is_B_candidate'),
                              ((pl.col('bcand') == 1) & (pl.col('type') > 0)).mean().alias('both')).sort('t'))
    print('target rate | in B hist:', C.filter(pl.col('type') > 0)['t'].mean(),
          ' | not in B hist:', C.filter(pl.col('type') == 0)['t'].mean(),
          ' | is B candidate:', C.filter(pl.col('bcand') == 1)['t'].mean(),
          ' | base:', C['t'].mean())
    print('\nevent type (oldest occurrence in B hist) of A candidates found in B history, by target:')
    print(C.filter(pl.col('type') > 0).group_by(['type']).agg(
        pl.len(), pl.col('t').mean().alias('target_rate'), pl.col('t').sum().alias('n_pos')).sort('type'))
    print('event_source of type-1 events of A positives found in B history:')
    print(C.filter((pl.col('type') == 1) & (pl.col('t') == 1)).group_by('src').len().sort('len', descending=True).head(10))
    print('event_source of type-1 events of A negatives found in B history:')
    print(C.filter((pl.col('type') == 1) & (pl.col('t') == 0)).group_by('src').len().sort('len', descending=True).head(10))
    print('\nrecall of positives in B hist by gap bucket:')
    C = C.with_columns(pl.col('gap').cut([128, 600, 3600, 6 * 3600, 86400, 3 * 86400, 7 * 86400],
                                         labels=['<=128s', '<=10m', '<=1h', '<=6h', '<=1d', '<=3d', '<=7d', '>7d']).alias('gb'))
    print(C.group_by('gb').agg(pl.len(), pl.col('t').mean().alias('base_rate'),
                               (pl.col('type') > 0).filter(pl.col('t') == 1).mean().alias('pos_in_Bhist'),
                               (pl.col('type') == 1).filter(pl.col('t') == 1).mean().alias('pos_in_Bhist_type1'),
                               (pl.col('type') > 0).filter(pl.col('t') == 0).mean().alias('neg_in_Bhist'),
                               pl.col('t').filter(pl.col('type') > 0).mean().alias('P(t|inBhist)'),
                               (pl.col('bcand') == 1).filter(pl.col('t') == 1).mean().alias('pos_is_Bcand'),
                               (pl.col('bcand') == 1).filter(pl.col('t') == 0).mean().alias('neg_is_Bcand'),
                               ).sort('gb'))
    C.write_parquet('<repo>/eda/history_leak/out_pairs_train_cands.parquet')


if __name__ == '__main__':
    main()
