"""Audit checks (b) own part, (c) and (e). CPU, read-only (reads preds/, subs/ through out/_flatscore_*.npy of a_files.py, data/flat). Output: out/b_branches.json.
(b) stored branch files preds/ens_<tag>_{ft,to}_{val,test}.npy: exist, length == number of val / test candidate rows, finite;
    final val recomputed from them (0.5 z(FT) + 0.5 z(TO), per-feed NDCG, src/metrics.py) == the final val on record to 5 decimals;
    the submission file == src/make_sub.py's rank transform of 0.5 z(FT test) + 0.5 z(TO test) of the stored branches (row-exact).
    (The read-out of build_r4.py / build_r3.py themselves is run by b_readout.sh.)
(c) member list from the builder sources (cands.members); for every member its four prediction files exist in preds/, have the right length, are finite,
    and are byte-identical to the file in the member's harness directory (nn_x/w5/preds or nn_x/abl/preds) where that file exists;
    branches REBUILT from the members (formula of build_r4.py / build_r3.py / build_w.py) compared with the stored branch files.
(e) test-score correlation between candidates from the stored branch files (final score 0.5 z(FT) + 0.5 z(TO)):
    pooled Pearson over the 2,904,806 rows (the statistic build_r4.py prints), feed-mean Pearson, feed-mean Spearman, share of feeds with the same top-1 item;
    and paired val deltas (per-feed NDCG) of every candidate against FD16P20 and W8P20.
"""
import sys, json, hashlib, datetime
sys.path.insert(0, '<repo>/nn_x/w12/freeze'); sys.path.insert(0, '<repo>/src')
import numpy as np
from cands import ROOT, OUT, PREDS, CANDS, SUFFIXES, members, harness
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
NROW = {'val': int(Mv.sum()), 'test': int(Mt.sum())}
OFF = {'val': ov, 'test': ot}; MM = {'val': Mv, 'test': Mt}
Z = lambda sp, x: feed_rank_z(OFF[sp], MM[sp], np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
fin = lambda sp, ft, to: 0.5 * Z(sp, ft) + 0.5 * Z(sp, to)
fidx_t = np.load(FLAT / 'test/cand_fidx.npy'); fidx_rep = {'val': np.repeat(np.arange(len(Mv)), Mv), 'test': np.repeat(np.arange(len(Mt)), Mt)}
assert (fidx_t == fidx_rep['test']).all(), 'data/flat/test is not in feed order'
MROW = {sp: np.repeat(MM[sp], MM[sp]) for sp in ('val', 'test')}
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def make_sub_rank(scores):
    """The rank src/make_sub.py writes (copied logic: lexsort by feed, score, fixed random key; rank 0 = worst)."""
    scores = np.asarray(scores, dtype=np.float64); key = np.random.default_rng(0).random(len(scores))
    order = np.lexsort((key, scores, fidx_t)); rank = np.empty(len(scores), np.float64); rank[order] = np.arange(len(scores)) - np.repeat(ot, Mt)
    return rank.astype(np.float32)


def within_rank(x, sp='test'):
    o = np.lexsort((x, fidx_rep[sp])); r = np.empty(len(x), np.float64); r[o] = np.arange(len(x)) - np.repeat(OFF[sp], MM[sp]); return r


def feed_pearson(a, b, sp='test'):
    """Pearson of a and b inside each feed (NaN for constant feeds / M == 1)."""
    f = fidx_rep[sp]; n = MM[sp].astype(np.float64); a = a.astype(np.float64); b = b.astype(np.float64)
    sa = np.bincount(f, a) / n; sb = np.bincount(f, b) / n; a0 = a - sa[f]; b0 = b - sb[f]
    cov = np.bincount(f, a0 * b0); va = np.bincount(f, a0 * a0); vb = np.bincount(f, b0 * b0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return cov / np.sqrt(va * vb)


def top1(x, sp='test'):
    o = np.lexsort((x, fidx_rep[sp])); last = OFF[sp] + MM[sp] - 1; return o[last]


res = {'stamp_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'rows': NROW, 'feeds': {'val': len(Mv), 'test': len(Mt)}, 'cands': {}}
FIN = {}; EV = {}; member_cache = {}


def load_member(m, suf):
    k = (m, suf)
    if k not in member_cache:
        member_cache[k] = np.load(PREDS / f'nn_{m}_{suf}.npy')
    return member_cache[k]


member_file_report = {}
for c in CANDS:
    cid, tag = c['id'], c['tag']; r = {}
    # ---------- (b) stored branches
    br = {}; r['b_branch_files'] = {}
    for sp in ('val', 'test'):
        for b in ('ft', 'to'):
            p = PREDS / f'ens_{tag}_{b}_{sp}.npy'; e = {'exists': p.exists()}
            if e['exists']:
                x = np.load(p); e.update(shape=list(x.shape), dtype=str(x.dtype), finite=bool(np.isfinite(x).all()), rows_ok=len(x) == NROW[sp]); br[(b, sp)] = x
            r['b_branch_files'][f'ens_{tag}_{b}_{sp}.npy'] = e
    r['b1_branch_files_ok'] = all(e['exists'] and e.get('finite') and e.get('rows_ok') for e in r['b_branch_files'].values())
    if not r['b1_branch_files_ok']:
        res['cands'][cid] = r; print(cid, 'BRANCH FILES NOT OK', r['b_branch_files']); continue
    fval = fin('val', br[('ft', 'val')], br[('to', 'val')]); e = ev(fval); EV[cid] = e
    r['final_val_from_stored'] = float(e.mean()); r['final_val_from_stored_5d'] = f'{e.mean():.5f}'; r['val_rec'] = c['val_rec']
    r['ft_val_from_stored'] = float(ev(br[('ft', 'val')]).mean()); r['to_val_from_stored'] = float(ev(br[('to', 'val')]).mean())
    r['b2_final_val_equals_record'] = r['final_val_from_stored_5d'] == c['val_rec']
    ftest = fin('test', br[('ft', 'test')], br[('to', 'test')]).astype(np.float32); FIN[cid] = ftest
    rank = make_sub_rank(ftest); fs = np.load(OUT / f'_flatscore_{tag}.npy')
    neq = rank != fs; r['file_rows_differing_from_rebuilt_rank'] = int(neq.sum()); r['file_feeds_differing'] = int(len(np.unique(fidx_t[neq])))
    rho = feed_pearson(rank, fs); r['file_vs_stored_min_feed_spearman'] = float(np.nanmin(rho)); r['b3_file_equals_stored_branches'] = r['file_rows_differing_from_rebuilt_rank'] == 0
    # ---------- (c) members
    base, pop, w = members(cid); r['members_base'] = base; r['members_pop'] = pop; r['w_pop'] = w; r['n_members'] = len(base) + len(pop)
    miss = []; bad = []; harness_diff = []; harness_absent = []
    for m in base + pop:
        for suf in SUFFIXES:
            p = PREDS / f'nn_{m}_{suf}.npy'; key = f'nn_{m}_{suf}.npy'
            if key in member_file_report:
                st = member_file_report[key]
            else:
                st = {'exists': p.exists()}
                if st['exists']:
                    x = load_member(m, suf); sp = 'val' if 'val' in suf else 'test'
                    st.update(rows_ok=len(x) == NROW[sp], finite=bool(np.isfinite(x).all()), dtype=str(x.dtype), sha256=sha(p))
                    hp = harness(m) / 'preds' / key
                    st['harness_copy_exists'] = hp.exists()
                    if hp.exists(): st['identical_to_harness_copy'] = sha(hp) == st['sha256']
                member_file_report[key] = st
            if not st['exists']: miss.append(key)
            elif not (st['rows_ok'] and st['finite']): bad.append(key)
            if st.get('harness_copy_exists') is False: harness_absent.append(key)
            if st.get('identical_to_harness_copy') is False: harness_diff.append(key)
    r['member_files_expected'] = 4 * r['n_members']; r['member_files_missing'] = miss; r['member_files_bad'] = bad
    r['member_files_differ_from_harness_copy'] = harness_diff; r['member_files_without_harness_copy'] = harness_absent
    r['c1_member_files_ok'] = not miss and not bad
    if r['c1_member_files_ok']:
        reb = {}
        for sp in ('val', 'test'):
            ftn, ton = SUF[sp]
            pl_ = lambda names: (np.mean([Z(sp, load_member(m, ftn)) for m in names], 0), np.mean([Z(sp, load_member(m, ton)) for m in names], 0))
            b = pl_(base)
            if pop:
                wr = w if not isinstance(w, tuple) else np.where(MROW[sp] >= 50, w[1], w[0]).astype(np.float32)
                p1 = pl_(pop); b = ((1 - wr) * Z(sp, b[0]) + wr * Z(sp, p1[0]), (1 - wr) * Z(sp, b[1]) + wr * Z(sp, p1[1]))
            reb[sp] = b
        mx = 0.0; exact = True; cmin = 1.0
        for sp in ('val', 'test'):
            for i, bname in enumerate(('ft', 'to')):
                a = np.asarray(reb[sp][i], np.float32); s = br[(bname, sp)]
                mx = max(mx, float(np.abs(a - s).max())); exact = exact and bool(np.array_equal(a, s)); cmin = min(cmin, float(np.corrcoef(a, s)[0, 1]))
        e2 = ev(fin('val', reb['val'][0], reb['val'][1]))
        r['rebuilt_vs_stored_max_abs_diff'] = mx; r['rebuilt_vs_stored_bit_exact'] = exact; r['rebuilt_vs_stored_min_corr'] = cmin
        r['final_val_from_members'] = float(e2.mean()); r['final_val_from_members_5d'] = f'{e2.mean():.5f}'
        r['c2_members_reproduce_stored_branches'] = mx <= 1e-5 and r['final_val_from_members_5d'] == c['val_rec']
        rk2 = make_sub_rank(fin('test', reb['test'][0], reb['test'][1]).astype(np.float32)); r['file_rows_differing_from_member_rebuilt_rank'] = int((rk2 != fs).sum())
    res['cands'][cid] = r
    print(f"{cid:18s} b1 {r['b1_branch_files_ok']} | final val stored {r['final_val_from_stored_5d']} record {c['val_rec']} b2 {r['b2_final_val_equals_record']} | file rows != rank of stored branches {r['file_rows_differing_from_rebuilt_rank']} b3 {r['b3_file_equals_stored_branches']}"
          f" | members {len(base)}+{len(pop)} files missing {len(miss)} bad {len(bad)} c1 {r['c1_member_files_ok']} | rebuilt-vs-stored max|d| {r.get('rebuilt_vs_stored_max_abs_diff')} exact {r.get('rebuilt_vs_stored_bit_exact')}"
          f" val from members {r.get('final_val_from_members_5d')} c2 {r.get('c2_members_reproduce_stored_branches')} | differ from harness copy {len(harness_diff)} no harness copy {len(harness_absent)}", flush=True)
res['member_files'] = member_file_report

# ---------- (e) correlations and paired val deltas
ids = [c['id'] for c in CANDS if c['id'] in FIN]
R = {sp_: within_rank(FIN[i]) for sp_, i in zip(ids, ids)}; T1 = {i: top1(FIN[i]) for i in ids}
corr = {}
for i, a in enumerate(ids):
    for b in ids[i + 1:]:
        fp = feed_pearson(FIN[a], FIN[b]); fs_ = feed_pearson(R[a], R[b])
        corr[f'{a}|{b}'] = dict(pooled_pearson=float(np.corrcoef(FIN[a], FIN[b])[0, 1]), feed_mean_pearson=float(np.nanmean(fp)), feed_mean_spearman=float(np.nanmean(fs_)),
                                feed_p5_spearman=float(np.nanpercentile(fs_, 5)), same_top1_share=float((T1[a] == T1[b]).mean()))
res['e_test_corr'] = corr
dv = {}
for ref in ('FD16P20', 'W8P20'):
    for i in ids:
        if i == ref: continue
        d = EV[i] - EV[ref]; dv[f'{i} - {ref}'] = dict(delta=float(d.mean()), paired_se=float(d.std() / np.sqrt(len(d))), paired_public_se=float(d.std() / np.sqrt(8562)))
res['val_delta'] = dv
print('\n(e) test-score correlation from the stored branch files (final = 0.5 z(FT) + 0.5 z(TO)):')
for k in ('FD16P20|W8P20', 'FD16P20|FD8P20'):
    v = corr[k]; print(f"  {k:28s} pooled Pearson {v['pooled_pearson']:.4f} | feed-mean Pearson {v['feed_mean_pearson']:.4f} | feed-mean Spearman {v['feed_mean_spearman']:.4f} (p5 {v['feed_p5_spearman']:.4f}) | same top-1 {100 * v['same_top1_share']:.1f}% of feeds")
print('  all pairs, pooled Pearson: ' + ' ; '.join(f"{k} {v['pooled_pearson']:.4f}" for k, v in corr.items()))
print('paired val deltas (per-feed NDCG, 14,423 val feeds; +- paired SE; [paired public SE = sd / sqrt(8562)]):')
for k, v in dv.items(): print(f"  {k:32s} {v['delta']:+.5f} +- {v['paired_se']:.5f}  [{v['paired_public_se']:.5f}]")
json.dump(res, open(OUT / 'b_branches.json', 'w'), indent=1)
print('written', OUT / 'b_branches.json')
