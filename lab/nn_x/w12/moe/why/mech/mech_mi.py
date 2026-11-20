"""w12/moe/why/mech: routing specialisation (mutual information with token attributes) and cross-seed routing consistency, from the
routing records saved by mech_run.py (CPU only).
  python mech_mi.py A|B
MI in bits from the full contingency table over real tokens; 'bias' = MI of the same table with the expert labels randomly permuted
(1 permutation; the plug-in bias), reported MI is net of it. frac = MI / H(expert). Expert variable = top-1 expert (the first of the
top-k selection = largest biased score) unless stated; pair = the unordered top-2 set.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = '<repo>'
sys.path.insert(0, ROOT + '/lab/nn_x/w12/moe/why/mech')
from mech_common import FLAT, OUT, holdout_feeds, feat_path  # noqa: E402

grp = sys.argv[1]
ARM = dict(A='kd05w192moeffn', B='kd05w192moevo')[grp]
SEEDS = (951, 952, 953, 954)
D = OUT / grp
rng = np.random.default_rng(0)


def H(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mi(a, e, na=None, ne=None):
    na = int(a.max()) + 1 if na is None else na
    ne = int(e.max()) + 1 if ne is None else ne
    c = np.bincount(a.astype(np.int64) * ne + e, minlength=na * ne).reshape(na, ne).astype(np.float64)
    p = c / c.sum()
    return H(p.sum(1)) + H(p.sum(0)) - H(p.ravel())


def mi_net(a, e):
    m = mi(a, e)
    b = mi(a, rng.permutation(e))
    return m - b, b


def qbin(x, q=10):
    x = np.asarray(x, np.float64)
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, q + 1)[1:-1]))
    return np.searchsorted(edges, x, side='right')


def popbin(x):
    x = np.nan_to_num(np.asarray(x, np.float64), nan=0.0)
    out = np.zeros(len(x), np.int64)
    pos = x > 0
    out[pos] = 1 + qbin(x[pos], 9)
    return out


def cand_attrs(split):
    if split == 'val':
        rows = None
        ld = lambda n: np.load(FLAT / 'val' / f'{n}.npy')
        import polars as pl
        f = pl.read_parquet(FLAT / 'val' / 'feeds.parquet')
        M = f['M'].to_numpy().astype(np.int64)
        pop = np.load(feat_path('val', 'it_hist_all_luo'))
    else:
        f, hold, _ = holdout_feeds()
        M_all = f['M'].to_numpy().astype(np.int64); co = f['cand_off'].to_numpy().astype(np.int64)
        M = M_all[hold]
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(co[hold], M)])
        ld = lambda n: np.asarray(np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')[rows])
        pop = np.asarray(np.load(feat_path('train', 'it_hist_all_luo'), mmap_mode='r')[rows])
    n = int(M.sum())
    fid = np.repeat(np.arange(len(M)), M)
    A = dict(category=ld('cand_category_id').astype(np.int64), age_f4=qbin(ld('cand_f4')), engage_f5=qbin(ld('cand_f5')),
             ucat_aff_f3=qbin(ld('cand_f3')), ucat_aff_f1=qbin(ld('cand_f1')), region_f2=qbin(ld('cand_f2')),
             popularity=popbin(pop), feed_size=qbin(np.repeat(M, M)), label=ld('cand_target').astype(np.int64),
             emb_missing=(ld('cand_emb_idx') < 0).astype(np.int64))
    if split == 'hold':
        A['position'] = qbin(ld('cand_pos'))
    assert all(len(v) == n for v in A.values())
    return A, fid, M


def within_rank_decile(score, M):
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    out = np.empty(len(score), np.int64)
    for o, m in zip(off, M):
        r = np.argsort(np.argsort(-score[o:o + m], kind='stable'), kind='stable')
        out[o:o + m] = np.minimum((10 * r) // m, 9)
    return out


def mi_feed(e, fid, nexp):
    """MI(expert; feed) = H(E) - sum_f p_f H(E | f), net of the permutation null."""
    def one(ee):
        c = np.bincount(fid * nexp + ee, minlength=(fid.max() + 1) * nexp).reshape(-1, nexp).astype(np.float64)
        nf = c.sum(1)
        pf = c / nf[:, None].clip(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            hf = -np.nansum(np.where(pf > 0, pf * np.log2(pf), 0.0), 1)
        return H(c.sum(0) / c.sum()) - float((nf * hf).sum() / nf.sum())
    m = one(e)
    return m - one(rng.permutation(e)), one(rng.permutation(e))


def nmi(a, b):
    m = mi(a, b)
    return m / np.sqrt(H(np.bincount(a) / len(a)) * H(np.bincount(b) / len(b)))


def matched_agreement(a, b, k):
    c = np.bincount(a * k + b, minlength=k * k).reshape(k, k)
    r, cc = linear_sum_assignment(-c)
    return float(c[r, cc].sum() / c.sum())


res = {}
if grp == 'A':
    blocks = ['cross.0', 'sets.0', 'cross.1', 'sets.1']
    E = 8
    for split in ('val', 'hold'):
        A, fid, M = cand_attrs(split)
        top1 = {}
        for s in SEEDS:
            v = f'w5_e10_{ARM}_s{s}'
            from mech_common import pred_paths
            pv, ph = pred_paths(v)
            score = np.load(pv if split == 'val' else ph)
            A_ = dict(A, own_rank_decile=within_rank_decile(score, M))
            for b in blocks:
                z = np.load(D / f'route_{v}_{split}_{b}.npz')
                sel = z['sel'].astype(np.int64)
                assert (sel >= 0).all()
                e1 = sel[:, 0]
                pair = np.minimum(sel[:, 0], sel[:, 1]) * E + np.maximum(sel[:, 0], sel[:, 1])
                top1[(s, b)] = e1
                he = H(np.bincount(e1, minlength=E) / len(e1))
                hp = H(np.bincount(pair) / len(pair))
                r = dict(H_top1=he, H_pair=hp, load_top1=(np.bincount(e1, minlength=E) / len(e1)).round(4).tolist())
                for an, av in A_.items():
                    m, bias = mi_net(av, e1)
                    mp, _ = mi_net(av, pair)
                    r[an] = dict(mi=round(m, 5), frac=round(m / he, 4), frac_pair=round(mp / hp, 4), bias=round(bias, 6))
                mf, bf = mi_feed(e1, fid, E)
                r['feed'] = dict(mi=round(mf, 4), frac=round(mf / he, 4), bias=round(bf, 4))
                g = z['gate'].astype(np.float32)
                r['gate_top1_mean'] = float(g[:, 0].mean()); r['gate_top1_std'] = float(g[:, 0].std())
                res[f'{split}.{v}.{b}'] = r
                print(split, v, b, 'H', round(he, 3), {k: r[k]['frac'] for k in r if isinstance(r[k], dict)}, flush=True)
        # cross-seed consistency (same block), and across blocks within a seed (context)
        for b in blocks:
            vals = []
            for i, s1 in enumerate(SEEDS):
                for s2 in SEEDS[i + 1:]:
                    vals.append((nmi(top1[(s1, b)], top1[(s2, b)]), matched_agreement(top1[(s1, b)], top1[(s2, b)], E)))
            res[f'{split}.xseed.{b}'] = dict(nmi_mean=float(np.mean([x[0] for x in vals])), nmi_min=float(np.min([x[0] for x in vals])),
                                            nmi_max=float(np.max([x[0] for x in vals])), agree_mean=float(np.mean([x[1] for x in vals])))
            print(split, 'xseed', b, res[f'{split}.xseed.{b}'], flush=True)
        for s in SEEDS:
            xb = [nmi(top1[(s, b1)], top1[(s, b2)]) for i, b1 in enumerate(blocks) for b2 in blocks[i + 1:]]
            res[f'{split}.xblock.{s}'] = dict(nmi_mean=float(np.mean(xb)), nmi_max=float(np.max(xb)))
        print(split, 'xblock', {s: res[f'{split}.xblock.{s}'] for s in SEEDS}, flush=True)
else:
    import polars as pl
    blocks = ['cross.0', 'cross.1']
    E, h = 4, 4
    # key-token attributes (val): canonical key index = val hist row (history keys) or n_hist + feed (user token)
    f = pl.read_parquet(FLAT / 'val' / 'feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64)
    Hh = f['H'].to_numpy().astype(np.int64); ho = f['hist_off'].to_numpy().astype(np.int64)
    het = np.load(FLAT / 'val/hist_event_type.npy').astype(np.int64)
    hes = np.load(FLAT / 'val/hist_event_source.npy').astype(np.int64)
    hcat = np.load(FLAT / 'val/hist_category_id.npy').astype(np.int64)
    nh = len(het)
    hfid = np.repeat(np.arange(len(Hh)), Hh)
    pos = np.arange(nh) - ho[hfid]
    keyA = dict(event_type=np.concatenate([het, np.zeros(len(M), np.int64)]),
                event_source=np.concatenate([hes, np.full(len(M), 31)]),
                category=np.concatenate([hcat, np.full(len(M), 54)]),
                recency=np.concatenate([np.floor(np.log2(1 + pos)).astype(np.int64), np.full(len(M), 10)]),
                is_user=np.concatenate([np.zeros(nh, np.int64), np.ones(len(M), np.int64)]))
    keyfid = np.concatenate([hfid, np.arange(len(M))])
    CA, cfid, _ = cand_attrs('val')
    top = {}
    for s in SEEDS:
        v = f'w5_e10_{ARM}_s{s}'
        for b in blocks:
            z = np.load(D / f'route_{v}_val_{b}.npz')
            for R, attrs, fidx in (('V', keyA, keyfid), ('O', CA, cfid)):
                sel = z[f'sel{R}'].astype(np.int64)
                ok = sel[:, 0, 0] >= 0
                sv = z[f's{R}'].astype(np.float32)[ok]
                r = {}
                for hh in range(h):
                    e1 = sel[ok, hh, 0]
                    pair = np.minimum(sel[ok, hh, 0], sel[ok, hh, 1]) * E + np.maximum(sel[ok, hh, 0], sel[ok, hh, 1])
                    top[(s, b, R, hh)] = (ok, e1)
                    he = H(np.bincount(e1, minlength=E) / len(e1)); hp = H(np.bincount(pair) / len(pair))
                    rr = dict(H_top1=he, H_pair=hp, n=int(ok.sum()))
                    for an, av in attrs.items():
                        m, bias = mi_net(av[ok], e1)
                        mp, _ = mi_net(av[ok], pair)
                        rr[an] = dict(mi=round(m, 5), frac=round(m / he, 4), frac_pair=round(mp / hp, 4))
                    mf, bf = mi_feed(e1, fidx[ok], E)
                    rr['feed'] = dict(mi=round(mf, 4), frac=round(mf / he, 4), bias=round(bf, 4))
                    # gate near-constancy: std across tokens of each sigmoid gate vs its mean; share of tokens on the modal pair
                    rr['s_mean'] = sv[:, hh].mean(0).round(4).tolist(); rr['s_std'] = sv[:, hh].std(0).round(4).tolist()
                    rr['modal_pair_share'] = float(np.bincount(pair).max() / len(pair))
                    r[f'h{hh}'] = rr
                res[f'{v}.{b}.{R}'] = r
                print(v, b, R, {hk: (round(r[hk]['H_top1'], 2), {k: r[hk][k]['frac'] for k in r[hk] if isinstance(r[hk][k], dict)}) for hk in r}, flush=True)
    for b in blocks:
        for R in 'VO':
            vals = []
            for i, s1 in enumerate(SEEDS):
                for s2 in SEEDS[i + 1:]:
                    C = np.zeros((h, h))
                    for h1 in range(h):
                        for h2 in range(h):
                            ok1, e1 = top[(s1, b, R, h1)]; ok2, e2 = top[(s2, b, R, h2)]
                            assert (ok1 == ok2).all()
                            C[h1, h2] = nmi(e1, e2)
                    r_, c_ = linear_sum_assignment(-C)
                    vals.append((C[r_, c_].mean(), np.diag(C).mean()))
            res[f'xseed.{b}.{R}'] = dict(nmi_head_matched=float(np.mean([x[0] for x in vals])), nmi_same_head=float(np.mean([x[1] for x in vals])))
            print('xseed', b, R, res[f'xseed.{b}.{R}'], flush=True)
(D / f'mi_{grp}.json').write_text(json.dumps(res, indent=1))
