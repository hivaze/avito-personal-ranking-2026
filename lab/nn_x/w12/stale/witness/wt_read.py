"""witness read-out: every staleness witness side by side for a pair of pools (A = arm, B = reference), train-only level, paired per feed.

Columns (definitions fixed in DECLARATION.txt block 1 (2)):
  val (all, M>=50, M<50), pmix-pub, pmix-priv, coldpos / warmpos   <- stored GPU train-only val predictions
  stale-val, SV2 = per-feed mean of (val delta, stale-val delta)   <- CPU fp32 stale-val of BOTH pools (stale_val_cpu.py)
  hold (30,000 in-period holdout feeds; reported only for the staleness family), hold on the subset, HOLD-SHORT   <- stored GPU holdout predictions
  lag0 / lag-priv / lag-pub on the SUBSET (13,424 feeds), D_priv, D_pub, pool level drops                          <- lagged_score.py (CPU fp32)
Pool spec syntax (CLI and API): 'W8' | 'hz@last' | 'name1,name2' | '0.8*W8+0.2*pop1kd' (weighted: sum_g w_g * z(mean member z of group g)).
  '0.3~F10h+0.35~F7xh+0.35~F7x70h' = raw-weighted group means (sum_g w_g * mean member z of group g; a member-equal pool with fewer seeds per group).
  --pairs hz:ref  restricts re-cool arms to the snapshots whose runs are complete on BOTH sides (logs/<variant>.json) and fully scored.

  cd <repo> && WT_THREADS=4 CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/witness/wt_read.py \
      pair '<spec A>' '<spec B>' [label]
"""
from wt_common import *
import lagged_score as LS
import stale_val_cpu as SVC


def parse(spec, suffix=''):
    """spec -> list of (weight, [members]); a single unweighted group -> [(1.0, members)]."""
    out = []
    for term in spec.split('+'):
        raw = '~' in term                       # 'w~group': raw-weighted group mean of member z (no re-standardisation of the group mean)
        w, g = (term.split('~') if raw else (term.split('*') if '*' in term else ('1', term)))
        suf = suffix
        if g.endswith('@last'):
            g, suf = g[:-5], '@last'
        mem = [m + suf for m in (G[g] if g in G else g.split(','))]
        out.append((float(w), mem, 'raw') if raw else (float(w), mem))
    return out


def members(spec):
    return [m for g in spec for m in g[1]]


class Space:
    """Rows of one evaluation sample + loaders of raw member predictions by kind."""

    def __init__(self, M, off, y, loader):
        self.M = np.asarray(M, np.int64); self.off = np.asarray(off, np.int64); self.y = np.asarray(y, np.float32); self.loader = loader
        self._z = {}; self._lv = {}

    def z(self, kind, v):
        k = (kind, v)
        if k not in self._z:
            self._z[k] = feed_z(self.off, self.M, self.loader(kind, v))
        return self._z[k]

    def pool(self, spec, kind):
        if len(spec) == 1:
            return np.mean([self.z(kind, v) for v in spec[0][1]], 0)
        acc = 0.0
        for g in spec:
            mz = np.mean([self.z(kind, v) for v in g[1]], 0)
            acc = acc + g[0] * (mz if len(g) > 2 and g[2] == 'raw' else feed_z(self.off, self.M, mz)).astype(np.float64)
        return acc

    def level(self, spec, kind):
        k = (json.dumps(spec), kind)
        if k not in self._lv:
            self._lv[k] = ndcg_fast(self.pool(spec, kind), self.y, self.off, self.M)
        return self._lv[k]


class Witness:
    def __init__(self):
        import polars as pl
        fv = pl.read_parquet(FLAT / 'val/feeds.parquet')
        self.Mv = fv['M'].to_numpy().astype(np.int64); self.ov = fv['cand_off'].to_numpy().astype(np.int64)
        yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)

        def lv(kind, v):
            p = {'val': stored_val(v), 'stale': SVC.sv_path(v, 'stale'), 'base': SVC.sv_path(v, 'base'), 'gstale': stored_stale(v)}[kind]
            return np.load(p).astype(np.float32)
        self.V = Space(self.Mv, self.ov, yv, lv)
        fid = np.repeat(np.arange(len(self.Mv)), self.Mv)
        luo = np.nan_to_num(np.load(FC / 'val/it_hist_all_luo.npy').astype(np.float64), nan=0.0)
        npos = np.bincount(fid, yv, len(self.Mv)); ncold = np.bincount(fid, yv * (luo == 0), len(self.Mv))
        self.coldpos = (npos > 0) & (ncold == npos); self.warmpos = (npos > 0) & (ncold == 0)       # nn_x/r6/r6_eval.py slices
        self.W = self._pmix()
        Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy')
        self.yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
        self.H = Space(Mh, oh, self.yh, lambda kind, v: np.load(stored_holdout(v)).astype(np.float32))
        self.L = LS.Lag()
        self.subfeeds = np.load(SUB / 'pick.npy')
        self._short = None

    def _pmix(self):
        f = CACHE / 'pmix_weights.npz'
        if not f.exists():
            sys.path.insert(0, str(ROOT / 'src'))
            import pmix
            w = pmix.weights()
            np.savez(f, public=w['public'], private=w['private'], cs=np.array([w['cs_val'], w['cs_pub'], w['cs_priv']]))
        w = np.load(f)
        assert len(w['public']) == len(self.Mv)
        return {'public': w['public'], 'private': w['private']}

    # ---- availability
    def have(self, spec, kind):
        chk = {'val': stored_val, 'stale': lambda v: SVC.sv_path(v, 'stale'), 'base': lambda v: SVC.sv_path(v, 'base'), 'gstale': stored_stale,
               'hold': stored_holdout, 'lag0': lambda v: LS.pred_path('lag0', v), 'priv': lambda v: LS.pred_path('priv', v),
               'pub': lambda v: LS.pred_path('pub', v)}[kind]
        return all(chk(v).exists() for v in members(spec))

    # ---- HOLD-SHORT (arm_read8.py construction, generalised to weighted pools)
    def short(self, A, B):
        if self._short is None:
            mshort = self.Mv[self.Mv < 50]; rng = np.random.default_rng(20261114); Mh, oh = self.H.M, self.H.off
            fid = np.repeat(np.arange(len(Mh)), Mh); reps = []
            for _ in range(4):
                msz = rng.choice(mshort, len(Mh)); key = rng.random(len(self.yh))
                pk = np.where(self.yh > 0, key, 2.0); o = np.lexsort((pk, fid)); first = o[oh]; key[first] = -1.0
                o = np.lexsort((key, fid)); rank = np.empty(len(key), np.int64); rank[o] = np.arange(len(key)) - np.repeat(oh, Mh)
                sel = np.flatnonzero(rank < np.repeat(msz, Mh)); Ms = msz.astype(np.int64)
                reps.append((sel, Ms, np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)))
            self._short = reps
        ds = np.zeros(len(self.H.M)); raw = {}
        for v in set(members(A) + members(B)):
            raw[v] = np.load(stored_holdout(v)).astype(np.float32)
        for sel, Ms, os_ in self._short:
            sp = Space(Ms, os_, self.yh[sel], lambda kind, v: raw[v][sel])
            ds += (sp.level(A, 'hold') - sp.level(B, 'hold')) / 4
        return ds

    # ---- the table
    def read(self, A, B, lag_views=('lag0', 'priv', 'pub'), short=True):
        """All witnesses for A - B. Returns dict name -> (mean, SE, n) plus levels and per-feed arrays under '_pf'."""
        r = {}; pf = {}
        big = self.Mv >= 50
        if self.have(A, 'val') and self.have(B, 'val'):
            a, b = self.V.level(A, 'val'), self.V.level(B, 'val'); d = a - b; pf['val'] = d
            r['A_val'], r['B_val'] = float(a.mean()), float(b.mean())
            r['val'] = ms(d) + (len(d),); r['val_M>=50'] = ms(d[big]) + (int(big.sum()),); r['val_M<50'] = ms(d[~big]) + (int((~big).sum()),)
            r['pmix_pub'] = ms(d, self.W['public']) + (len(d),); r['pmix_priv'] = ms(d, self.W['private']) + (len(d),)
            r['coldpos'] = ms(d[self.coldpos]) + (int(self.coldpos.sum()),); r['warmpos'] = ms(d[self.warmpos]) + (int(self.warmpos.sum()),)
        if self.have(A, 'stale') and self.have(B, 'stale'):
            a, b = self.V.level(A, 'stale'), self.V.level(B, 'stale'); ds = a - b; pf['stale'] = ds
            r['A_stale'], r['B_stale'] = float(a.mean()), float(b.mean())
            r['stale'] = ms(ds) + (len(ds),); r['stale_M>=50'] = ms(ds[big]) + (int(big.sum()),); r['stale_M<50'] = ms(ds[~big]) + (int((~big).sum()),)
            r['stale_pmix_priv'] = ms(ds, self.W['private']) + (len(ds),)
            r['stale_coldpos'] = ms(ds[self.coldpos]) + (int(self.coldpos.sum()),); r['stale_warmpos'] = ms(ds[self.warmpos]) + (int(self.warmpos.sum()),)
            if 'val' in pf:
                r['SV2'] = ms((pf['val'] + ds) / 2) + (len(ds),); pf['SV2'] = (pf['val'] + ds) / 2
                r['SVpriv'] = ms(0.27 * pf['val'] + 0.73 * ds) + (len(ds),)          # private-like position on the cold-share axis (DECLARATION block 11)
                r['stale_minus_val'] = ms(ds - pf['val']) + (len(ds),)          # change of the delta when the val inputs go stale (GPU val vs CPU stale: includes numerics)
        if self.have(A, 'hold') and self.have(B, 'hold'):
            a, b = self.H.level(A, 'hold'), self.H.level(B, 'hold'); dh = a - b; pf['hold'] = dh
            r['A_hold'], r['B_hold'] = float(a.mean()), float(b.mean())
            r['hold'] = ms(dh) + (len(dh),); r['hold_subset'] = ms(dh[self.subfeeds]) + (len(self.subfeeds),)
            if short:
                dsh = self.short(A, B); r['HOLD-SHORT'] = ms(dsh) + (len(dsh),)
        views = [w for w in lag_views if self.have(A, w) and self.have(B, w)]
        if views:
            c = self.L.compare(A, B, views)
            for w in views:
                r[f'lag_{w}'] = c[f'd_{w}'] + (c['n'],); r[f'A_lag_{w}'], r[f'B_lag_{w}'] = c[f'A_{w}'], c[f'B_{w}']; pf[f'lag_{w}'] = c['_per_feed'][w]
                if w != 'lag0' and 'lag0' in views:
                    r[f'D_{w}'] = c[f'D_{w}'] + (c['n'],); r[f'dropA_{w}'] = c[f'dropA_{w}'] + (c['n'],); r[f'dropB_{w}'] = c[f'dropB_{w}'] + (c['n'],)
        r['_pf'] = pf
        return r


def ivw(*xs):
    """Inverse-variance mean of independent (mean, SE, ...) estimates."""
    w = np.array([1 / x[1] ** 2 for x in xs]); m = np.array([x[0] for x in xs])
    return float((w * m).sum() / w.sum()), float(np.sqrt(1 / w.sum()))


def cell(x):
    return 'n/a' if x is None else f'{x[0]:+.5f} +- {x[1]:.5f}'


ORDER = ['val', 'val_M>=50', 'val_M<50', 'pmix_pub', 'pmix_priv', 'coldpos', 'warmpos', 'stale', 'stale_M>=50', 'stale_M<50', 'stale_pmix_priv',
         'stale_coldpos', 'stale_warmpos', 'SV2', 'SVpriv', 'stale_minus_val', 'hold', 'hold_subset', 'HOLD-SHORT', 'lag_lag0', 'lag_priv', 'lag_pub', 'D_priv', 'D_pub',
         'dropA_priv', 'dropB_priv', 'dropA_pub', 'dropB_pub']


def show(r, title=''):
    out = [f'== {title}']
    lv = ' | '.join(f'{k} {r[k]:.5f}' for k in r if k.startswith(('A_', 'B_')))
    out.append(f'   levels: {lv}')
    for k in ORDER:
        if k in r:
            m, s, n = r[k]
            out.append(f'   {k:16s} {m:+.5f} +- {s:.5f}  (z {m / s:+.2f}; n {n})')
    return '\n'.join(out)


def dump(r):
    return {k: v for k, v in r.items() if k != '_pf'}


if __name__ == '__main__':
    assert sys.argv[1] == 'pair'
    A, B = parse(sys.argv[2]), parse(sys.argv[3]); label = sys.argv[4] if len(sys.argv) > 4 else f'{sys.argv[2]}__vs__{sys.argv[3]}'
    wt = Witness(); r = wt.read(A, B)
    s = show(r, f'{sys.argv[2]}  vs  {sys.argv[3]}   ({now()})'); print(s)
    (CACHE / 'readout').mkdir(exist_ok=True)
    json.dump(dump(r), open(CACHE / 'readout' / f"{label.replace('/', '_')}.json", 'w'), indent=1)
