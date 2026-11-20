"""BLOCKS: (1) the W25 tree arms read with the round-2 NATS screen (they were run on 11-13, before the screen existed);
(2) every on-disk block of the CatBoost rounds / W25 as a LINEAR composite next to the W16 train-only pool, same protocol for all blocks:
    per-feed z of each column (NaN -> 0) + a missing-flag z-column where the NaN pattern varies inside feeds; composite coefficients fitted
    on one user half and read on the other (2-fold, user-grouped, inside each witness); val is also read with coefficients fitted on ALL
    30,000 holdout feeds (train-period labels only: the direction a trained model works in).
    NATS = gain of [z_W16, composite] over [z_W16]; dNDCG = NDCG(b1 z_W16 + b2 composite) - NDCG(z_W16) at the coefficients of the eval half's own
    2-parameter fit (optimistic by one fitted mixing weight).  This is the RESIDUAL form: its sensitivity is calibrated in cal_posctrl.py.
Usage: blocks_read.py arms | blocks   -> out/blocks_arms.txt, out/blocks_lin.txt (+ .json)
"""
from cbi_common import *
import json, time
T0 = time.time()
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold')
lines = []
def say(s=''):
    print(s, flush=True); lines.append(s)

mode = sys.argv[1]
R25 = ROOT / 'nn_x/w25'; FB = ROOT / 'nn_x/w12/cb/famB'

if mode == 'arms':
    ARMS = {'A0_cb26repro (134K feeds, 177)': 'A0_cb26repro', 'A2c_cb26 (134K, 177 + 62 W25)': 'A2c_cb26', 'A1_base_xm (485K, 177)': 'A1_base_xm', 'A1_base_xm_s1 (seed 1)': 'A1_base_xm_s1',
            'A2_all_xm (485K, 239)': 'A2_all_xm', 'abl_loo_G1 (A2 minus G1)': 'abl_loo_G1', 'abl_loo_G3 (A2 minus G3)': 'abl_loo_G3', 'abl_add_G1 (A1 + G1)': 'abl_add_G1'}
    out = {}
    m50 = WV.M >= 50
    D = {}
    for lab, tag in ARMS.items():
        zv = WV.z(np.load(R25 / f'out/{tag}_val.npy')); zh = WH.z(np.load(R25 / f'out/{tag}_trH.npy'))
        dv, bv = nats(zv, WV.zw16, WV); dh, bh = nats(zh, WH.zw16, WH); dv50, _ = nats(zv, WV.zw16, WV, m50)
        ev = WV.ndcg(zv); eh = WH.ndcg(zh); D[tag] = (dv, dh, ev, eh, dv50)
        out[tag] = dict(level=(float(ev.mean()), float(eh.mean())), nats_val=sd_se(dv), nats_hold=sd_se(dh), nats_val_m50=sd_se(dv50))
        say(f'{lab:34s} level {ev.mean():.5f} / {eh.mean():.5f} | NATS over W16: val {f2(dv)}  holdout {f2(dh)}  val M>=50 {f2(dv50)}')
    say()
    for a, b, lab in (('A2c_cb26', 'A0_cb26repro', 'A2c - A0 (62 W25 columns on the cb26 rows)'), ('A2_all_xm', 'A1_base_xm', 'A2 - A1 (62 W25 columns on 485K feeds)'),
                      ('A2_all_xm', 'abl_loo_G1', 'A2 - loo_G1 (G1 LAE scores on top)'), ('A2_all_xm', 'abl_loo_G3', 'A2 - loo_G3 (G3 target statistics on top)'),
                      ('abl_add_G1', 'A1_base_xm', 'add_G1 - A1 (G1 alone)'), ('A1_base_xm_s1', 'A1_base_xm', 'A1 seed 1 - seed 0 (NOISE)')):
        x, y = D[a], D[b]
        say(f'{lab:46s} dNATS val {f2(x[0] - y[0])}  holdout {f2(x[1] - y[1])}  val M>=50 {f2(x[4] - y[4])} | dLEVEL val {f2(x[2] - y[2])}  holdout {f2(x[3] - y[3])}')
        out[lab] = dict(dnats_val=sd_se(x[0] - y[0]), dnats_hold=sd_se(x[1] - y[1]), dlevel_val=sd_se(x[2] - y[2]), dlevel_hold=sd_se(x[3] - y[3]))
    json.dump(out, open(CBI_OUT + '/blocks_arms.json', 'w'), indent=1)
    open(CBI_OUT + '/blocks_arms.txt', 'w').write('\n'.join(lines) + f'\n# elapsed {time.time() - T0:.0f} s\n')
    sys.exit()

# ---------------- block loaders: dict name -> (val array, holdout array) ----------------
def feat(name, w):
    for d in (FEATS, FEATS_NEW):
        p = d / ('val' if w.sp == 'val' else 'train') / f'{name}.npy'
        if p.exists():
            x = np.load(p, mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
    raise FileNotFoundError(name)

rows25 = np.load(R25 / 'data/train/rows.npy', mmap_mode='r'); pos25 = np.searchsorted(rows25, WH.rows); assert (np.asarray(rows25[pos25]) == WH.rows).all()
def w25(name, w):
    if w.sp == 'val': return np.load(R25 / f'data/val/{name}.npy')
    return np.asarray(np.load(R25 / f'data/train/{name}.npy', mmap_mode='r')[pos25], np.float32)

FBN = {b: json.load(open(FB / f'feat/{b}_names.json')) for b in 'abcd'}
_fb = {}
def famb(block, cols, w, lag=False):
    k = (block, w.sp, lag)
    if k not in _fb: _fb[k] = np.load(FB / f"feat/{'lag_' if lag else ''}{block}_{'val' if w.sp == 'val' else 'hold'}.npy", mmap_mode='r')
    return [np.asarray(_fb[k][:, FBN[block].index(c)], np.float32) for c in cols]

G = json.load(open(R25 / 'data/groups.json'))
TTE = 'tte_c_ctr,tte_it_coec,tte_it_ctr,tte_it_imp,tte_it_pos,tte_u_coec,tte_u_ctr,tte_u_imp,tte_uc_ctr,tte_uc_imp,tte_ui_imp,tte_ui_pos'.split(',')
A_CLICK = [c for c in FBN['a'] if any(s in c for s in ('pos', 'coec'))]; A_FREE = [c for c in FBN['a'] if c not in A_CLICK]
B_ALL = FBN['b']
BLOCKS = [  # (key, description, loader -> list of (name, val col, hold col))
    ('tte12', 'the 12 tte_* target statistics of the 177 tree features (in B0 already)', lambda w: [(c, feat(c, w)) for c in TTE]),
    ('tte_item5', 'tte_it_{imp,pos,ctr,coec} + tte_c_ctr (item / category level)', lambda w: [(c, feat(c, w)) for c in TTE if c.startswith(('tte_it', 'tte_c'))]),
    ('tte_user7', 'tte_u_*, tte_uc_*, tte_ui_* (same-user earlier feeds)', lambda w: [(c, feat(c, w)) for c in TTE if c.startswith(('tte_u_', 'tte_uc', 'tte_ui'))]),
    ('xp_imp_c50', 'ONE exposure count (pop1 view input)', lambda w: [('xp_imp_c50', feat('xp_imp_c50', w))]),
    ('xp_ctr_c', 'ONE item click rate (pop2 view input)', lambda w: [('xp_ctr_c', feat('xp_ctr_c', w))]),
    ('famB_a_all', 'block a: the user own earlier impressions (39)', lambda w: list(zip(FBN['a'], famb('a', FBN['a'], w)))),
    ('famB_a_click', 'block a, click statistics only', lambda w: list(zip(A_CLICK, famb('a', A_CLICK, w)))),
    ('famB_a_free', 'block a, click-free columns (impression counts, cosines to shown items, recency)', lambda w: list(zip(A_FREE, famb('a', A_FREE, w)))),
    ('famB_b', 'block b: transition click rates + user-cluster x item-cluster lifts (25)', lambda w: list(zip(B_ALL, famb('b', B_ALL, w)))),
    ('famB_c_density', 'block c label-free pair kn_cos1, kn_cos20', lambda w: list(zip(['kn_cos1', 'kn_cos20'], famb('c', ['kn_cos1', 'kn_cos20'], w)))),
    ('famB_c_click', 'block c label-aware kn_coec5, kn_coec20, kn_wcoec20', lambda w: list(zip(['kn_coec5', 'kn_coec20', 'kn_wcoec20'], famb('c', ['kn_coec5', 'kn_coec20', 'kn_wcoec20'], w)))),
    ('famB_c_ex', 'block c exposure of the neighbours kn_ex20', lambda w: list(zip(['kn_ex20'], famb('c', ['kn_ex20'], w)))),
    ('famB_d_density', 'block d label-free: dup_n99/97/95/90, kn_cos2/5/10', lambda w: list(zip(FBN['d'][:7], famb('d', FBN['d'][:7], w)))),
    ('famB_d_click', 'block d label-aware dup_coec97/95 + their exposure', lambda w: list(zip(FBN['d'][7:], famb('d', FBN['d'][7:], w)))),
    ('famB_cd_density', 'all 9 label-free density columns of c + d', lambda w: list(zip(['kn_cos1', 'kn_cos20'] + FBN['d'][:7], famb('c', ['kn_cos1', 'kn_cos20'], w) + famb('d', FBN['d'][:7], w)))),
    ('w25_G1', 'W25 G1: group-level EASE-DAN / RDLAE / EASE scores (15, label-free)', lambda w: [(c, w25(c, w)) for c in G['G1']]),
    ('w25_G5', 'W25 G5: recency / type-weighted LAE scores (11, label-free)', lambda w: [(c, w25(c, w)) for c in G['G5']]),
    ('w25_G2', 'W25 G2: group contact ratio x LAE score (6)', lambda w: [(c, w25(c, w)) for c in G['G2']]),
    ('w25_G3', 'W25 G3: contact-ratio target statistics incl. once-per-user (26)', lambda w: [(c, w25(c, w)) for c in G['G3']]),
    ('w25_G3_item', 'W25 G3, item-level columns (w25_it_*)', lambda w: [(c, w25(c, w)) for c in G['G3'] if c.startswith('w25_it_')]),
    ('w25_G3_group', 'W25 G3, category / cluster-level columns (w25_c_*, w25_g_*)', lambda w: [(c, w25(c, w)) for c in G['G3'] if c.startswith(('w25_c_', 'w25_g_'))]),
    ('w25_G3_user', 'W25 G3, user-level columns (w25_u_*, w25_uc_*, w25_ucl_*)', lambda w: [(c, w25(c, w)) for c in G['G3'] if c.startswith(('w25_u_', 'w25_uc_', 'w25_ucl_'))]),
    ('w25_G4', 'W25 G4: last-contact crosses (4)', lambda w: [(c, w25(c, w)) for c in G['G4']]),
    ('w25_Gexcl', 'W25 columns excluded by PSI > 0.5 (11; information only: parity-broken)', lambda w: [(c, w25(c, w)) for c in G['G_excl']]),
]
LAG = [('famB_a_all_LAG', 'block a rebuilt with the 6-9 d test lag (holdout rows only)', lambda w: list(zip(FBN['a'], famb('a', FBN['a'], w, True)))),
       ('famB_b_LAG', 'block b rebuilt with the test lag (holdout rows only)', lambda w: list(zip(B_ALL, famb('b', B_ALL, w, True)))),
       ('famB_c_click_LAG', 'block c label-aware, test lag (holdout rows only)', lambda w: list(zip(['kn_coec5', 'kn_coec20', 'kn_wcoec20'], famb('c', ['kn_coec5', 'kn_coec20', 'kn_wcoec20'], w, True))))]


def design(cols, w):
    X = []; nm = []
    for c, x in cols:
        x = np.asarray(x, np.float64); ok = np.isfinite(x); z = w.zcol(x)
        if z.any(): X.append(z); nm.append(c)
        if 0.005 < 1 - ok.mean() < 0.995:
            zi = w.zcol(ok.astype(np.float64))
            if zi.any(): X.append(zi); nm.append(c + ':has')
    # drop exact duplicates of missing flags (many columns share one NaN pattern)
    keep = []; seen = []
    for j, n_ in enumerate(nm):
        if n_.endswith(':has'):
            sig = (round(float(X[j][::997].sum()), 6), round(float(np.abs(X[j][::991]).sum()), 6))
            if sig in seen: continue
            seen.append(sig)
        keep.append(j)
    return np.column_stack([X[j] for j in keep]), [nm[j] for j in keep]


def read(Xe, be, w, feeds, base):
    """composite = Xe @ be (coefficients from elsewhere); NATS and dNDCG on the feeds given."""
    comp = Xe @ be; d, b2 = nats(comp, base, w, feeds)
    mix = w.ndcg(b2[0] * base + b2[1] * comp) - w.ndcg(base)
    return d, (mix if feeds is None else mix[feeds]), float(b2[1])


res = {}
tree_b0 = {'val': WV.z(np.load(ROOT / 'nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy')), 'hold': WH.z(np.load(ROOT / 'nn_x/w12/cb/infra/out/ts_lr020_s0_hold.npy'))}
say('key | n columns (+ missing flags) | holdout: NATS xfit, dNDCG xfit, beta | val: NATS xfit, dNDCG xfit, beta | val with holdout-fitted coefficients: NATS, dNDCG, beta | over [W16, B0] xfit: holdout NATS, val NATS')
for key, desc, loader in BLOCKS + LAG:
    t1 = time.time(); r = {'desc': desc}
    lag = key.endswith('_LAG')
    Xh, nm = design(loader(WH), WH); r['ncol'] = len(nm)
    bh_all = None
    out_ = {}
    wits = [(WH, Xh)] if lag else [(WH, Xh), (WV, None)]
    for w, X in wits:
        if X is None:
            X, nmv = design(loader(w), w)
            if nmv != nm:   # align the val design to the holdout design (a flag may be missing on one side)
                Xa = np.zeros((w.n, len(nm)));
                for j, n_ in enumerate(nm):
                    if n_ in nmv: Xa[:, j] = X[:, nmv.index(n_)]
                X = Xa
        half = w.halves(0); d = np.zeros(w.nf); mx = np.zeros(w.nf); d2 = np.zeros(w.nf); betas = []
        for h in (True, False):
            fit = half == h
            b, _ = clogit(np.column_stack([w.zw16, X]), w, fit, ridge=1e-6)
            dd, mm, bb = read(X, b[1:], w, ~fit, w.zw16); d[~fit] = dd; mx[~fit] = mm; betas.append(bb)
            b3, _ = clogit(np.column_stack([w.zw16, tree_b0[w.sp], X]), w, fit, ridge=1e-6)
            comp = X @ b3[2:]; dd2, _ = nats(comp, np.column_stack([w.zw16, tree_b0[w.sp]]), w, ~fit); d2[~fit] = dd2
        out_[w.sp] = dict(nats=sd_se(d), dndcg=sd_se(mx), beta=float(np.mean(betas)), nats_over_w16_b0=sd_se(d2))
        if w.sp == 'hold':
            bh_all, _ = clogit(np.column_stack([w.zw16, X]), w, None, ridge=1e-6)
        else:
            dd, mm, bb = read(X, bh_all[1:], w, None, w.zw16); out_['val_holdcoef'] = dict(nats=sd_se(dd), dndcg=sd_se(mm), beta=bb)
    r.update(out_); res[key] = r
    h = out_['hold']; s = f"{key:18s} | {len(nm):3d} | hold {h['nats'][0]:+.5f} +- {h['nats'][1]:.5f}, {h['dndcg'][0]:+.5f} +- {h['dndcg'][1]:.5f}, b {h['beta']:+.2f}"
    if 'val' in out_:
        v = out_['val']; vh = out_['val_holdcoef']
        s += f" | val {v['nats'][0]:+.5f} +- {v['nats'][1]:.5f}, {v['dndcg'][0]:+.5f} +- {v['dndcg'][1]:.5f}, b {v['beta']:+.2f} | val(hold coef) {vh['nats'][0]:+.5f} +- {vh['nats'][1]:.5f}, {vh['dndcg'][0]:+.5f} +- {vh['dndcg'][1]:.5f}, b {vh['beta']:+.2f}"
        s += f" | over [W16,B0]: hold {h['nats_over_w16_b0'][0]:+.5f} +- {h['nats_over_w16_b0'][1]:.5f}, val {v['nats_over_w16_b0'][0]:+.5f} +- {v['nats_over_w16_b0'][1]:.5f}"
    else:
        s += f" | over [W16,B0]: hold {h['nats_over_w16_b0'][0]:+.5f} +- {h['nats_over_w16_b0'][1]:.5f}"
    say(s + f'   [{desc}; {time.time() - t1:.0f} s]')
    json.dump(res, open(CBI_OUT + '/blocks_lin.json', 'w'), indent=1)
open(CBI_OUT + '/blocks_lin.txt', 'w').write('\n'.join(lines) + f'\n# elapsed {time.time() - T0:.0f} s\n')
