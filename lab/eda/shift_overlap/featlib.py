"""Feed- and candidate-level descriptive features for adversarial validation."""
from common import *
NID = 111_400_000

def subset(s, fsel=None):
    f = feeds(s)
    if fsel is not None:
        f = f[np.sort(fsel)]
    M = f['M'].to_numpy(); H = f['H'].to_numpy()
    co = f['cand_off'].to_numpy(); ho = f['hist_off'].to_numpy()
    cidx = np.repeat(co, M) + (np.arange(M.sum()) - np.repeat(np.cumsum(M) - M, M))
    hidx = np.repeat(ho, H) + (np.arange(H.sum()) - np.repeat(np.cumsum(H) - H, H))
    C = {c: np.asarray(cand(s, c)[cidx]) for c in ['item_id', 'category_id', 'f1', 'f2', 'f3', 'f4', 'f5', 'emb_idx']}
    if s != 'test':
        C['target'] = np.asarray(cand(s, 'target')[cidx])
    Hh = {c: np.asarray(hist(s, c)[hidx]) for c in ['item_id', 'category_id', 'event_type', 'event_source', 'emb_idx', 'hpos']}
    ue = np.load(f'{FLAT}/{s}/user_emb.npy', mmap_mode='r')[f['fidx'].to_numpy()].astype(np.float32)
    cf = np.repeat(np.arange(len(f)), M); hf = np.repeat(np.arange(len(f)), H)
    return dict(f=f, M=M, H=H, C=C, Hh=Hh, ue=ue, cf=cf, hf=hf)

def feed_features(d, counts=None, train_users=None, with_uemb=False):
    f, M, H, C, Hh, cf, hf = d['f'], d['M'], d['H'], d['C'], d['Hh'], d['cf'], d['hf']
    n = len(f)
    X = {'M': M.astype(float), 'H': H.astype(float), 'user0': (f['user_id'].to_numpy() == 0).astype(float)}
    ue = d['ue']
    X['uemb_zero'] = (np.abs(ue).sum(1) == 0).astype(float)
    X['uemb_norm'] = np.linalg.norm(ue, axis=1)
    X['uemb_mean'] = ue.mean(1)
    if with_uemb:
        for j in range(ue.shape[1]):
            X[f'ue{j}'] = ue[:, j]
    Hs = np.maximum(H, 1)
    et = Hh['event_type'].astype(int)
    for e in range(1, 11):
        X[f'h_et{e}'] = np.bincount(hf, weights=(et == e), minlength=n) / Hs
    es = Hh['event_source'].astype(int)
    for e in range(0, 8):
        X[f'h_es{e}'] = np.bincount(hf, weights=(es == e), minlength=n) / Hs
    X['h_embmiss'] = np.bincount(hf, weights=(Hh['emb_idx'] < 0), minlength=n) / Hs
    # distinct categories in history / candidates
    def nuniq(fid, v):
        k = np.unique(fid.astype(np.int64) * 256 + v.astype(np.int64))
        return np.bincount((k // 256).astype(np.int64), minlength=n).astype(float)
    X['h_ncat'] = nuniq(hf, Hh['category_id'])
    X['c_ncat'] = nuniq(cf, C['category_id'])
    for c in ['f1', 'f2', 'f3', 'f4', 'f5']:
        v = C[c].astype(float)
        m1 = np.bincount(cf, weights=v, minlength=n) / M
        m2 = np.bincount(cf, weights=v * v, minlength=n) / M
        X[f'c_{c}_mean'] = m1; X[f'c_{c}_std'] = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    X['c_f2_zero'] = np.bincount(cf, weights=(C['f2'] == 0), minlength=n) / M
    X['c_f1_min'] = np.bincount(cf, weights=(C['f1'] <= 0.0179), minlength=n) / M
    X['c_embmiss'] = np.bincount(cf, weights=(C['emb_idx'] < 0), minlength=n) / M
    # share of candidates whose category occurs in the user's history
    if counts is not None:
        it = C['item_id']
        X['c_in_train_cand'] = np.bincount(cf, weights=(counts['train_cand'][it] > 0), minlength=n) / M
        X['c_in_train_hist'] = np.bincount(cf, weights=(counts['train_hist'][it] > 0), minlength=n) / M
        X['c_log_train_hist'] = np.bincount(cf, weights=np.log1p(counts['train_hist'][it]), minlength=n) / M
        hi = Hh['item_id']
        X['h_in_train_hist'] = np.bincount(hf, weights=(counts['train_hist'][hi] > 0), minlength=n) / Hs
    if train_users is not None:
        X['user_in_train'] = np.isin(f['user_id'].to_numpy(), train_users).astype(float)
    X['hour'] = f['hour'].to_numpy().astype(float)
    return pl.DataFrame(X)

def cand_features(d, counts=None, emb=False, split=None, emb_dims=16):
    C = d['C']; cf = d['cf']
    X = {c: C[c].astype(np.float32) for c in ['f1', 'f2', 'f3', 'f4', 'f5']}
    X['category_id'] = C['category_id'].astype(np.int32)
    X['emb_missing'] = (C['emb_idx'] < 0).astype(np.float32)
    X['M'] = d['M'][cf].astype(np.float32)
    if counts is not None:
        it = C['item_id']
        X['log_train_cand'] = np.log1p(counts['train_cand'][it]).astype(np.float32)
        X['log_train_hist'] = np.log1p(counts['train_hist'][it]).astype(np.float32)
    if emb:
        mm = np.memmap(f'<repo>/data/competition/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r').reshape(-1, 128)
        ei = C['emb_idx']
        E = np.zeros((len(ei), 128), np.float32)
        ok = ei >= 0
        E[ok] = mm[ei[ok]].astype(np.float32)
        for j in range(emb_dims):
            X[f'e{j}'] = E[:, j]
        X['e_norm'] = np.linalg.norm(E, axis=1)
    return pl.DataFrame(X)
