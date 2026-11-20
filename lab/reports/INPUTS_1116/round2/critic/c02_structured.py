"""CRITIC2 c02: STRUCTURED placebos for the lens (labels are opened). A within-feed permutation destroys every relation between the column and the
candidate; a density column is a function of the candidate's CONTENT VECTOR, which the ranker reads raw and the lens's control block does not hold.
Question: how does the calibrated score test (null model = pool + own inputs + nx3) read content-derived columns that carry NO catalogue or exposure
information by construction?
  RD   cosine of the content vector to a random direction (100 draws)
  RA   cosine of the content vector to the content vector of a random train item (100 draws; 'similarity to an arbitrary anchor')
  CL   a random N(0,1) effect per content cluster, 4096 / 1024 / 256 clusters (100 draws each)
Entered as within-feed z (clip 5), as kn1_5:z is. Printed: sd, max |cz|, share |cz| >= 2.5, share beyond the read of kn1_5:z, on H (holdout, lag view) and
T (val rows; these columns have no lag, so T = natural val rows with the twin nx3 in the null model, as c01's T).
usage: c02_structured.py    output out/c02_structured.txt"""
import json
from cc_common import *
P = Log(OUT / 'c02_structured.txt')
V = rs.Views(None)
ND = 100
CLU = ROOT / 'data/clu_work'


def emb_table(split):
    d = ROOT / f'data/competition/{split}/item_embeddings'; mt = json.loads((d / 'data.meta').read_text())
    return np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(mt['shape']))


def summarize(tag, cz_, read):
    a = np.abs(np.array(cz_))
    P(f'  {tag:34s}: n {len(a)}, mean {np.mean(cz_):+.2f}, sd {np.std(cz_):.2f}, rms {np.sqrt(np.mean(a ** 2)):.2f}, max |cz| {a.max():.2f}, share |cz| >= 2.5: {np.mean(a >= 2.5):.3f}, share >= |kn1_5:z read| ({abs(read):.2f}): {np.mean(a >= abs(read)):.3f}')
    return float(np.sqrt(np.mean(a ** 2)))


for wit in ('H', 'T'):
    sp = 'hold' if wit == 'H' else 'val'; q = 'hold' if wit == 'H' else 'val_twin'
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']
    rows = V.rows(q)[0]; fsp = 'train' if wit == 'H' else 'val'
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        view = np.repeat(V.A['train']['slot'][V.hold] >= 6, M)
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
        view = np.ones(len(rows), bool)
    N = MyNull(D, cl + [nz, nr])
    read = N.cz(form(rget(q, 'kn1_5'), off, M, None, 'z'))[0]
    P(f'\n=== witness {wit}: feeds {D["n"]}, rows {len(rows)}; kn1_5:z reads {read:+.2f} here ===')
    ei = np.asarray(np.load(FLAT / fsp / 'cand_emb_idx.npy', mmap_mode='r')[rows]).astype(np.int64)
    ET = emb_table(fsp); E = np.zeros((len(rows), 128), np.float32); ok = ei >= 0
    o_ = np.argsort(ei[ok]); E[np.nonzero(ok)[0][o_]] = np.asarray(ET[ei[ok][o_]], np.float32)
    nrm = np.sqrt((E ** 2).sum(1)); nrm[nrm == 0] = 1.0; E /= nrm[:, None]
    P(f'  content vectors gathered; rows without a vector {np.mean(~ok):.4f}')
    rng = np.random.default_rng(777 if wit == 'H' else 778)

    def enter(x):
        x = np.where(ok, x, np.nan).astype(np.float64)
        if wit == 'H':
            x = np.where(view, x, np.nan)       # as the rescue arrays: no value on a feed without the view
        return form(x, off, M, None, 'z')

    Wd = rng.standard_normal((128, ND)).astype(np.float32); Wd /= np.sqrt((Wd ** 2).sum(0))[None]
    XR = E @ Wd
    r1 = summarize('RD cosine to a random direction', [N.cz(enter(XR[:, k]))[0] for k in range(ND)], read)
    ET_tr = emb_table('train'); anc = np.asarray(ET_tr[np.sort(rng.choice(ET_tr.shape[0], ND, replace=False))], np.float32); anc /= np.maximum(np.sqrt((anc ** 2).sum(1)), 1e-9)[:, None]
    XR = E @ anc.T
    r2 = summarize('RA cosine to a random train item', [N.cz(enter(XR[:, k]))[0] for k in range(ND)], read)
    del XR
    for kc in (4096, 1024, 256):
        c = np.asarray(np.load(CLU / f'cand_cl_{kc}_{fsp}.npy', mmap_mode='r')[rows]).astype(np.int64)
        cz_ = []
        for k in range(ND):
            eff = rng.standard_normal(kc + 1); x = np.where(c >= 0, eff[np.maximum(c, 0)], np.nan)
            if wit == 'H':
                x = np.where(view, x, np.nan)
            cz_.append(N.cz(form(x, off, M, None, 'z'))[0])
        summarize(f'CL random effect per cluster, k = {kc}', cz_, read)
    # the unstructured reference on the same null model
    xk = form(rget(q, 'kn1_5'), off, M, None, 'z'); cz_ = [N.cz(xk[perm_order(N.fid, rng)])[0] for _ in range(ND)]
    summarize('within-feed permutations of kn1_5:z', cz_, read)
    del N, D, E
P('DONE')
