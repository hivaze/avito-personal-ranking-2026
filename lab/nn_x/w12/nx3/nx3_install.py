"""nx3: INSTALL the files of record as train-stage inputs (BUILD_SPEC 4; model_inputs.md 2.2; precedent nn_x/abl/pop1/build_xp_imp_c50.py).
Runs ONLY if nn_x/w12/nx3/out/nx3_gates.txt ends with "STEP 0 VERDICT (amended rule A2): PASS".
  data/feats/{train,val,test}/{nx_rp3_z,nx_rp3_rk}.npy      NEW files, exclusive create (never overwrites)
  nn/feat_cache/{train,val,test}/{nx_rp3_z,nx_rp3_rk}.npy   NEW hard links of them (link count 2), the way every other xt column is exposed
has:nx_rp3_z needs no file: the loader (nn_x/abl/data.py attach_xt) derives it as the non-NaN indicator of nx_rp3_z.
The stale-val twins stay in nn_x/w12/nx3/staleval/ and are never linked into nn/feat_cache. CPU only. Output out/install.txt (exclusive create)."""
from nx3_common import *
P = Log(OUT / f'install{TAG}.txt')
ver = [ln for ln in open(OUT / f'nx3_gates{TAG}.txt') if 'STEP 0 VERDICT' in ln]
assert len(ver) == 1 and ver[0].strip().endswith('STEP 0 VERDICT (amended rule A2): PASS'), f'STEP 0 is not PASS: {ver}'
FEATS = ROOT / 'data/feats'; FC = ROOT / 'nn/feat_cache'
NAMES = ('nx_rp3_z', 'nx_rp3_rk')
for sp in SPLITS:
    for n in NAMES:
        for d in (FEATS, FC, ROOT / 'data/feats_new'):
            assert not os.path.lexists(d / sp / f'{n}.npy'), f'{d / sp / n}.npy exists: refusing'
for sp in SPLITS:
    for n in NAMES:
        x = np.load(FILES_DIR / sp / f'{n}.npy'); assert x.dtype == np.float32 and x.shape == (NROWS[sp],)
        dst, lnk = FEATS / sp / f'{n}.npy', FC / sp / f'{n}.npy'
        save_new(dst, x); os.link(dst, lnk)
        sd, sl = os.stat(dst), os.stat(lnk); assert sd.st_ino == sl.st_ino and sd.st_nlink == 2
        back = np.load(lnk); assert back.dtype == np.float32 and np.array_equal(back, x, equal_nan=True)
        m1, m2 = md5(str(dst)), md5(str(FILES_DIR / sp / f'{n}.npy')); assert m1 == m2
        P(f'{sp}: wrote {dst} ({x.shape[0]:,} rows, {os.path.getsize(dst)} bytes, md5 {m1} = staged file), hard link {lnk} (inode {sd.st_ino}, nlink 2)')
# what the loader will hand to the model (attach_xt transform, CPU emulation by the same rules: has: = non-NaN indicator; *_z clip +-5; *_rk raw; NaN -> 0)
for sp in SPLITS:
    z = np.load(FC / sp / 'nx_rp3_z.npy'); rk = np.load(FC / sp / 'nx_rp3_rk.npy'); hasv = (~np.isnan(z)).astype(np.float32)
    zt = np.nan_to_num(np.clip(z, -5, 5), nan=0.0); rt = np.nan_to_num(rk, nan=0.0)
    P(f'{sp}: as the loader reads them: has:nx_rp3_z mean {hasv.mean():.4f}; nx_rp3_z mean {zt.mean():+.4f} sd {zt.std():.4f} min {zt.min():.3f} max {zt.max():.3f}; nx_rp3_rk mean {rt.mean():+.5f} sd {rt.std():.4f} min {rt.min():.3f} max {rt.max():.3f}; fp16 round-trip max |err| z {np.abs(zt.astype(np.float16).astype(np.float32) - zt).max():.2e}')
P('INSTALLED')
