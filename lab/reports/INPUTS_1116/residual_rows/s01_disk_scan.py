"""INVENTORY B, step 1: scan the project (and the session scratchpad of the CCB batch) for per-candidate arrays.
An array counts as a per-split array when its first dimension equals the candidate count of a split:
train 246,745,836 | holdout 4,630,966 | val 1,389,971 | test 2,904,806 (or the feed counts 3,151,208 / 30,000 / 14,423 / 28,595).
Reads only .npy headers (mmap) and .npz member headers. CPU only, read-only. Output: out/disk_scan.tsv"""
import os, sys, zipfile, numpy as np
from numpy.lib import format as npf
ROWS = {246745836: 'train', 4630966: 'hold', 1389971: 'val', 2904806: 'test', 3151208: 'trainF', 30000: 'holdF', 14423: 'valF', 28595: 'testF',
        246745836 - 4630966: 'train-hold', 1389971 + 2904806: 'val+test', 1389971 + 4630966: 'val+hold'}
roots = sys.argv[1:]
out = open('<repo>/reports/INPUTS_1116/residual_rows/out/disk_scan.tsv', 'a')
def hdr(fp):
    v = npf.read_magic(fp)
    return npf.read_array_header_1_0(fp) if v == (1, 0) else npf.read_array_header_2_0(fp)
n = 0
for root in roots:
    for dp, dn, fn in os.walk(root):
        if '/ckpt' in dp or '__pycache__' in dp: continue
        for f in fn:
            p = os.path.join(dp, f)
            try:
                if f.endswith('.npy'):
                    with open(p, 'rb') as fp: shape, _, dt = hdr(fp)
                    if shape and shape[0] in ROWS:
                        out.write(f'{ROWS[shape[0]]}\t{shape}\t{dt}\t{os.path.getmtime(p):.0f}\t{p}\n'); n += 1
                elif f.endswith('.npz'):
                    with zipfile.ZipFile(p) as z:
                        for nm in z.namelist():
                            with z.open(nm) as fp:
                                shape, _, dt = hdr(fp)
                            if shape and shape[0] in ROWS:
                                out.write(f'{ROWS[shape[0]]}\t{shape}\t{dt}\t{os.path.getmtime(p):.0f}\t{p}::{nm}\n'); n += 1
            except Exception as e:
                pass
print(roots, n)
