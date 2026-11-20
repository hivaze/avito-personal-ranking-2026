"""Pre-declared final overnight decision (CURRENT_STATUS 23:45 protocol, run 08:20 UTC 11-12): holdout-selected pool
Pall with F7 replaced by F7x (decide3 11:00 pre-declared candidate; holdout tie with Pall+F7x broken on val, 11:35), spec 0.5*z(ft) + 0.5*z(to)."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from stack_val import feed_rank_z
from make_sub import make_sub
P = Path('<repo>/preds')
F6 = ['v13_lr1_mix1','v14_s92','v14_s93','v14_s94','v14_s95','v14_s96']
FAM = {'F5cd': ['v15_cd_all_e10','v15_cd_all_s81','v15_cd_all_s82','v15_cd_allprew_s83','v15_cd_allprew_s84','v15_cd_allprew_s85'],
       'F6': F6, 'F7x': [f'r6_cd_m02_r{s}' for s in range(101,108)], 'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401,407)],
       'F10': [f'v23_gru_pre2s_s{s}' for s in range(501,507)], 'F12': [f'v26_d192_pre2s_s{s}' for s in range(701,708)]}
ms = [m for f in FAM.values() for m in f]; print(len(ms), 'members')
for split, ftn, ton in [('val','n2cpv3_val_oof','trainonly_val'), ('test','n2cpv3_test','trainonly_test')]:
    f, M, o = feed_meta(split); z = lambda x: feed_rank_z(o, M, np.asarray(x, np.float32))[:, 1]
    np.save(P/f'ens_PF7x_ft_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ftn}.npy')) for m in ms], 0).astype(np.float32))
    np.save(P/f'ens_PF7x_to_{split}.npy', np.mean([z(np.load(P/f'nn_{m}_{ton}.npy')) for m in ms], 0).astype(np.float32))
ft_, Mt, ot = feed_meta('test'); zt = lambda n: feed_rank_z(ot, Mt, np.load(P/f'{n}.npy'))[:, 1]
make_sub((0.5*zt('ens_PF7x_ft_test') + 0.5*zt('ens_PF7x_to_test')).astype(np.float32), '<repo>/subs/cand_PF7x_memeq_6fam.parquet')
print('written subs/cand_PF7x_memeq_6fam.parquet')
