"""e2d: correlation facts quoted in the report (train-only val z): held-out against full-data students of the same / a different seed number; fine-tuned branch correlations; branch-to-branch correlation.
Output out/misc_corr.json."""
from ens_common import *
m = Meta(); st = Store(m); r = {}
C = np.corrcoef(np.vstack([st.mat('to_val', W16), st.mat('to_val', FD16)]))
r['to_W16xFD16_same_seed'] = float(np.mean([C[i, 16 + i] for i in range(16)])); r['to_W16xFD16_diff_seed_same_arm'] = float(np.mean([C[i, 16 + j] for i in range(16) for j in range(16) if i != j and (i < 8) == (j < 8)]))
r['to_within_W16'] = float(C[:16, :16][np.triu_indices(16, 1)].mean()); r['to_within_FD16'] = float(C[16:, 16:][np.triu_indices(16, 1)].mean())
F = np.corrcoef(np.vstack([st.mat('ft_val', W16), st.mat('ft_val', FD16)])); r['ft_within_W16'] = float(F[:16, :16][np.triu_indices(16, 1)].mean()); r['ft_within_FD16'] = float(F[16:, 16:][np.triu_indices(16, 1)].mean())
r['ft_W16xFD16'] = float(F[:16, 16:].mean())
X = np.corrcoef(np.vstack([st.mat('to_val', FD16), st.mat('ft_val', FD16)])); r['FD16_member_to_x_own_ft'] = float(np.mean([X[i, 16 + i] for i in range(16)])); r['FD16_to_x_other_ft'] = float(np.mean([X[i, 16 + j] for i in range(16) for j in range(16) if i != j]))
r['FD16_pool_to_x_pool_ft'] = float(np.corrcoef(st.pool('to_val', FD16), st.pool('ft_val', FD16))[0, 1])
Y = np.corrcoef(np.vstack([st.mat('to_val', FD16), st.mat('to_val', FDXHZ)])); r['to_FD16xFDXHZ_same_root'] = float(np.mean([Y[i, 16 + i] for i in range(16)])); r['to_FD16xFDXHZ_diff_root'] = float(np.mean([Y[i, 16 + j] for i in range(16) for j in range(16) if i != j]))
Y = np.corrcoef(np.vstack([st.mat('ft_val', FD16), st.mat('ft_val', FDXHZ)])); r['ft_FD16xFDXHZ_same_root'] = float(np.mean([Y[i, 16 + i] for i in range(16)])); r['ft_FD16xFDXHZ_diff_root'] = float(np.mean([Y[i, 16 + j] for i in range(16) for j in range(16) if i != j]))
Pp = np.corrcoef(np.vstack([st.mat('to_val', FD16), st.mat('to_val', FDP12), st.mat('to_val', FDPF), st.mat('to_val', FDP2)])); r['to_within_FDP12'] = float(Pp[16:28, 16:28][np.triu_indices(12, 1)].mean()); r['to_FD16xFDP12'] = float(Pp[:16, 16:28].mean())
r['to_FDP12xFDPF'] = float(Pp[16:28, 28:36].mean()); r['to_FDP12xFDP2'] = float(Pp[16:28, 36:40].mean()); r['to_FD16xFDPF'] = float(Pp[:16, 28:36].mean()); r['to_within_FDPF'] = float(Pp[28:36, 28:36][np.triu_indices(8, 1)].mean())
json.dump(r, open(OUT / 'misc_corr.json', 'w'), indent=0); [print(f'{k:36s} {v:.4f}') for k, v in r.items()]
