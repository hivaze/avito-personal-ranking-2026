"""e8: markdown tables for the report from the json outputs (no new measurement except the frontier regression across arms). Output out/tables.md."""
from ens_common import *
import collections
J = lambda f: json.load(open(OUT / f)); L = []; P_ = L.append
inv = J('inventory.json')['members']; ia = J('inventory_arms.json'); arms = J('arms.json'); late = J('arms_late.json'); arms['arms'].update(late['arms']); law = J('law.json'); fl = J('file_law.json')
f5 = lambda x: '' if x is None else f'{x:.5f}'; pm = lambda t: f'{t[0]:+.5f} ± {t[1]:.5f}'
# ---------- T1 inventory (condensed): merge K / L lineages of the re-cool arms; classes
def cls(a):
    if re.match(r'^w5_(e10|fd10)_kd05(t25)?w192$', a): return '1 wide students (the recipe)'
    if a.startswith('w5_fdx'): return '2 all-data re-cools'
    if re.match(r'^w5_[kl]cd_', a): return '3 held-out re-cools of the 16 wide snapshots'
    if re.match(r'^abl_(e10|fd10)_(pop1kd|pop2kd|pfkd)', a): return '4 popularity-light students'
    if re.match(r'^w5_e10_(kd03|kd05|kd05t25|kd08|kd08t25|kd10|kx05|kx08)$', a): return '6 d128 students'
    if re.match(r'^w5_e10_(k|u20)', a): return '5 other d192 / d224 / d256 student arms'
    return '7 hard-label members'
rows = collections.defaultdict(list)
for a, r in ia.items():
    if a.endswith('np'): continue
    key = re.sub(r'^w5_[kl]cd_', 'w5_{k,l}cd_', a); rows[(cls(a), key)].append((a, r))
P_('### T1. Inventory by arm (10-epoch class; `out/inventory_arms.md` has all arms one by one)\n')
P_('| class | arm | members | seeds | all data | TO val + test | FT val + test | holdout file | TO val member | TO val arm pool | FT val arm pool | holdout arm pool (out of sample) |'); P_('|' + '---|' * 12)
for (c, key) in sorted(rows):
    rs = rows[(c, key)]; n = sum(r['n'] for _, r in rs); nft = sum(r['n_ft'] for _, r in rs); nh = sum(r['n_hold'] for _, r in rs); s = rs[0][1]['seeds']; fd = rs[0][1]['full_data']
    ss = f'{s[0]}-{s[-1]}' if len(s) > 2 else ','.join(s)
    mem = np.mean([r['to_member'] for _, r in rs]); pool = ' / '.join(f5(r['to_pool']) for _, r in rs); ftp = ' / '.join(f5(r.get('ft_pool')) for _, r in rs if r.get('ft_pool')); hp = ' / '.join(f5(r.get('hold_pool')) for _, r in rs if r.get('hold_pool') and not fd)
    P_(f'| {c[2:]} | {key} | {n} | {ss}{" (K and L)" if len(rs) > 1 else ""} | {"yes" if fd else "no"} | {n} | {nft} | {nh} | {mem:.5f} | {pool} | {ftp} | {hp} |')
P_('')
# ---------- T2 law
P_('### T2. Pool-size law N(n) = N_inf - c / n (random subsets; val unless stated)\n')
P_('| set | branch | N_inf | c | SE of c (feeds) | SE of c (member jackknife) | max residual | 16 -> 32 | 16 -> 64 | 16 -> 128 | 16 -> infinity | realised N(all) minus mean half-size pool (direct) |'); P_('|' + '---|' * 12)
for s in ('W16', 'FD16', 'W32'):
    for b, bn in (('to', 'train-only'), ('ft', 'fine-tuned'), ('fin', 'final 0.5 / 0.5'), ('hold', 'train-only, holdout')):
        if b not in law[s]: continue
        r = law[s][b]; g = r['gain_from_16']; d = r['direct_N_minus_half']
        P_(f'| {s} | {bn} | {r["all"]["a"]:.5f} | {r["all"]["c"]:.5f} | {r["all"]["se_c_feed"]:.5f} | {r["jk_se_c"]:.5f} | {r["all"]["max_resid"]:.5f} | {g["32"]["gain"]:+.5f} ± {g["32"]["se"]:.5f} | {g["64"]["gain"]:+.5f} ± {g["64"]["se"]:.5f} | {g["128"]["gain"]:+.5f} ± {g["128"]["se"]:.5f} | {g["inf"]["gain"]:+.5f} ± {g["inf"]["se"]:.5f} | {d["mean"]:+.5f} ± {d["se"]:.5f} |')
P_('')
P_('Curves E[N(n)] (final val): ' + ' | '.join(f'{s}: ' + ', '.join(f'n={n} {law[s]["fin"]["curve"][str(n)]:.5f}' for n in (1, 2, 4, 8, 16) + ((32,) if s == 'W32' else ())) for s in ('W16', 'FD16', 'W32')) + '\n')
# ---------- T3 file law
P_('### T3. Two-pool law at file level F(nb, np) = a - cb / nb - cp / np (all-data members: nb of 16 wide, np of 12 wide pop1; 40 random draws per cell)\n')
P_('| weights | a | cb | cp | max residual | 16+4 -> 16+12 | 16+12 -> 32+12 | 16+12 -> 16+24 | 16+12 -> 64+48 | 16+12 -> infinity | 16+4 -> infinity | one more base student at 16 | one more pop student at 12 |'); P_('|' + '---|' * 13)
for w in ('val', 'public', 'private'):
    r = fl[w]; f = r['forecast']
    P_(f'| {w if w == "val" else "pmix " + w} | {r["a"]:.5f} | {r["cb"]:.5f} ± {r["se"][1]:.5f} | {r["cp"]:.5f} ± {r["se"][2]:.5f} | {r["max_resid"]:.5f} | ' + ' | '.join(pm(f[k]) for k in ('16+4 -> 16+12', '16+12 -> 32+12', '16+12 -> 16+24', '16+12 -> 64+48', '16+12 -> inf', '16+4 -> inf')) + f' | {r["marginal_base_at16"]:+.6f} | {r["marginal_pop_at12"]:+.6f} |')
P_('')
# ---------- T4 second arm
P_('### T4. Second arm at equal size (train-only branch; val | in-period holdout; h = members per half)\n')
P_('| arm | n | member val \\| hold | corr within arm | corr arm x wide (different seed / root) | same seed / root | h | level vs h wide | replace half of a 2h wide pool (val) | (holdout) | pmix public / private | add all to W16 (val \\| hold) | within 0.001 |'); P_('|' + '---|' * 13)
ww = arms['wide_within']; pts = []
order = ['w5_e10_kd08w192', 'w5_e10_kd05w192gru', 'w5_e10_kd05w224', 'RC:ref@last', 'RC:ref2@last', 'RC:hz@last', 'RC:th@last', 'RC:xd30@last', 'RC:hzt@last', 'RC:hznFS@last', 'RC:hz2@last', 'RC:hz2nF@last', 'w5_e10_kwh6', 'w5_e10_kwr32', 'w5_e10_kwd2', 'w5_e10_kwsb2',
         'w5_e10_kdmixw192', 'w5_e10_kd05g2w192', 'w5_e10_kd03g2w192', 'w5_e10_kx05', 'w5_e10_kwwd05', 'w5_e10_kwh30', 'w5_e10_kd05', 'w5_e10_kd05t25', 'w5_e10_kd08', 'w5_e10_kd03', 'w5_e10_kd10', 'w5_e10_kx08', 'w5_e10_kwaux0', 'w5_e10_kwwd10', 'w5_e10_kw256d2',
         'abl_e10_pop1kd05w192', 'abl_e10_pop1kd05t25', 'r6_cd_m02', 'r6_cd_m02_p70k', 'w5_e10_base', 'v23_gru_pre2s', 'v17_pre2s', 'v26_d192_pre2s', 'v22_pre2s_lr1', 'abl_e10_pop1', 'abl_e10_pop2', 'abl_e10_xdrop_pop']
for a in order:
    r = arms['arms'].get(a)
    if r is None: P_(f'| {a} | not on disk |' + ' |' * 11); continue
    h = 8 if 'h8' in r else 4; q = r[f'h{h}']; near = abs(q['level_val'][0]) <= 0.001
    P_(f'| {a} | {r["n"]} | {r["member_val"]:.5f} \\| {r["member_hold"]:.5f} | {r["corr_within"]:.4f} | {r["corr_cross_diff"]:.4f} | {"" if r["corr_cross_same"] is None else format(r["corr_cross_same"], ".4f")} | {h} | {pm(q["level_val"])} \\| {pm(q["level_hold"])} | {pm(q["replace_val"])} | {pm(q["replace_hold"])} | '
       f'{q["replace_pub"][0]:+.5f} / {q["replace_priv"][0]:+.5f} | {pm(r["add_val"])} \\| {pm(r["add_hold"])} | {"yes" if near else "no"} |')
for a, r in arms['arms'].items():
    q = r['h4']; pts.append((a, q['level_val'][0], q['level_hold'][0], ww - r['corr_cross_diff'], q['replace_val'][0], q['replace_hold'][0], q['replace_val'][1], q['replace_hold'][1]))
P_(f'\nMean pairwise correlation inside the 16 wide students: {ww:.4f}.\n')
# frontier regression: replace(h = 4) = alpha * level gap + beta * (rho_wide - rho_cross), all arms, both witnesses
X = np.array([[p[1], p[3]] for p in pts]); yv = np.array([p[4] for p in pts]); cv, *_ = np.linalg.lstsq(X, yv, rcond=None); rv = yv - X @ cv
Xh = np.array([[p[2], p[3]] for p in pts]); yh = np.array([p[5] for p in pts]); ch, *_ = np.linalg.lstsq(Xh, yh, rcond=None); rh = yh - Xh @ ch
# without the popularity-view arms (the holdout is biased against them)
keep = np.array([not re.search(r'pop', p[0]) for p in pts]); ch2, *_ = np.linalg.lstsq(Xh[keep], yh[keep], rcond=None); cv2, *_ = np.linalg.lstsq(X[keep], yv[keep], rcond=None)
fr = dict(n_arms=len(pts), val=dict(alpha=float(cv[0]), beta=float(cv[1]), resid_sd=float(rv.std()), r2=float(1 - rv.var() / yv.var())), hold=dict(alpha=float(ch[0]), beta=float(ch[1]), resid_sd=float(rh.std()), r2=float(1 - rh.var() / yh.var())),
          val_nopop=dict(alpha=float(cv2[0]), beta=float(cv2[1])), hold_nopop=dict(alpha=float(ch2[0]), beta=float(ch2[1])))
json.dump(fr, open(OUT / 'frontier.json', 'w'), indent=0)
P_(f'Frontier regression over {len(pts)} arms (h = 4): replace = alpha x (arm level - wide level) + beta x (rho_wide - rho_cross). val: alpha {cv[0]:.3f}, beta {cv[1]:.4f}, residual sd {rv.std():.5f}, R2 {fr["val"]["r2"]:.2f}; '
   f'holdout: alpha {ch[0]:.3f}, beta {ch[1]:.4f}, residual sd {rh.std():.5f}, R2 {fr["hold"]["r2"]:.2f}; without the popularity-view arms: val alpha {cv2[0]:.3f} beta {cv2[1]:.4f}, holdout alpha {ch2[0]:.3f} beta {ch2[1]:.4f}.\n')
# ---------- T5 two-level
tl = arms['two_level']; P_('### T5. Cooldown branches of one plateau snapshot: N(R roots, B branches) = a - c_root / R - c_branch / (R B)\n')
P_('| branches | witness | a | c_root | c_branch | max residual | observed 16x1 | observed 16 x all | 16x1 -> 16x2 | 16x1 -> 16x3 | 16x1 -> 16 x infinity | 16x1 -> 32x1 | 16x1 -> 32x3 | 16x1 -> infinity x 1 |'); P_('|' + '---|' * 14)
for k, nm, nb_ in (('same3', 'orig / ref / ref2 (same recipe)', 3), ('six', 'orig / ref / ref2 / hz / th / hzt', 6)):
    for w in ('val', 'hold'):
        r = tl[k][w]; f = r['forecast']
        P_(f'| {nm} | {w} | {r["a"]:.5f} | {r["c_root"]:.5f} ± {r["se"][1]:.5f} | {r["c_branch"]:.5f} ± {r["se"][2]:.5f} | {r["max_resid"]:.5f} | {r["obs"]["16x1"]:.5f} | {r["obs"][f"16x{nb_}"]:.5f} | ' + ' | '.join(pm(f[x]) for x in ('16x1->16x2', '16x1->16x3', '16x1->16xinf', '16x1->32x1', '16x1->32x3', '16x1->infx1')) + ' |')
P_('')
# ---------- T6 forms
F = J('forms.json'); P_('### T6. Zero-training structures and scans on FINAL val (paired per feed; ± = SE over 14,423 feeds)\n')
P_('| form | kind | final val | FT branch | TO branch | vs FD16P20 | pmix public | pmix private | vs W8P20 | pmix public / private |'); P_('|' + '---|' * 10)
for k, r in F.items():
    if k.startswith('_'): continue
    a, b = r['vs']['FD16P20'], r['vs']['W8P20']
    P_(f'| {k} | {r["kind"]} | {r["val"]:.5f} | {f5(r.get("ft"))} | {f5(r.get("to"))} | {a["d"]:+.5f} ± {a["se"]:.5f} | {a["pub"]:+.5f} ± {a["pub_se"]:.5f} | {a["priv"]:+.5f} ± {a["priv_se"]:.5f} | {b["d"]:+.5f} ± {b["se"]:.5f} | {b["pub"]:+.5f} / {b["priv"]:+.5f} |')
P_('')
d4 = F['_draws4of12']; P_(f'FD16 + 4 of the 12 wide pop1 students, all 495 draws: mean {d4["mean"]:.5f}, sd {d4["sd"]:.5f}, min {d4["min"]:.5f}, max {d4["max"]:.5f}; the realised FD16P20 is above {d4["realised_rank"]} of 495.\n')
P_('Leave-one-arm-out (final val delta when the arm is removed): ' + ' ; '.join(f'**{t}** ' + ', '.join(f'{k} {v[0]:+.5f} ± {v[1]:.5f}' for k, v in F[f'_loao_{t}'].items()) for t in ('A', 'B')) + '\n')
# ---------- T7 pairs
PR = J('pairs.json'); P_('### T7. Fixed structures against B, A, FD16P20 and the expected FD16P20-type file\n'); P_('| structure | final val | vs B | vs A | vs FD16P20 | vs expected FD16P20-type | pmix public / private (vs expected) |'); P_('|' + '---|' * 7)
for k, r in PR.items():
    if k.startswith('_'): continue
    v = r['vs']; e = v['E[FD16P20-type]']; P_(f'| {k} | {r["val"]:.5f} | {v["B"]["d"]:+.5f} ± {v["B"]["se"]:.5f} | {v["A"]["d"]:+.5f} ± {v["A"]["se"]:.5f} | {v["FD16P20"]["d"]:+.5f} ± {v["FD16P20"]["se"]:.5f} | {e["d"]:+.5f} ± {e["se"]:.5f} | {e["pub"]:+.5f} ± {e["pub_se"]:.5f} / {e["priv"]:+.5f} ± {e["priv_se"]:.5f} |')
P_('')
# ---------- T8 nested
N = J('nested.json'); P_('### T8. Learned combiners, nested by user (5 outer folds = the fine-tune folds)\n'); P_('| form | columns | in-sample val | NESTED val | nested vs B | per-fold nested vs B | nested pmix public / private vs B | nested vs A | nested vs FD16P20 | nested vs W8P20 | in-sample vs B | FT share of weight |'); P_('|' + '---|' * 12)
for k, r in N.items():
    P_(f'| {k} | {r["ncols"]} | {r["insample"]:.5f} | {r["nested"]:.5f} | {r["vsB"]["d"]:+.5f} ± {r["vsB"]["se"]:.5f} | {" ".join(f"{x:+.4f}" for x in r["fold_d_vsB"])} | {r["vsB"]["pub"]:+.5f} / {r["vsB"]["priv"]:+.5f} | {r["vsA"]["d"]:+.5f} ± {r["vsA"]["se"]:.5f} | {r["vsFD16P20"]["d"]:+.5f} ± {r["vsFD16P20"]["se"]:.5f} | {r["vsW8P20"]["d"]:+.5f} ± {r["vsW8P20"]["se"]:.5f} | {r["insample_vsB"]["d"]:+.5f} | {r["branch_weight"]["ft"]:.2f} |')
P_('')
# ---------- T9 rank replication
RR = J('rank_rep.json'); P_('### T9. Replication read of rank-z pooling (rank minus mean-z pooling, same members)\n'); P_('| member set | mean-z final val | final delta | FT branch | TO branch | pmix public | pmix private | holdout (TO pool) |'); P_('|' + '---|' * 8)
for k, r in RR.items():
    if k.startswith('_'): continue
    P_(f'| {k} | {r["mean"]:.5f} | {r["d"]:+.5f} ± {r["se"]:.5f} | {r["ft"]:+.5f} ± {r["ft_se"]:.5f} | {r["to"]:+.5f} ± {r["to_se"]:.5f} | {r["pub"]:+.5f} ± {r["pub_se"]:.5f} | {r["priv"]:+.5f} ± {r["priv_se"]:.5f} | {pm(r["hold"]) if "hold" in r else ""} |')
P_(f'\nPooled over S1..S4: {RR["_pooled"]["mean"]:+.5f}, z {RR["_pooled"]["z"]:+.2f} -> {RR["_verdict"]}.\n')
# ---------- T10 draws
D = J('draws.json'); P_('### T10. Is a favourable member draw a property of the members? (split-half transfer over val users)\n'); P_('| draws | weights | sd of the file level over draws | split-half r | reliability 2r/(1+r) | realised FD16P20 draw excess | expected on new feeds |'); P_('|' + '---|' * 7)
for k, r in D.items():
    if not isinstance(r, dict): continue
    for w in ('val', 'public', 'private'):
        q = r[w]; P_(f'| {k} | {w} | {q["draw_sd"]:.5f} | {q["split_half_r"]:+.3f} | {q["reliability"]:+.3f} | {"" if "realised_excess" not in q else format(q["realised_excess"], "+.5f")} | {"" if "expected_transfer" not in q else format(q["expected_transfer"], "+.5f")} |')
P_(f'\nFD16 + all 12 pop minus the mean 4-of-12 file: {D["all12_minus_mean4"][0]:+.5f} ± {D["all12_minus_mean4"][1]:.5f}.\n')
# ---------- T11 FT preview
FP = J('ft_preview.json'); P_('### T11. Fine-tunes of existing held-out members: what the train-only branch shows (baseline H0 = W16 + P1d + P1w)\n'); P_('| arm added to the base pool | fine-tunes needed (GPU-h) | TO-preview (FT branch unchanged) | TO branch val | TO branch holdout | law value if seed-like |'); P_('|' + '---|' * 6)
for k, r in FP.items():
    if k.startswith('_'): continue
    P_(f'| {k} | {r["k"]} | {pm(r["preview"])} | {pm(r["to_branch"])} | {pm(r["hold"])} | {r["law_seedlike"]:+.5f} |')
for k in ('_control_HZL8', '_control_FDXHZ'):
    r = FP[k]; P_(f'\nControl {k[9:]} (fine-tunes exist): both branches {pm(r["full"])}; FT branch {pm(r["ft_branch"])}; TO branch {pm(r["to_branch"])}; the TO-preview alone would have read {pm(r["preview"])}.')
P_('')
# ---------- T12 views
V = J('views.json'); P_('### T12. Views: decorrelation beyond seed noise against the standard view (hard-label members)\n'); P_('| arm | member gap val \\| hold | corr to base (different seed) | excess decorrelation | matched mix vs base (val \\| hold) |'); P_('|' + '---|' * 5)
for k, r in sorted(V.items(), key=lambda t: -t[1]['excess_decorr']):
    P_(f'| {k} | {r["level_val"]:+.5f} \\| {r["level_hold"]:+.5f} | {r["cross"]:.4f} | {r["excess_decorr"]:+.4f} | {pm(r["mix_val"])} \\| {pm(r["mix_hold"])} |')
open(OUT / 'tables.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
