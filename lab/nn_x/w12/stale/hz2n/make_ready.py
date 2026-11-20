"""Write READY.txt from the json / log files of this directory (re-run to refresh after a later proof finishes).  python make_ready.py"""
import glob
import json
import os
import subprocess
import time

H = '<repo>/nn_x/w12/stale/hz2n'
R = '<repo>'
J = lambda n: json.load(open(f'{H}/{n}')) if os.path.exists(f'{H}/{n}') else None  # noqa: E731
bt, btP, btF, p2, p2P, p2n, gf, gn, gx, sq = (J(n) for n in ('build_train.json', 'build_trainP.json', 'build_trainF.json', 'p2_val_parity.json', 'p2_val_parityP.json',
                                                           'p2n_val_parity.json', 'gate_full.json', 'gate_new.json', 'gate_g4_extra.json', 'build_seq0.json'))
p4a = {}
for fn in sorted(glob.glob(f'{H}/p4a_full*.json')):
    p4a.update(json.load(open(fn)))
md5 = dict(reversed(l.split()) for l in open(f'{H}/MD5SUMS.txt').read().strip().split('\n'))
code = subprocess.run(['md5sum', 'cd_arms6c.py'], cwd=H, capture_output=True, text=True).stdout.split()[0]
now = time.strftime('%FT%TZ', time.gmtime())
KA = f'{R}/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy'
KB = f'{R}/nn_x/w12/stale/hz2/feats_L/train/hz2_keep_rows.npy'
L = []
A = L.append
A(f'READY.txt of arm hz2n (builder), written {now}. Builder start 2026-11-15T07:12:44Z. CPU only; no GPU process started; no val / test / holdout label opened.')
A('Marks: [M] measured here, [D] read in a project file, [I] inferred. Declarations with their time stamps: DECLARATION.txt (+ ADDENDUM 1-3), PREREG_hz2n_arm.txt.')
A('')
A('0. WHAT TO DO (one screen)')
A(f' a. copy the module next to the trainer (I may not write there):   cp {H}/cd_arms6c.py {R}/nn_x/w5/cd_arms6c.py   (md5 {code})')
A(' b. ONE-LINE import change, nn_x/w5/train.py line 56 (the same line is nn_x/w12/kd/fd/train_full2.py line 69):')
A('      - from cd_arms6b import save_holdout_hz, setup_hzcd  # ...')
A('      + from cd_arms6c import save_holdout_hz, setup_hzcd  # wave-5 hzcd; 6c = cd_arms6b + optional encoder-score columns pre2_hz_* in HZCD_DIR (hz2n); identical to 6b when HZCD_DIR holds none')
A('    (nn_x/abl/train.py imports cd_arms6, not 6b: untouched. AST of train.py with the new line: 35 top-level statements, exactly 1 differs = this import [M].)')
A(' c. per arm export HZCD_DIR (+ HZ2_KEEP), as lane.sh lines 233-236 do for hz2 / hz2L; run with --hzcd exactly like hz / hz2. Nothing else changes.')
A(' d. check in the run log: the setup_hzcd line must contain "[6c: <n> swapped columns [...]; HZ2_KEEP rows on stored values: <k>; ENCODER-SCORE VIEW ON: <m> of them are')
A('    encoder-score columns [...] <- [...]". Expected <n> / <m> / <k> per directory in section 1. No "ENCODER-SCORE VIEW ON" in the line = the run trained on a plain 6b view.')
A('')
A('1. DIRECTORIES (HZCD_DIR -> HZ2_KEEP). Every directory holds symlinks to all files of its base view that cd_arms6b reads, plus the new files.')
A(' THE THREE VIEWS AS ASKED (fe head only = the stale-val witness construction build_pre2both, bit for bit):')
A(f'  (a) {H}/feats_n     HZ2_KEEP={KA}     log: 10 swapped / 2 encoder / keep 46918534   hz tensor 4.60 GB (hz2: 3.68)')
A(f'  (b) {H}/feats_Ln    HZ2_KEEP={KB}   log: 10 / 2 / 46918534   4.60 GB')
A(f'  (c) {H}/feats_hn    HZ2_KEEP unset                                                                     log: 7 / 2 / 0          3.22 GB (hz: 2.30)')
A(' DECLARED EXTENSION F (ADDENDUM 2; as-asked fe-head files + the NEW head rebuilt the same way = witness construction build_pre2full, bit for bit):')
A(f'  (aF) {H}/feats_nF   HZ2_KEEP as (a)   log: 12 / 4 / 46918534   5.52 GB')
A(f'  (bF) {H}/feats_LnF  HZ2_KEEP as (b)   log: 12 / 4 / 46918534   5.52 GB')
A(f'  (cF) {H}/feats_hnF  HZ2_KEEP unset    log: 9 / 4 / 0           4.14 GB')
A(f'  (cFS) {H}/feats_hnFS HZ2_KEEP unset   log: 12 / 4 / 0          5.52 GB   (ADDENDUM 3: cF + seq_cv_hz_* = 0 on view-cold rows, as on every cold val / test row)')
A(' DECLARED SENSITIVITY P (ADDENDUM 1; fe head with the no-ID score on the fe calibration; NOT recommended for GPU, see 5):  feats_nP, feats_LnP, feats_hnP (same keep, 10 / 2, 10 / 2, 7 / 2).')
A(' GPU memory: resident = the hz tensor above (0.46 GB per swapped column); setup peak = 2x that for a moment (torch.stack in load_hz, unchanged from 6b).')
A(' The two keep masks are byte-identical files [M]. Keep rows carry no encoder score at all (0 of 46,918,534 keep rows have a finite pre2_fe [M]: feeds of Sep 6-8 lie before')
A(' the first encoder window), so the keep mask does not interact with the new columns.')
A('')
A('2. DEFINITION (all 246,745,836 train rows)')
A(' view it_hist_all = the RAW counter file <base>/train/it_hist_hz_all.npy (float32; a: hz2/feats, b: hz2/feats_L, c: data/feats_new); stored counter = nn/feat_cache/train/it_hist_all_luo.npy.')
A(' newly cold row nc = (view == 0) & (stored > 0) & ~keep. Raw vs model scale: 0 rows with view > 0 whose fp16(log1p) is 0, 0 rows with view > 0 and stored == 0, no NaN [M].')
A(' on nc rows: pre2_fe := data/feats/train/pre2_fe_noid.npy; pre2_fe_s := float32((pre2_fe_noid + 20) / 6); pre2_fe_z := mean over the 2 configs of the within-feed z of the mixed')
A(' per-config scores, recomputed over the WHOLE feed (as the witness does); has:pre2_fe follows the mixed score. Every non-nc row keeps the stored bits of pre2_fe / pre2_fe_s / has;')
A(' pre2_fe_z keeps the stored bits in every feed without an nc row, and changes on all rows of a feed with one (re-standardisation; mean |dz| on the other rows 0.059 a, 0.062 b, 0.015 c [M]).')
A(' has:pre2_fe: 0 of 246,745,836 rows change in every directory [M] -> nothing to swap; the file sits in <dir>/aux/pre2_hz_fe_has.npy (cd_arms6c would swap it if it were in train/).')
A(' Files a model under --xt cvnd_pre2s reads from the new set: pre2_hz_fe_z.npy, pre2_hz_fe_s.npy (+ pre2_hz_new_z.npy, pre2_hz_new_s.npy in F). pre2_hz_fe.npy / pre2_hz_new.npy are the')
A(' raw mixed scores (not inputs of that preset). hz2n_newly_cold.npy = the nc mask.')
A('')
A('3. PROOFS (one line each)')
A(f" P0 [M] PASS: the per-config arithmetic without override reproduces stored pre2_fe, pre2_fe_noid, pre2_fe_z BITWISE on all {bt['rows_in_windows']:,} window rows (0 differing rows per window; "
  f"stored pre2_fe_s == float32((pre2_fe + 20) / 6) bitwise); outside the windows no row has a score ({bt['rows_outside_windows_with_finite_fe_or_noid']} finite). New head (P0n): same, 0 differing rows.")
A(' P1 [M] PASS: cd_arms6c vs nn_x/w5/cd_arms6b.py with HZCD_DIR unset / hz2/feats (+keep) / hz2/feats_L (+keep) / hz2/feats (no keep): AST: changed defs exactly {luo_cols, setup_hzcd}, no new def,')
A('    only constants ENC_TO_HZ / ENC_GROUPS added; identical column lists (W8 preset and pop-light list), bit-identical load_hz tensors, 40 batches x 22 keys bit-identical on 2,000 random')
A('    non-holdout train feeds (158,751 cells), identical hz_stats, log line identical apart from the tag "6c". Diff: cd_arms6c_vs_cd_arms6b.diff (made by make_6c.py).')
A(f" P2 [M] PASS: my row-wise function on the VAL rows with the witness counters reproduces solvers/out/stale/cf/pre2both_pre2_fe / pre2_fe_z / pre2_fe_s.npy with max|d| = "
  f"{p2['P2_max_abs_pre2_fe']:.1e} / {p2['P2_max_abs_pre2_fe_z']:.1e} / {p2['P2_max_abs_pre2_fe_s']:.1e} (0 of 1,389,971 rows differ bitwise in each; nc == newly_cold.npy; has identical).")
A(f"    New head (P2n) vs pre2full_pre2_new / _z / _s.npy: max|d| = {p2n['P2n_max_abs_pre2_new']:.1e} / {p2n['P2n_max_abs_pre2_new_z']:.1e} / {p2n['P2n_max_abs_pre2_new_s']:.1e} (0 rows differ bitwise). "
  f"P family vs the witness: {p2P['P2_max_abs_pre2_fe']:.1e} / {p2P['P2_max_abs_pre2_fe_z']:.1e} / {p2P['P2_max_abs_pre2_fe_s']:.1e} (NOT witness-exact, as declared).")
v = bt['views']
A(' P3 [M] PASS (all 246,745,836 rows, each directory): rows outside nc with changed bits in pre2_hz_fe / pre2_hz_fe_s / has: 0 / 0 / 0; pre2_hz_fe_z rows changed in feeds without an nc row: 0;')
A('    keep rows: never nc and without any score. New-head files (P3n): 0 / 0. Rows changed (fe_s; fe_z): ' + '; '.join(
    f"{k} {x['rows_changed']['fs']:,}; {x['rows_changed']['fz']:,}" for k, x in v.items()) + '.')
ok = sorted(k for k, x in p4a.items() if x['pass']); bad = sorted(k for k, x in p4a.items() if not x['pass'])
A(f" P4a [M] {'PASS' if ok and not bad else 'INCOMPLETE/FAIL'}: the REAL setup_hzcd of cd_arms6c (unwrapped load_hz, the real HZ2_KEEP file) over the FULL row space on a CPU stand-in split: every cell of every swapped "
  f"column == keep ? stored : fp16(attach_xt transform(view file)) (0 differing cells of 246,745,836 per column); encoder columns: 0 rows changed outside nc, 0 nc rows that are not the no-ID value; "
  f"gather of 500 random feeds == those cells. Directories passed: {ok}; failed: {bad or 'none'}; pending: {sorted(set(['feats_n', 'feats_Ln', 'feats_hn', 'feats_nF', 'feats_LnF', 'feats_hnF', 'feats_hnFS', 'feats_nP', 'feats_LnP', 'feats_hnP']) - set(p4a)) or 'none'}.")
A(' P4b [M] PASS (10 directories, MiniSplit of 2,000 real train feeds, setup_hzcd with the sample keep file): log clause present; view tensor == keep ? stored : fp16(transform(file)); popularity columns')
A('    bit-identical to the 6b view of the base directory; pre2_fe_s == no-ID value on nc cells and stored bits elsewhere; every other key / column == plain gather; pop-light xt list swaps')
A('    [pre2_fe_z, pre2_fe_s(, xp_imp_c50)]; a partial head is refused; preset cvnd_pre2 (pre2_id_contrib input) is refused; has: hook works when the file is in train/.')
A(' P4c [M] PASS: real Ranker (w5_e10_kd05w192_s951@77500), 30 AdamW steps on batches gathered through HzTrainView(cd_arms6c) for feats_n / feats_nP / feats_nF: finite losses, weights move.')
A('    Label-free sensitivity before any step: mean |d within-feed z of the logit| vs the 6b hz2 view: 0.012 all cells, 0.022 nc cells (the hz2 view itself vs stored inputs: 0.160).')
A(' P4 LIMIT, stated plainly: nn_x/w5/train.py itself was NOT run on CPU. The fulldata harness cpu_run.py loads the whole train split (about 45 GB resident, 60+ GB peak by my')
A('    count [I]), above my 40 GB limit. Flag-off identity therefore rests on: (1) the one-line import is the only AST difference of train.py; (2) cd_arms6c at import only reads the two')
A('    environment variables and defines names (same top-level statements as 6b + 2 constants); (3) with --hzcd absent no function of the module is called. The first GPU run is the')
A('    end-to-end test of train.py + cd_arms6c: read its setup_hzcd log line (0.d) before trusting the lane.')
A('')
A('4. GATE (label-free; all 246,745,836 train rows; "pool" = trPool private-like feeds of days >= Sep 13, the population of the review\'s +0.21 and the one exchangeable with test feeds)')
for l in open(f'{H}/gate_table.txt').read().rstrip().split('\n'):
    A(' ' + l)
A('')
A(' GATE VERDICTS against the declared rules:')
A(' G1 (target 0): PASS for (a) (b) (c): 0 view-cold rows keep an fe-head ID contribution (before: 23,725,776 / 25,607,841 / 6,199,225 rows = 23.6% / 24.5% / 11.0% of view-cold rows;')
A('    30.2% / 31.7% / 13.5% on days >= Sep 13). val / public / private: 0 as built. P family: 13,001 / 13,502 / 3,607 rows left by a 1e-3 tolerance reading (0.013% of cold rows;')
A('    8,791 / 9,082 / 2,107 of them are rows where the assembler\'s guard "raw score > -2" differs between the score and its no-ID twin [M]; the rest not examined).')
A(' G2(i) rows made cold AND ID-carrying, pre2_fe_s vs private cold rows: on the pool population +0.207 -> -0.003 (a), +0.203 -> +0.010 (b), +0.117 -> -0.043 (c): PASS (the review\'s +0.210 is')
A('    reproduced: +0.2101 against private feeds with M >= 50). On ALL train feeds (the population my declaration named first) +0.641 -> +0.366 (a), +0.633 -> +0.369 (b), +0.571 -> +0.367 (c):')
A('    FAIL BY THE LETTER (needed < half of before). Diagnosis [M]: rows the view never touches (D, as-built cold) read +0.265 on the same population and -0.038 on the pool, i.e. production-')
A('    ordered train feeds carry a level offset of about +0.3 SD against test feeds that has nothing to do with the view. My criterion ignored it. After the change, B sits at the level')
A('    of the other cold rows of its own population (C +0.370, A +0.334). I report the FAIL as declared and do not count it as a defect of the view; the lead decides.')
A(' G2(ii) all view-cold rows: PASS on all feeds for every directory (|after| < |before|); on the pool PASS for a, b and FAIL for c (-0.035 -> -0.052 vs private; +0.000 -> -0.017 vs public).')
A(' G3: rows whose pre2_fe_s changes 20.8% (a) / 22.3% (b) / 6.9% (c) of all train rows (25.8% / 29.4% / 8.6% on days >= Sep 13); pre2_fe_z changes on 72% / 72% / 66% (99.9% / 99.9% / 90%');
A('    on days >= Sep 13) because the feed is re-standardised; feeds with a changed row 71.9% / 72.0% / 60.7% (98.7% / 99.1% / 81.9%). Sep 6-8 feeds: 0 rows change (no encoder score there).')
A(' G4, NOT repaired by the three views as asked (say-so):')
A('  (1) NEW encoder head. pre2_new_z / pre2_new_s carry the item-ID contribution on exactly the same rows: P(new-head ID | view cold) 0.236 (a) / 0.245 (b) / 0.110 (c), 0.0000 on cold val /')
A('      public / private rows. The witness both_pre2both does not replace this head either. After the as-asked change these rows read "fe head without ID + new head with ID", a pattern')
A('      that exists on no test row and on no as-built train row, only in the stale-val witness inputs. Repaired in F (after: 18 / 20 / 5 coincidental rows; pre2_new_s on B rows, pool:')
A('      +0.150 -> -0.017 (a), +0.152 -> -0.003 (b), +0.065 -> -0.058 (c)).')
A('  (2) View c only: it_hist_all == 0 with seq_cv_pop > 0 on 9.7% of view-cold rows (22.7% of the rows it makes cold; seq_cv_invd20 / norm 1.0%), impossible on val / test (0.0000 exactly).')
A('      This is in arm hz as trained. Repaired in feats_hnFS only.')
A('  (3) Train calibration: on rows WITHOUT any ID contribution the stored pre2_fe_noid sits below pre2_fe by 0.257 / 0.184 / 0.008 (windows Sep 9-11 / Sep 12-15 / Sep 17-20; pre2_fe scale,')
A('      SD 4.3; val +0.003, test 0 exactly). So "as asked" also lowers pre2_fe_s of nc rows that never carried an ID, by 0.04 / 0.03 / 0.001 (0.06 / 0.04 / 0.002 private-row SD).')
A('      The review\'s "within 0.028" holds for the last window only. The P family removes this shift; level evidence does not separate the two (difference <= 0.03 SD, sign depends on the')
A('      population), so P is a sensitivity variant, not a recommendation.')
A('  (4) Unchanged by design and still unlike test: warm rows are 1-2 d stale instead of 7-10 d, non-integer counts (review MAJOR 1); Sep 6-8 keep hybrid (review MEDIUM 3); pre2_hmax32')
A('      (ID-dependent on 98% of cold rows on val / test and on train alike, so no tell by this reading; a candidate-ID part cannot be separated here); xp_imp_c50 > 0 on 40% of view-cold')
A('      rows under view c against 18-31% on targets (a level difference, not an impossible combination).')
A('')
A('5. REASONS NOT TO TRAIN / RECOMMENDATION (the choice is the lead\'s)')
A(' (a) feats_n: none from the proofs; G2(i) fired by the letter on the all-feeds reading (see above). Limit: half of the encoder tell (new head) stays. Trains on the stale-val witness pattern.')
A(' (b) feats_Ln: no base arm exists (no hz2L run on record [D hz2_review.md]) and the base view fails the hz2 study\'s own gate by G1 [D]; an hz2Ln arm can only be read against hz2, which')
A('     mixes "lag-first" with the encoder change. Do not train (b) unless an hz2L base arm is run next to it.')
A(' (c) feats_hn: none from the proofs; keeps tells (1) and (2) of G4. If one view-c arm is run, feats_hnFS is the one in which no cold-row combination is train-only by my checks.')
A(' F directories: the only ones in which no encoder-score input carries an item-ID contribution on a view-cold row. Read their stale-val with BOTH scenarios: both_pre2both (stale_pred.py)')
A('     feeds "fe without ID + new with ID" on newly cold val rows, which F models never saw; the like-for-like scenario is both_pre2full: stale_pred_full.py in this directory (a 10-line')
A('     wrapper that imports nn_x/w5/stale_pred.py unchanged and adds the three pre2full files; GPU script, NOT run by me).')
A(' My order if GPUs are short [I]: feats_hnFS (the chosen recipe X = hz with every cold-row tell removed) and feats_nF first; feats_n and feats_hn (as asked) next; (b) and P last / not at all.')
A(' Lagged-holdout read through a new view for ANY member (ref / hz / hz2 / hz2n): hold_infer_n.py in this directory (wrapper around the validated hz2/hold_infer.py; its attach()')
A('     self-test PASSED on the 30,000 holdout feeds for feats_n / feats_nF / feats_hnFS; scoring itself not run by me). Each hz2n run also writes its own nn_<v>_holdout_hz.npy through its view.')
A(' Pre-registration of what I expect: PREREG_hz2n_arm.txt (short: no arm of this family separates from its base arm by more than 1.5 SE on val or stale-val).')
A('')
A('5b. EXPECTED VALUES IN THE setup_hzcd LOG LINE of a GPU run [M, P4a: the same function over all rows on CPU]: "rows changed" per column (fp16 scale), W8 preset')
N = 246745836
for dn in ('feats_n', 'feats_Ln', 'feats_hn', 'feats_nF', 'feats_LnF', 'feats_hnF', 'feats_hnFS'):
    if dn in p4a:
        r = p4a[dn]
        A(f"  {dn:10s} {len(r['columns']):2d} columns, hz tensor {r['hz_GB']:.2f} GB: " + ', '.join(f"{k} {v / N:.4f}" for k, v in r['rows_view_ne_stored'].items()))
A('  (zero rate of the pre2 columns is unchanged, 0.279 -> 0.279: rows without an encoder score stay 0.)')
A('')
A('5c. STATE SEEN AT 07:43Z (read-only look, not my action): nn_x/w5/cd_arms6c.py exists with md5 == mine; nn_x/w5/train.py.6c differs from train.py by the import line only; lane.sh has the arms')
A('  hzn / hznF / hznFS / hz2n / hz2nF / hz2Ln / hz2LnF with the directories and keep files of section 1 (checked by reading lines 157 and 238-244).')
A('  Verified for the witness correction of 07:42Z [M, head_mismatch.json]: on val / public / private the set of rows with an fe-head ID contribution and the set with a new-head ID')
A('  contribution are IDENTICAL (0 rows with one and not the other; 628,350 / 373,332 / 810,938 rows), and cold rows have neither. The witness both_pre2both feeds 43,419 val rows')
A('  (3.1% of val rows, 6.4% of its cold rows) with "fe head replaced, new head still with ID"; both_pre2full replaces the new head on exactly those 43,419 rows.')
A('')
A('6. MD5')
A(f' {code}  cd_arms6c.py')
for k in sorted(md5):
    if k.startswith('feats_'):
        A(f' {md5[k]}  {k}')
A(' (symlinked base-view files keep the checksums of nn_x/w12/stale/hz2/MD5SUMS_feats.txt and data/feats_new; full list incl. scripts: MD5SUMS.txt)')
A('')
A('7. FILES: DECLARATION.txt, PREREG_hz2n_arm.txt, PROGRESS.md, cd_arms6c.py, make_6c.py, cd_arms6c_vs_cd_arms6b.diff, train_py_import_6c.txt, enc_common.py, build_enc.py, build_new.py,')
A('   build_seq0.py, gate.py, gate_new.py, gate_table.py, test_cd_arms6c.py, p4a_full.py, stale_pred_full.py, hold_infer_n.py, make_ready.py, *.json, gate_table.txt, logs/. Report: reports/STALE_1115/hz2n.md.')
open(f'{H}/READY.txt', 'w').write('\n'.join(L) + '\n')
print(f'READY.txt written {now}: {len(L)} lines; p4a passed {ok}, failed {bad}')
