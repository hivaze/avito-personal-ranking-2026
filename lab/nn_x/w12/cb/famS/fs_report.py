"""Assembles reports/CATBOOST_1115/famS.md from parts/ (third run of the famS task). Text only, no model, no label.
Order: 10_head (verdict, arm table) | 20_body section 2 | 25_yeti (2.4) | 30_seeds (3) | 20_body sections 4-7 (with the rows of the final reads added)
       | 38_limits (8) | 40_cmd (9, 10) | section 11 = verbatim nats.py lines from logs/final3_reads.log.
Every edit of a drafted part is an exact-match replacement that must hit exactly once.
usage: python fs_report.py"""
from pathlib import Path
H = Path("<repo>/nn_x/w12/cb/famS")
OUT = Path("<repo>/reports/CATBOOST_1115/famS.md")
rd = lambda n: (H / "parts" / n).read_text().rstrip("\n") + "\n"


def sub(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new)


body = rd("20_body.md")
i = body.index("## 4. Student against hard-label tree")
sec2, rest = body[:i], body[i:]

# ---- section 2: point to the new sub-section
sec2 = sub(sec2, "the plateau read for the best arm is the \"long\" run of section 3.",
           "the plateau read for the best arm is the \"long\" run of section 3.2. The YetiRank arm (117 trees) is in section 2.4.")

sec2 = sub(sec2, "is the worst S arm: 0.38021 / 0.30501, at the level of the hard-label tree.",
           "has the lowest val level of the family: 0.38021 / 0.30501, at the level of the hard-label tree.")
sec2 = sub(sec2, "(S4b) is the best S arm per tree and per CPU minute.",
           "(S4b) is the best S arm per CPU minute and, together with YetiRank on the teacher logit (section 2.4), per tree.")

# ---- section 4: rows of the final reads (3-seed bundle, plateau run, YetiRank arm)
row_s4b_h = "| | holdout | +0.00043 ± 0.00015 | +0.00051 ± 0.00017 | +0.00052 ± 0.00017 | +0.00049 ± 0.00016 | +0.00001 ± 0.00002 | +0.00009 ± 0.00007 | 0.964 | 0.867 |\n"
new4 = (
    "| **3-seed bundle of S1f** | val | +0.00054 ± 0.00026 | +0.00117 ± 0.00038 | +0.00124 ± 0.00039 | +0.00089 ± 0.00033 | **+0.00007 ± 0.00010** | **+0.00070 ± 0.00030** | 0.949 | 0.805 |\n"
    "| | holdout | +0.00040 ± 0.00015 | +0.00051 ± 0.00017 | +0.00052 ± 0.00017 | +0.00048 ± 0.00016 | **+0.00000 ± 0.00001** | **+0.00012 ± 0.00008** | 0.950 | 0.809 |\n"
    "| S1f plateau run (seed 1, 1,608 trees) | val | +0.00049 ± 0.00025 | +0.00117 ± 0.00038 | +0.00122 ± 0.00039 | +0.00086 ± 0.00033 | +0.00005 ± 0.00008 | +0.00073 ± 0.00030 | 0.940 | 0.769 |\n"
    "| | holdout | +0.00043 ± 0.00015 | +0.00051 ± 0.00017 | +0.00053 ± 0.00017 | +0.00050 ± 0.00016 | +0.00002 ± 0.00003 | +0.00011 ± 0.00008 | 0.941 | 0.773 |\n"
    "| S4y (YetiRank on the teacher logit) | val | +0.00058 ± 0.00027 | +0.00117 ± 0.00038 | +0.00131 ± 0.00040 | +0.00086 ± 0.00033 | +0.00014 ± 0.00013 | +0.00073 ± 0.00030 | 0.960 | 0.863 |\n"
    "| | holdout | +0.00038 ± 0.00014 | +0.00051 ± 0.00017 | +0.00051 ± 0.00017 | +0.00046 ± 0.00016 | +0.00000 ± 0.00000 | +0.00013 ± 0.00008 | 0.961 | 0.865 |\n")
rest = sub(rest, row_s4b_h, row_s4b_h + new4)
rest = sub(rest, "member rule -0.00048 ± 0.00039 / +0.00005 ± 0.00021, pooled z -0.69, no pass.",
           "member rule -0.00048 ± 0.00039 / +0.00005 ± 0.00021, pooled z -0.69, no pass. The same read with the 3-seed bundle as the student (addendum 2 (e)): level 0.38493 / 0.30812, "
           "NATS +0.00093 ± 0.00034 / +0.00050 ± 0.00016, not promising; member rule -0.00014 ± 0.00038 / +0.00010 ± 0.00020, pooled z +0.08, no pass. "
           "(The table's bundle column re-standardises the z-mean per feed, the tool does not: 0.00089 / 0.00048 against 0.00093 / 0.00050; the saved bundle file reads +0.00054 as W16 + S where the tool on the three seeds reads +0.00056.)\n\n"
           "**The lead's question in one line** (student = 3-seed bundle of the best arm, hard-label tree = `ts_lr020_s0`; NATS val / holdout): student alone +0.00056 / +0.00040; hard-label tree alone +0.00117 / +0.00051; "
           "BOTH as one bundle +0.00093 / +0.00050; BOTH as two covariates +0.00124 / +0.00052. Adding the student to the hard-label tree gains +0.00007 ± 0.00010 / +0.00000 ± 0.00001; adding the hard-label tree to the student gains +0.00070 ± 0.00030 / +0.00012 ± 0.00008. "
           "Same information, and the student holds the smaller share of it.")
rest = sub(rest, "(z between -2.8 and +1.1)", "(z between -2.8 and +1.2)")

# ---- section 5: M-bin rows of the final reads
row5 = "| S4b | 0.01993 ± 0.00247 | +0.00143 ± 0.00070 | -0.00084 ± 0.00071 | 0.30584 | 0.01476 ± 0.00151 | +0.00032 ± 0.00025 | +0.00006 ± 0.00042 | 0.01599 ± 0.00081 |\n"
new5 = (
    "| S1f seed 1 (886 trees) | 0.02105 ± 0.00248 | +0.00086 ± 0.00054 | -0.00047 ± 0.00070 | 0.30441 | 0.01619 ± 0.00153 | +0.00023 ± 0.00021 | -0.00002 ± 0.00042 | 0.01763 ± 0.00081 |\n"
    "| S1f seed 2 | 0.01997 ± 0.00252 | +0.00080 ± 0.00052 | +0.00024 ± 0.00067 | 0.30466 | 0.01594 ± 0.00154 | +0.00036 ± 0.00026 | -0.00032 ± 0.00042 | 0.01694 ± 0.00082 |\n"
    "| **3-seed bundle of S1f** | 0.01942 ± 0.00247 | +0.00087 ± 0.00054 | -0.00052 ± 0.00069 | 0.30557 | 0.01503 ± 0.00150 | +0.00034 ± 0.00025 | -0.00009 ± 0.00041 | 0.01663 ± 0.00081 |\n"
    "| S1f plateau run (1,608 trees) | 0.01953 ± 0.00247 | +0.00087 ± 0.00054 | -0.00058 ± 0.00068 | 0.30547 | 0.01513 ± 0.00150 | +0.00027 ± 0.00023 | -0.00007 ± 0.00041 | 0.01667 ± 0.00080 |\n"
    "| S4y | 0.02197 ± 0.00254 | +0.00113 ± 0.00061 | -0.00114 ± 0.00084 | 0.30280 | 0.01780 ± 0.00158 | +0.00027 ± 0.00023 | -0.00013 ± 0.00051 | 0.01999 ± 0.00084 |\n")
rest = sub(rest, row5, row5 + new5)

# ---- section 6: measured cost of the YetiRank arm, caps of the late arms, tools of the third run
rest = sub(rest, "YetiRank on the real-valued teacher logit about 24 s [I from the dry run: 6.5x RMSE].",
           "YetiRank on the real-valued teacher logit 17.1 s (the dry run had suggested 24 s).")
rest = sub(rest, "S1 135 min, S1f 55, S3 75, S4 35, S4b 55, S2 45.",
           "S1 135 min, S1f 55, S3 75, S4 35, S4b 55, S2 45, S4y 33; seed 2 of S1f: 886 trees (60 min, guard 70); seed 1: 1,772 trees asked, 1,761 built at the 112-minute guard.")
rest = sub(rest, "- rho: the nats tool prints",
           "- Third run: `fs_final.sh` and `fs_final3.sh` (all final reads; `nats.py` itself on every arm and on the bundles), `fs_natsdiff.py` (paired NATS(X) - NATS(hard-label reference); it reproduces the second run's values to every printed digit), "
           "`fs_trunc.py` (seed 1 at the tree cap T0: prediction at its best selection iteration among the first 886 trees of the long model; boosting is sequential, so this is the 886-tree run of that seed).\n"
           "- rho: the nats tool prints")

# ---- section 7: leak line of the YetiRank arm
rest = sub(rest, "- Seeds of S1f and their bundle: same as S1f;",
           "- S4y (YetiRank on the teacher logit): teacher target only, 177 inputs; as S1.\n"
           "- S1f plateau run (seed 1, up to 1,761 trees): same as S1f; its tree count is chosen on the train-side selection split like every other arm.\n"
           "- Seeds of S1f and their bundle: same as S1f;")

cmd = rd("40_cmd.md")
cmd = sub(cmd, "$F/fs_final.sh                                                        # all final reads in one go (log: logs/final_reads.log)",
          "$F/fs_final.sh                                                        # all final reads in one go (log: logs/final_reads.log)\n"
          "$F/fs_final3.sh > $F/logs/final3_reads.log 2>&1                       # third run: YetiRank arm reads + nats.py itself on every arm and bundle (section 11)\n"
          ".venv/bin/python $F/fs_natsdiff.py ts_lr020_s0 <tag | tagA+tagB+...> [...]   # paired NATS(X) - NATS(hard-label reference) (log: logs/natsdiff3.log)\n"
          ".venv/bin/python -u $F/fs_staged.py s1f_rmse_d8_lr030_s1long none 300,500,700,886,1000,1200,1400,1608,1761   # plateau table of section 3.2\n"
          ".venv/bin/python $F/fs_report.py                                      # assembles this report from parts/")
cmd = sub(cmd, "`fs_table.py`, `fs_lines.py`,", "`fs_table.py`, `fs_lines.py`, `fs_natsdiff.py`, `fs_final3.sh`, `fs_report.py`,")
cmd = sub(cmd, "`reads/dist_pca.json`, `reads/cmp_lgb_students.json`;",
          "`reads/dist_pca.json`, `reads/cmp_lgb_students.json`, `reads/cmp_lgb_<tag>.json`, `reads/natsdiff_vs_hard.json` and `reads/natsdiff3_vs_ts_lr020_s0.json`, `reads/bundle_weights3.json` (bundles at w 0.05 / 0.1 / 0.2);")

log = (H / "logs/final3_reads.log").read_text()
lines = [l for l in log[log.index("== nats.py itself"):].splitlines() if ": level " in l]
assert len(lines) == 14, len(lines)
sec11 = ("## 11. Verbatim lines of the shared tool `nn_x/w12/cb/nats.py` (third run, 06:27Z; `logs/final3_reads.log`)\n\n"
         "A bundle is the tool's own bundle form (several val / holdout pairs). `ts_lr020_s0` is the plan's reference line and reproduces it.\n\n```\n" + "\n".join(lines) + "\n```\n")

parts = [rd("10_head.md"), sec2, rd("25_yeti.md"), rd("30_seeds.md"), rest, rd("38_limits.md"), cmd, sec11]
txt = "\n".join(p.rstrip("\n") + "\n" for p in parts)
assert "PLACEHOLDER" not in txt and "INTERIM" not in txt
OUT.write_text(txt)
print(f"wrote {OUT} ({len(txt)} chars, {txt.count(chr(10))} lines)")
