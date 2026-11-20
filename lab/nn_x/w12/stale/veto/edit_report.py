"""veto (resume run): apply the resume-run additions to my own report reports/STALE_1115/veto.md. Every anchor must occur exactly once;
otherwise nothing is written. A copy of the report as it stood before the resume edits is out/veto_report_before_resume.md."""
from pathlib import Path

ROOT = Path("<repo>"); HERE = ROOT / "nn_x/w12/stale/veto"; REP = ROOT / "reports/STALE_1115/veto.md"
s = REP.read_text()
assert "[R]" not in s.split("## 1. The list")[1], "resume edits already applied"


def rep(old, new):
    global s
    assert s.count(old) == 1, f"anchor occurs {s.count(old)} times: {old[:80]!r}"
    s = s.replace(old, new)


# ---- header + section 0
rep("Rules of every re-read: `nn_x/w12/stale/veto/PREDECLARED.txt` (02:23:09Z, before the first re-read).",
    "Rules of every re-read: `nn_x/w12/stale/veto/PREDECLARED.txt` (02:23:09Z, before the first re-read; addenda R1 03:39:52Z and R2 for the resume-run reads, "
    "each written before its read). [R] marks what the resume run (03:31Z onward) added or corrected.")
a, b = s.index("## 0. Verdict, and the plain list of what to re-open"), s.index("## 1. The list")
s = s[:a] + (HERE / "out/sec0_new.md").read_text() + s[b:]

# ---- section 1
rep("Screened: 273 idea rows of `reports/COVERAGE_MAP.md` (file lines 17-331; row numbers below are file line numbers)",
    "Screened: 277 idea rows of `reports/COVERAGE_MAP.md` (274 table lines of the 336-line file of 03:40Z, three of which hold two rows; the first run counted 273 "
    "in lines 17-331; row numbers below are file line numbers)")
rep("`reports/nn_report.md` R2, `nn_x/w5/logs/E_kd_predeclared.txt`.",
    "`reports/nn_report.md` R2, `nn_x/w5/logs/E_kd_predeclared.txt`. [R] Second screen of the resume run: two pattern passes over every row that carries a holdout, veto, "
    "stale, pmix or cold/warm word (89 of the 114 such rows read again), `nn_x/abl/ledger/ledger.md` section 1 (components A-I, for the reverse check), `docs/05_METHODOLOGY.md` "
    "(rule table), `reports/RECHECK_1114/pool_composition.md`, `reports/FEATURES_REVIEW.md`, the `E_kd_predeclared.txt` blocks up to 03:35Z. It added five rows "
    "(301, 261, 266, 268, U20) and found no further closure of kind (a), (b) or (c).")
rep("| veto stands (not a cold/warm trade); proposed under the two-witness rule as item 10 of section 5 |",
    "| veto stands (not a cold/warm trade). [R] CORRECTION: at @last the 7-pool reads val +0.00009 +- 0.00025, pmix-priv +0.00007 +- 0.00035, holdout -0.00011 +- 0.00012; "
    "the +0.00070 exists only at holdout-selected checkpoints (section 3.9). Item 10 of section 5 |")
rep("\n\nReverse cases (holdout positive, val negative, closed on val) were not counted:",
    "\n| 301 [R] | pool-composition screen of 11-14, judged holdout-first: \"0 of 52 families adds on the holdout at 2 SE\", \"C3 admixture hurts the holdout at every weight\", "
    "forward selection | a | W8 + pop1kd x4 member-equal: val +0.00020 +- 0.00046, holdout -0.00113 +- 0.00030; W8P = 0.7 z(W8) + 0.3 z(hard pop1 x8), final: val +0.00039 +- 0.00041, "
    "pmix-priv +0.00077 +- 0.00066, TO holdout -0.00117 +- 0.00029 (\"rejected by the holdout-first rule\"); selection fitted on val picks pop1 groups and reads -0.00095 on the holdout "
    "[D `RECHECK_1114/pool_composition.md`] | section 3.8: 85 stored families as a 0.2 admixture to W16; 5 flags (6.8 expected by chance), no two-witness pass | "
    "its popularity-light part is rows 302 / 330 (staleness family, re-opened); for every other family the screen stands |\n"
    "| 261 [R] | C04 co-visitation variants + NB log-lift (residual on G3) | b | val -0.00020 +- 0.00034, holdout -0.00030 (z -1.70), pmix +0.00033 +- 0.00053, "
    "coldpos +0.00595 / warmpos -0.00245 [D `ccb_1113/RESULT.md`] | - | BORDERLINE by mechanism (other co-visitation windows; measured as a cold/warm trade). FAIL: val < 0, pmix 0.6 SE; "
    "it failed its val bar (+0.0005) first |\n"
    "| 266, 268 [R] | C09 RP3beta walk; C11 typed implicit rating (residuals on G3) | a by the letter | val +0.00012 +- 0.00036 / +0.00006 +- 0.00033, holdout -0.00013 / -0.00018, "
    "pmix -0.00056 +- 0.00055 / -0.00071 +- 0.00048 (veto), C09 coldpos -0.00400 / warmpos +0.00211 | - | stand: both failed the val bar first and the private-mix lens is against both |\n"
    "| `E_kd` 03:13Z [R] | U20: fold-0 feeds kept with teacher-only targets (B) against the fold dropped (A), 2 seeds | c | B - A val +0.00158 +- 0.00071, holdout -0.00017 +- 0.00039; "
    "the KILL clause \"B - A < +0.0004 on the holdout\" was met | as a 0.2 admixture to W16 the u20b 2-pool is null (val +0.00005 +- 0.00029) | NOT closed: extended to 4 seeds on the "
    "owner's directive (GO iff val >= +2 SE and holdout >= -1 SE). Not staleness family: the holdout stays a guard, which is the right form for it |"
    "\n\nReverse cases (holdout positive, val negative, closed on val) were not counted:")

# ---- section 3.5: sources of the CPU block + the extra rows
rep("<!-- cpu-block-end -->",
    "<!-- cpu-block-end -->\n\n" + (HERE / "out/sec35_add.md").read_text().rstrip("\n"))
# ---- sections 3.8, 3.9 before section 4
rep("## 4. The reverse check: what was adopted with the holdout as a witness", (HERE / "out/sec38_39.md").read_text() + "## 4. The reverse check: what was adopted with the holdout as a witness")
# ---- section 4 rows + notes
rep("\n\nNotes.\n- Checkpoint selection. The 40 hard-label re-cool runs", "\n" + (HERE / "out/sec4_rows.md").read_text().rstrip("\n") + "\n\nNotes.\n- Checkpoint selection. The 40 hard-label re-cool runs")
rep("## 5. Ranked re-open list", (HERE / "out/sec4_notes.md").read_text() + "## 5. Ranked re-open list")
# ---- section 5: row 10 + section 5.1
rep("| 10 | **aux weight 1.0 under distillation** (not staleness family: two-witness rule, the holdout stays in) | +0.0002 (0 .. +0.0005) | 0.15 |",
    "| 10 | **aux weight 1.0 under distillation** (not staleness family: two-witness rule, the holdout stays in). [R] basis corrected, see 5.1 | +0.0001 (0 .. +0.0004), lowered [R] | 0.10 [R] |")
rep("## 6. Method, limits, and what I did not check", (HERE / "out/sec51.md").read_text() + "## 6. Method, limits, and what I did not check")
# ---- section 6 additions, before section 7
rep("## 7. Commands", (HERE / "out/sec6_add.md").read_text() + "## 7. Commands")
# ---- sections 7-9
rep("setsid nohup $PY $D/stale_cpu.py <variant> [<variant> ...] > $D/logs/<name>.log 2>&1 < /dev/null &   # CPU stale-val, 5-7 min per d192 pass\n```",
    "setsid nohup $PY $D/stale_cpu.py <variant> [<variant> ...] > $D/logs/<name>.log 2>&1 < /dev/null &   # CPU stale-val, 5-7 min per d192 pass\n"
    "# resume run [R]\n"
    "$PY $D/recheck.py    > $D/logs/recheck.txt     # independent re-check of six first-run pool reads (src/metrics.ndcg_per_feed, own z)\n"
    "$PY $D/read8.py      > $D/logs/read8.txt       # row 301: 85 stored families as a 0.2 admixture to W16\n"
    "$PY $D/read9.py      > $D/logs/read9.txt       # reverse check: lr 2e-3 vs 1e-3, hard-label width, GRU family\n"
    "$PY $D/read10.py     > $D/logs/read10.txt      # aux weight basis with the seeds finished now; student re-cools selected vs @last; harness control\n"
    "$PY $D/read11.py     > $D/logs/read11.txt      # checkpoint-kind robustness of the first run's passes\n"
    "$PY $D/crosscheck.py > $D/logs/crosscheck.txt  # my CPU stale-val files against the witness cache\n"
    "$PY $D/read4.py > $D/logs/read4.txt && $PY $D/mkblock.py   # CPU stale-val table of section 3.5\n"
    "$PY $D/read12.py     > $D/logs/read12.txt      # CPU stale-val of the declared W16P20x8 form, teacher axis, controls\n"
    "$PY $D/edit_report.py                          # applies the resume additions to this file (anchors asserted)\n```")
rep("- `reports/STALE_1115/veto.md` (this file).",
    "- `reports/STALE_1115/veto.md` (this file).\n"
    "- [R] resume run: `recheck.py`, `read8.py` .. `read12.py`, `crosscheck.py`, `edit_report.py`; `logs/recheck.txt`, `logs/read8.txt` .. `logs/read12.txt`, `logs/crosscheck.txt`; "
    "`out/read8.json` .. `out/read12.json`; `out/sec0_new.md`, `out/sec35_add.md`, `out/sec38_39.md`, `out/sec4_rows.md`, `out/sec4_notes.md`, `out/sec51.md`, `out/sec6_add.md` "
    "(the text blocks that `edit_report.py` inserted); `out/veto_report_before_resume.md` (this report as the first run left it); two addenda in `PREDECLARED.txt`.")
rep("- Rules written before the reads; one bar of mine was missed and is applied (section 3.5).",
    "- Rules written before the reads; one bar of mine was missed and is applied (section 3.5).\n"
    "- [R] Resume run. Found five processes of the first run alive (two bash waiters with one `stale_cpu.py` each, one bash waiter). Stopped two of my own `stale_cpu.py` jobs by exact "
    "PID after reading `/proc/<pid>/cmdline` (4062266 at 03:33Z, 4089312 at 03:42Z): the members they still had to compute were already in the witness agent's cache. No pattern kill; "
    "no process of anybody else touched. From 03:33Z at most one inference job plus one read script of mine ran at a time; from 03:42Z no job of mine was left running. "
    "Read scripts used 4-8 threads. The witness agent's cache (`nn_x/w12/stale/witness/cache/staleval/`) was read, never written. `nn_x/w12/kd/out/teacher_*.npy` and "
    "`nn/feat_cache/{val,test}/*.npy` were opened read-only (memory-mapped) for the two label-free checks of section 5.1. One checkpoint was loaded on the CPU to print its config.")
REP.write_text(s)
print("report written:", len(s.split("\n")), "lines")
