# archive/ (created 2026-11-13 23:13 UTC)

Reversible archive of files and directories that the active pipeline no longer needs. Nothing here is referenced by the
code paths that build, gate or reproduce the frozen candidate set (reports/FREEZE_PROTOCOL_1207.md section 2); the gate was
re-run on all eight candidates after the move (G3, Q2a, C1, C2, C3, Q1a: PASS; Pall, PF7x: PASS with their 6 recorded warnings).

Contents (original path preserved under archive/):
- backup_1112/      the 2026-11-12 backup; byte-identical copies of closed-experiment predictions were deleted (verified
                    sha256 against the surviving copy), the copies of the frozen-set files and all old submissions remain
- baseline_org/     the organisers' baseline
- preds/            1,089 member predictions of closed experiments (54 families), moved out of preds/
- preds_quarantine/, preds_x/, nnx/ (with the former nn_x/optim symlink target)   superseded prediction sets and the old nnx tree
- subs/             18 submission files that were never submitted
- RUNNING_PID.txt   stale PID file

MANIFEST.tsv: one line per action (time, kind, source, target, bytes, group). To restore a file: mv the target back to the
source path. The verified plan and its evidence: nn_x/insights/cleanup_1113/ (copied from the session scratch).
