"""Assemble reports/SEG_1115/refill.md from the report parts of this directory."""
from pathlib import Path
H = Path("<repo>/nn_x/w12/seg/refill")
parts = ["rep_0_verdict.txt", "rep_1_method.txt", "rep_2_alignment.txt", "rep_3_passes.txt", "rep_4_read.txt", "rep_5_size.txt", "rep_6_arm.txt", "rep_7_notes.txt"]
txt = "\n".join((H / p).read_text().rstrip("\n") + "\n" for p in parts)
assert "—" not in txt
out = Path("<repo>/reports/SEG_1115/refill.md")
out.write_text(txt)
print(out, len(txt.splitlines()), "lines", len(txt), "chars")
