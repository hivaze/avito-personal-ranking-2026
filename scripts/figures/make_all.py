"""Draw every figure of docs/figures and write docs/figures/figures.json (title, caption and alt text of each).

Usage, from the repository root:

    python scripts/figures/make_all.py                 # all ten figures
    python scripts/figures/make_all.py pipeline ensemble

Each figure reads only the small files in docs/figures/data and writes <name>.svg and <name>.png (2x).
figures.json is always rewritten for all ten figures, so its numbers follow the data even when only some
figures are redrawn. The output is byte-for-byte the same on every run with the same matplotlib version.
"""

from __future__ import annotations

import json
import sys

import architecture
import data_timeline
import distillation
import ensemble
import experiments_outcomes
import near_copies_noise
import pipeline
import public_progress
import style
import val_vs_public
import work_loop

MODULES = [public_progress, val_vs_public, data_timeline, pipeline, architecture, distillation, ensemble,
           near_copies_noise, experiments_outcomes, work_loop]


def main(names: list[str]) -> None:
    known = {m.NAME: m for m in MODULES}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown figure {unknown}; choose from {sorted(known)}")
    for m in MODULES:
        if names and m.NAME not in names:
            continue
        m.make()
        print(f"wrote docs/figures/{m.NAME}.svg and .png")
    index = []
    for m in MODULES:
        t = m.texts()
        index.append({"name": m.NAME, "svg": f"{m.NAME}.svg", "png": f"{m.NAME}.png",
                      "title": t.title.replace("\n", " "), "caption": t.caption, "alt": t.alt,
                      "source_note": t.source})
    (style.FIG_DIR / "figures.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
