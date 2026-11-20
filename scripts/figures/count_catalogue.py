"""Count the catalogue tables of docs/07-what-failed.md into docs/figures/data/experiments_outcomes.csv.

Usage, from the repository root, after any edit of the catalogue:

    python scripts/figures/count_catalogue.py           # rewrite the counts, print what changed
    python scripts/figures/count_catalogue.py --check   # exit 1 if the CSV and the page disagree

The page holds one table per theme under "## The catalogue by theme", each headed "### <theme> (<n>)", one row
per idea and the outcome in the second column. A heading is matched to a CSV row by its theme or its label;
the theme and label columns are kept as they are. The script stops when a heading has no row, a row has no
heading, an outcome is not one of the six, or a heading's count differs from its number of rows.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from collections import Counter

import style

PAGE = style.ROOT / "docs" / "07-what-failed.md"
CSV = style.DATA_DIR / "experiments_outcomes.csv"
OUTCOMES = {"adopted": "adopted", "helped then superseded": "helped_then_superseded", "no effect": "no_effect",
            "hurt": "hurt", "inconclusive": "inconclusive", "not run": "not_run"}


def count_page() -> dict[str, Counter]:
    """Outcome counts per theme heading of the catalogue section."""
    text = PAGE.read_text().split("\n## The catalogue by theme\n", 1)[1].split("\n## ", 1)[0]
    counts: dict[str, Counter] = {}
    stated: dict[str, int] = {}
    theme = None
    for line in text.splitlines():
        m = re.fullmatch(r"### (.+) \((\d+)\)\s*", line)
        if m:
            theme = m.group(1)
            counts[theme], stated[theme] = Counter(), int(m.group(2))
            continue
        if theme is None or not line.startswith("|") or re.fullmatch(r"\|[-| :]+\|?\s*", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[:2] == ["Idea", "Outcome"]:
            continue
        assert cells[1] in OUTCOMES, f"unknown outcome {cells[1]!r} under {theme!r}: {line[:80]}"
        counts[theme][OUTCOMES[cells[1]]] += 1
    for theme, n in stated.items():
        assert sum(counts[theme].values()) == n, f"{theme!r}: heading says {n}, table has {sum(counts[theme].values())}"
    return counts


def main(check: bool) -> None:
    rows = list(csv.DictReader(CSV.open(newline="")))
    fields = list(rows[0])
    counts = count_page()
    by_name = {r["theme"]: r for r in rows} | {r["label"]: r for r in rows}
    missing = [t for t in counts if t not in by_name]
    assert not missing, f"headings without a CSV row: {missing}"
    matched = {id(by_name[t]) for t in counts}
    orphans = [r["theme"] for r in rows if id(r) not in matched]
    assert not orphans, f"CSV rows without a heading: {orphans}"
    changes = []
    for theme, c in counts.items():
        r = by_name[theme]
        new = {k: str(c.get(k, 0)) for k in OUTCOMES.values()} | {"total": str(sum(c.values()))}
        changes += [f"{r['label']}: {k} {r[k]} -> {v}" for k, v in new.items() if r[k] != v]
        r.update(new)
    print("\n".join(changes) or "no change", f"({sum(int(r['total']) for r in rows)} entries)")
    if check:
        raise SystemExit(1 if changes else 0)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    CSV.write_text(out.getvalue())


if __name__ == "__main__":
    main("--check" in sys.argv[1:])
