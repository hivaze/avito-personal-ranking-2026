"""Export the catalogue of docs/07-what-failed.md to docs/catalogue.csv, one row per idea.

    python scripts/export_catalogue.py            # rewrite docs/catalogue.csv, print the counts
    python scripts/export_catalogue.py --check    # write nothing, exit 1 if the stored CSV differs from the page

The page holds one table per theme under "The catalogue by theme", each headed
"### <theme> (<n>)", with the columns Idea, Outcome, Measured effect and Where.
The themes, their order, the six outcome classes and the counts per theme come
from ``scripts/figures/count_catalogue.py``, the parser behind the outcomes
figure. This script reads the cells of the same rows and stops if its counts
per theme differ from that parser's.

Columns, in the order of the page:

    theme            heading of the table, without its count
    idea             first cell
    outcome          second cell, as the page words it
    measured_effect  third cell
    where            fourth cell; its paths start at the repository root
    where_path       target of a markdown link in the fourth cell, as a path
                     from the repository root with its anchor; empty when the
                     cell holds no link
    outcome_class    class the counter assigns: adopted, helped_then_superseded,
                     no_effect, hurt, inconclusive or not_run

Cells are plain text: a markdown link is reduced to its text and code marks
are removed. The file is UTF-8 and comma-separated, with a header row and
standard quoting, in the order of the page. CPU only, under a second. Run it
after every edit of the catalogue, next to ``count_catalogue.py``.
"""
from __future__ import annotations

import argparse
import csv
import io
import posixpath
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["theme", "idea", "outcome", "measured_effect", "where", "where_path", "outcome_class"]
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def plain(cell: str) -> str:
    """A table cell without markdown: links reduced to their text, code marks removed."""
    return LINK.sub(r"\1", cell).replace("`", "")


def link_paths(cell: str, page: Path) -> str:
    """Targets of the markdown links of a cell, as paths from the repository root, joined by "; "."""
    targets = []
    for _, target in LINK.findall(cell):
        if "://" not in target:
            path, hash_mark, anchor = target.partition("#")
            folder = page.parent.relative_to(ROOT).as_posix()
            path = posixpath.normpath(posixpath.join(folder, path)) if path else page.relative_to(ROOT).as_posix()
            target = path + hash_mark + anchor
        targets.append(target)
    return "; ".join(targets)


def catalogue_rows(page: Path, outcomes: dict[str, str], themes: list[str]) -> list[dict[str, str]]:
    """One record per table row under the given theme headings, in the order of the page."""
    rows, theme = [], None
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            name = line.lstrip("#").strip().rsplit(" (", 1)[0]
            theme = name if line.startswith("### ") and name in themes else None
            continue
        if theme is None or not line.startswith("|") or set(line.strip()) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[:2] == ["Idea", "Outcome"]:
            continue
        assert len(cells) == 4, f"{len(cells)} cells under {theme!r}: {line[:80]}"
        idea, outcome, effect, where = cells
        rows.append({"theme": theme, "idea": plain(idea), "outcome": outcome, "measured_effect": plain(effect),
                     "where": plain(where), "where_path": link_paths(where, page),
                     "outcome_class": outcomes[outcome]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-file", metavar="FILE", type=Path, default=ROOT / "docs" / "catalogue.csv",
                        help="the CSV to write or to check (default: docs/catalogue.csv)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if the stored CSV differs from the page")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts" / "figures"))   # the parser imports its neighbour style.py by name
    import count_catalogue

    counts = count_catalogue.count_page()                    # theme -> outcome counts, in the order of the page
    rows = catalogue_rows(count_catalogue.PAGE, count_catalogue.OUTCOMES, list(counts))
    for theme, expected in counts.items():
        found = Counter(row["outcome_class"] for row in rows if row["theme"] == theme)
        assert found == expected, f"{theme!r}: the counter has {dict(expected)}, the export {dict(found)}"
    empty = [(row["idea"][:40], name) for row in rows for name in FIELDS if name != "where_path" and not row[name]]
    assert not empty, f"empty cells: {empty}"

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    total = Counter(row["outcome_class"] for row in rows)
    summary = ", ".join(f"{total[name]} {name}" for name in count_catalogue.OUTCOMES.values())
    print(f"{len(rows)} ideas in {len(counts)} themes: {summary}")
    out_file = args.out_file.resolve()
    shown = out_file.relative_to(ROOT).as_posix() if out_file.is_relative_to(ROOT) else str(args.out_file)
    if args.check:
        stored = args.out_file.read_text(encoding="utf-8") if args.out_file.exists() else None
        print(f"{shown}: {'equal to the page' if stored == out.getvalue() else 'DIFFERS from the page'}")
        raise SystemExit(0 if stored == out.getvalue() else 1)
    args.out_file.write_text(out.getvalue(), encoding="utf-8")
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
