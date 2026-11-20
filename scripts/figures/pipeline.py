"""pipeline: the stages from raw parquet to the checked submission file, with hardware and run time."""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import Rectangle

import style

NAME = "pipeline"
TITLE = ("Nine stages from raw parquet to a checked submission file;\n"
         "the GPU stages add up to about 150 hours, the students alone to 82")
CAPTION = (
    "The stages of the final pipeline in the order they run, with the hardware and the run time of each in the "
    "original run. GPU-hours are wall-clock sums over runs on GPUs shared by several processes, so they are "
    "upper bounds."
)
ALT = (
    "A vertical chain of nine boxes between the competition data (17.9 GiB zipped) at the top and the submission "
    "file at the bottom: flatten, exploration probability, candidate inputs, teacher rankers, teacher files, "
    "students, fine-tune, ensemble and file, pre-submission gate. A column on the right gives hardware and run "
    "time in the original run: the CPU stages take seconds to about half an hour, encoder pre-training 5.2 "
    "GPU-hours, the 25 teacher rankers 58.0, the 20 students 81.8 and their fine-tunes 6.6."
)
SOURCE = "Data: docs/figures/data/pipeline_stages.csv, from the run logs of the final members."

X_BOX, W_BOX = style.MARGIN, 5.35
X_RIGHT = X_BOX + W_BOX + 0.3
H_BOX, H_PILL, GAP = 0.45, 0.34, 0.13
BAR_MAX = 1.15  # inches for the longest GPU-hours bar


def texts() -> style.Texts:
    return style.Texts(TITLE, CAPTION, ALT, SOURCE)


def make() -> None:
    d = pd.read_csv(style.DATA_DIR / "pipeline_stages.csv")
    total = d.gpu_hours.sum()
    assert len(d) == 9 and abs(total - 151.6) < 0.05

    fig = style.figure(8.7)
    head = style.header(fig, TITLE)
    ax = style.canvas(fig)
    y = fig.get_size_inches()[1] - head - 0.32

    ax.text(X_BOX, y, "Stage", fontsize=style.FS_SMALL, color=style.INK2, va="center")
    ax.text(X_RIGHT, y, "Original run: hardware, time", fontsize=style.FS_SMALL, color=style.INK2,
            va="center")
    y -= 0.2

    def pill(text: str) -> None:
        nonlocal y
        style.box(ax, X_BOX, y - H_PILL, W_BOX, H_PILL, lines=(text,), kind="data", pill=True)
        y -= H_PILL

    def down() -> None:
        nonlocal y
        style.arrow(ax, [(X_BOX + W_BOX / 2, y), (X_BOX + W_BOX / 2, y - GAP)], lw=1.1)
        y -= GAP

    pill("Competition data: parquet and embedding tables, 17.9 GiB zipped")
    for _, r in d.iterrows():
        down()
        yc = y - H_BOX / 2
        style.box(ax, X_BOX, y - H_BOX, W_BOX, H_BOX, kind="orange" if r.needs_gpu else "blue")
        ax.text(X_BOX + 0.13, yc + style.LINE / 2, f"{r.stage}", fontsize=style.FS_TEXT, fontweight="bold",
                color=style.INK, va="center", ha="left", zorder=3)
        ax.text(X_BOX + 0.36, yc + style.LINE / 2, r["name"], fontsize=style.FS_TEXT, fontweight="bold",
                color=style.INK, va="center", ha="left", zorder=3)
        ax.text(X_BOX + 0.36, yc - style.LINE / 2, r.what_it_does.replace(" x ", " × "), fontsize=style.FS_SMALL,
                color=style.INK, va="center", ha="left", zorder=3)
        has_bar = pd.notna(r.gpu_hours)
        ax.text(X_RIGHT, yc + (style.LINE / 2 if has_bar else 0), r.hardware_and_run_time, fontsize=style.FS_SMALL,
                color=style.INK, va="center", ha="left")
        if has_bar:
            w = BAR_MAX * r.gpu_hours / d.gpu_hours.max()
            ax.add_patch(Rectangle((X_RIGHT, yc - style.LINE / 2 - 0.055), w, 0.11, facecolor=style.ORANGE,
                                   edgecolor="none"))
            ax.text(X_RIGHT + w + 0.08, yc - style.LINE / 2, f"{r.gpu_hours:.1f} h, {r.gpu_hours_of}",
                    fontsize=style.FS_SMALL, color=style.INK, va="center", ha="left")
        y -= H_BOX
    down()
    pill("Submission file: 2,904,806 rows, one score per test candidate")

    y -= 0.3
    style.key(ax, X_BOX, y, [("data", "data on disk"), ("blue", "CPU stage"), ("orange", "stage that needs a GPU")])
    y -= 0.32
    ax.text(X_BOX, y, f"Bars: GPU-hours, wall-clock sums over runs on GPUs shared by several processes: "
                      f"{total:.1f} h in all.", fontsize=style.FS_SMALL, color=style.INK2, va="center")

    style.save(fig, NAME, texts())


if __name__ == "__main__":
    make()
