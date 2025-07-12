#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plotting.base_plotting import plot_column_per_patient  

# config
X_MIN = 10  # lower bound,, dataset has no patients below this point
CUTOFF = 10_000  # everything above this goes to "rest"
BINS = 40
FIG_SIZE = (12, 8 * 2 / 3)
STYLE = "seaborn-v0_8-pastel"
CMAP = plt.cm.viridis
BAR_ALPHA = 0.85
BAR_GAP = 0.1  # proportion of bin width to leave empty


# plot
def generate_patient_event_distribution_plot(
    path_to_data_csv: str | None = None,
    output_dir: str | None = None,
    cutoff: int = CUTOFF,
    bins: int = BINS,
    x_min: int = X_MIN,
    bar_gap: float = BAR_GAP,
) -> str:

    # CSV location
    if path_to_data_csv is None:
        for rel in (
            "./EHRSHOT_ASSETS/data/ehrshot.csv",
            "../EHRSHOT_ASSETS/data/ehrshot.csv",
            "../../EHRSHOT_ASSETS/data/ehrshot.csv",
        ):
            if os.path.exists(rel):
                path_to_data_csv = rel
                break
        else:
            raise FileNotFoundError("ehrshot csv not found")

    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_to_data_csv, low_memory=False)

    mpl.rcParams.update({"axes.spines.right": False, "axes.spines.top": False})
    plt.style.use(STYLE)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # get event counts
    event_counts = df.groupby("patient_id", observed=False).size().to_numpy()

    # separate data: below cutoff and above cutoff
    below_cutoff = event_counts[event_counts <= cutoff]
    above_cutoff = event_counts[event_counts > cutoff]

    # create histogram for below-cutoff data
    counts, edges = np.histogram(below_cutoff, bins=bins, range=(x_min, cutoff))

    # draw regular histogram bars for below-cutoff data
    norm = mpl.colors.Normalize(vmin=0, vmax=max(counts.max(), len(above_cutoff)))
    for left, width, height in zip(edges[:-1], np.diff(edges), counts, strict=True):
        eff_w = width * (1 - bar_gap)
        offset = (width - eff_w) / 2
        ax.bar(
            left + offset,
            height,
            width=eff_w,
            align="edge",
            color=CMAP(norm(height)),
            alpha=BAR_ALPHA,
            ec="white",
            lw=0.3,
        )

    # add the "rest" bar for above-cutoff data
    if len(above_cutoff) > 0:
        rest_width = (cutoff - x_min) / bins
        rest_left = cutoff + rest_width * 0.5
        rest_height = len(above_cutoff)

        ax.bar(
            rest_left,
            rest_height,
            width=rest_width * (1 - bar_gap),
            align="center",
            color=CMAP(norm(rest_height)),
            alpha=BAR_ALPHA,
            ec="white",
            lw=0.3,
        )

    ax.set_xlim(x_min, cutoff + (cutoff - x_min) / bins * 2)  # Make room for rest bar
    ax.set_xlabel("# of Events", weight="semibold")
    ax.set_ylabel("# of Patients", weight="semibold")

    from matplotlib.ticker import FixedLocator

    tick_positions = [10, 2000, 4000, 6000, 8000, 10000]
    rest_pos = cutoff + (cutoff - x_min) / bins * 0.5
    tick_labels = ["10", "2K", "4K", "6K", "8K", "10K+"]

    ax.set_xticks(tick_positions + [rest_pos])
    ax.set_xticklabels(tick_labels + [""])

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()

    outfile = output_dir / "patient_event_count_distribution.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"histogram with {cutoff // 1000}k cutoff saved to {outfile}")
    return str(outfile)


if __name__ == "__main__":
    plot_file = generate_patient_event_distribution_plot()
    print(f"generated plot successfully: {plot_file}")
    sys.exit(0)
