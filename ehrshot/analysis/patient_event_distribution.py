#!/usr/bin/env python3
"""
Generate a modern histogram of **# of Events per Patient** with:

* x-axis starting at **10** (dataset has no patients below that point).
* x-axis capped at 40 000.
* 200-bin resolution, pastel/viridis styling, small bar gaps.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make helper importable regardless of cwd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from plotting.base_plotting import plot_column_per_patient  # type: ignore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BINS = 150
X_MIN = 10      # <— NEW lower bound
X_MAX = 30_000
FIG_SIZE = (12, 8 * 2 / 3)
STYLE = "seaborn-v0_8-pastel"
CMAP = plt.cm.viridis
BAR_ALPHA = 0.85
BAR_GAP = 0.1   # proportion of bin width to leave empty


# ---------------------------------------------------------------------------
# Plot generator
# ---------------------------------------------------------------------------

def generate_patient_event_distribution_plot(
    path_to_data_csv: str | None = None,
    output_dir: str | None = None,
    bins: int = BINS,
    x_min: int = X_MIN,
    x_max: int = X_MAX,
    bar_gap: float = BAR_GAP,
) -> str:
    """Create the histogram PNG with a 10‒40 000 x-axis range."""

    # CSV location ------------------------------------------------------------
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
            raise FileNotFoundError("Could not locate *ehrshot.csv*.")

    # Output dir --------------------------------------------------------------
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data ---------------------------------------------------------------
    df = pd.read_csv(path_to_data_csv, low_memory=False)

    # Style tweaks ------------------------------------------------------------
    mpl.rcParams.update({"axes.spines.right": False, "axes.spines.top": False})
    plt.style.use(STYLE)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Histogram data ----------------------------------------------------------
    event_counts = df.groupby("patient_id", observed=False).size().to_numpy()
    counts, edges = np.histogram(event_counts, bins=bins, range=(x_min, x_max))

    # Draw bars with colour-mapped heights and gaps ---------------------------
    norm = mpl.colors.Normalize(vmin=counts.min(), vmax=counts.max())
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

    # Axis labels & limits ----------------------------------------------------
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("# of Events", weight="semibold")
    ax.set_ylabel("# of Patients", weight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save --------------------------------------------------------------------
    outfile = output_dir / "patient_event_count_distribution.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved ➜ {outfile}")
    return str(outfile)


if __name__ == "__main__":
    sys.exit(0 if generate_patient_event_distribution_plot() else 1)