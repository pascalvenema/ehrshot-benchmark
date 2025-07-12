#!/usr/bin/env python3

import os
import sys
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from ..utils import (
        LABELING_FUNCTION_2_PAPER_NAME,
        TASK_GROUP_2_PAPER_NAME,
        TASK_GROUP_2_LABELING_FUNCTION,
        MODEL_2_INFO,
        HEAD_2_INFO,
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    MODEL_2_INFO,
    HEAD_2_INFO,
)


def plot_all_heads_comparison(
    df_results: pd.DataFrame, score: str, path_to_output_dir: str
):
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    n_groups = len(task_groups)

    # create a 2x3 layout: top row = CLMBR, bottom row = ClinicalBERT
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # get shared prediction heads between CLMBR and ClinicalBERT Type 3
    clmbr_heads = set(MODEL_2_INFO["clmbr"]["heads"])
    cb_type3_heads = set(MODEL_2_INFO["clinicalbert_type3_clinicalbert_pool"]["heads"])
    shared_heads = sorted(list(clmbr_heads.intersection(cb_type3_heads)))

    # define colors for each head
    head_colors = {
        "lr_lbfgs": "#1f77b4",
        "knn": "#ff7f0e",
        "rf": "#2ca02c",
        "gbm": "#d62728",
    }

    # define markers for each head
    head_markers = {"lr_lbfgs": "s", "knn": "^", "rf": "o", "gbm": "D"}

    # models to plot
    models = [
        ("clmbr", "CLMBR"),
        ("clinicalbert_type3_clinicalbert_pool", "ClinicalBERT"),
    ]

    for model_idx, (model_name, model_label) in enumerate(models):
        for task_idx, task_group in enumerate(task_groups):
            ax = axes[model_idx, task_idx]
            labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]

            # get task-specific data
            task_data = df_results[
                (df_results["score"] == score)
                & (df_results["labeling_function"].isin(labeling_functions))
            ]

            # get all `k` shots tested and handle "All" data point
            ks = sorted(task_data["k"].unique().tolist())
            x_tick_labels = [str(k) for k in ks]
            if -1 in ks:
                ks.remove(-1)
                full_data_k = 2 * max(ks) if ks else 256
                ks.append(full_data_k)
                x_tick_labels = [str(k) if k != full_data_k else "All" for k in ks]
                # update the data to use fake k value for plotting
                task_data = task_data.copy()
                task_data.loc[task_data["k"] == -1, "k"] = full_data_k
            else:
                full_data_k = None

            # plot each head for this model
            for head in shared_heads:
                model_head_data = task_data[
                    (task_data["model"] == model_name) & (task_data["head"] == head)
                ]

                if not model_head_data.empty:
                    grouped = (
                        model_head_data.groupby("k")["value"]
                        .agg(["mean", "std"])
                        .reset_index()
                    )

                    # create label for legend
                    head_label = HEAD_2_INFO[head]["label"]

                    ax.errorbar(
                        grouped["k"],
                        grouped["mean"],
                        yerr=grouped["std"],
                        color=head_colors[head],
                        linestyle="-",
                        marker=head_markers[head],
                        label=head_label,
                        linewidth=2.5,
                        markersize=7,
                        alpha=0.9,
                    )

            # cstomize subplot
            ax.set_xlabel("# of Train Examples per Class", fontsize=12)
            ax.set_ylabel(f"{score.upper()}", fontsize=12)

            # Set title: model name for first row, task group for all
            if model_idx == 0:  # Top row
                ax.set_title(
                    f"{model_label}\n{TASK_GROUP_2_PAPER_NAME[task_group]}",
                    fontsize=14,
                    fontweight="bold",
                )
            else:  # Bottom row
                ax.set_title(
                    f"{model_label}\n{TASK_GROUP_2_PAPER_NAME[task_group]}",
                    fontsize=14,
                    fontweight="bold",
                )

            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels(x_tick_labels)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc="best")

            # Make the plot look more polished
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()

    # save to output directory
    output_file = os.path.join(
        path_to_output_dir, f"clmbr_vs_clinicalbert_type3_by_model_{score}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close("all")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_results_dir", required=True, type=str)
    parser.add_argument("--path_to_output_dir", required=True, type=str)

    args = parser.parse_args()

    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Load all results
    dfs: List[pd.DataFrame] = []
    for labeling_function in tqdm(LABELING_FUNCTION_2_PAPER_NAME.keys()):
        path_to_csv = os.path.join(
            args.path_to_results_dir, f"{labeling_function}/all_results.csv"
        )
        if not os.path.exists(path_to_csv):
            continue
        dfs.append(pd.read_csv(path_to_csv))

    if not dfs:
        return

    df_results = pd.concat(dfs, ignore_index=True)

    # check what models and heads are available
    available_models = df_results["model"].unique()

    clmbr_data = df_results[df_results["model"] == "clmbr"]
    cb_type3_data = df_results[
        df_results["model"] == "clinicalbert_type3_clinicalbert_pool"
    ]

    # generate plots for both AUROC and AUPRC
    for score in ["auroc", "auprc"]:
        if score in df_results["score"].unique():
            plot_all_heads_comparison(df_results, score, args.path_to_output_dir)


if __name__ == "__main__":
    main()
