#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

try:
    from ..utils import (
        LABELING_FUNCTION_2_PAPER_NAME,
        TASK_GROUP_2_LABELING_FUNCTION,
        TASK_GROUP_2_PAPER_NAME,
    )
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils import (
        LABELING_FUNCTION_2_PAPER_NAME,
        TASK_GROUP_2_LABELING_FUNCTION,
        TASK_GROUP_2_PAPER_NAME,
    )


def load_baseline_results(
    results_dir: str, tasks: List[str], models: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    baseline_results = {}

    for task in tasks:
        csv_path = os.path.join(results_dir, f"{task}/all_results.csv")

        df = pd.read_csv(csv_path)

        # filter for kNN results with full data (k=-1)
        knn_baseline = df[
            (df["head"] == "knn")
            & (df["k"] == -1)
            & (df["replicate"] == 0)  # not applicable for full data
        ]

        task_results = {}

        for model in models:
            model_data = knn_baseline[knn_baseline["model"] == model]
            if not model_data.empty:
                task_results[model] = {}
                for _, row in model_data.iterrows():
                    task_results[model][row["score"]] = row["value"]

        baseline_results[task] = task_results

    return baseline_results


def create_category_based_dimensionality_plots(
    results_df: pd.DataFrame,
    baseline_results: Dict,
    output_dir: str,
    include_baseline: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # get task categories
    task_groups = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    n_groups = len(task_groups)

    models = ["clmbr", "clinicalbert_type3_clinicalbert_pool"]
    model_labels = ["CLMBR", "ClinicalBERT"]
    methods = ["pca", "umap"]

    # define colors for each model-method combination
    colors = {
        ("clmbr", "pca"): "#1f77b4",
        ("clmbr", "umap"): "#9467bd",
        ("clinicalbert_type3_clinicalbert_pool", "pca"): "#ff7f0e",
        ("clinicalbert_type3_clinicalbert_pool", "umap"): "#d62728",
    }

    # create separate plots for AUROC and AUPRC
    for metric in ["auroc", "auprc"]:

        # create figure with 3 subplots (one for each category)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, task_group in enumerate(task_groups):
            ax = axes[idx]
            labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]

            # filter data for this task category
            category_data = results_df[
                (results_df["task"].isin(labeling_functions))
                & (results_df["metric"] == metric)
            ]

            if category_data.empty:
                ax.set_title(
                    f"{TASK_GROUP_2_PAPER_NAME[task_group]}\n(No data available)"
                )
                ax.set_xlabel("# of Dimensions")
                ax.set_ylabel(f"{metric.upper()}")
                continue

            # get all dimensions for this category
            all_dimensions = set(category_data["dimension"].unique())
            dimensions = sorted(list(all_dimensions))

            # Plot each model-method combination
            for model, model_label in zip(models, model_labels):
                for method in methods:
                    # filter data for this model, method, and metric
                    method_data = category_data[
                        (category_data["model"] == model)
                        & (category_data["method"] == method)
                    ]

                    if not method_data.empty:
                        # calculate mean performance across tasks in this category for each dimension
                        grouped = (
                            method_data.groupby("dimension")["score"]
                            .agg(["mean", "std"])
                            .reset_index()
                        )

                        color = colors.get((model, method), "gray")
                        label = f"{model_label} {method.upper()}"

                        ax.errorbar(
                            grouped["dimension"],
                            grouped["mean"],
                            yerr=grouped["std"],
                            color=color,
                            marker="o",
                            linewidth=3,
                            markersize=8,
                            label=label,
                            alpha=0.9,
                        )

            # aadd baselines if available and requested
            if include_baseline and baseline_results:
                for model, model_label in zip(models, model_labels):
                    all_baseline_scores = []

                    # get baseline scores for this model across all tasks in this category
                    for task in labeling_functions:
                        if (
                            task in baseline_results
                            and model in baseline_results[task]
                            and metric in baseline_results[task][model]
                        ):
                            all_baseline_scores.append(
                                baseline_results[task][model][metric]
                            )

                    if all_baseline_scores:
                        baseline_mean = np.mean(all_baseline_scores)
                        baseline_color = "#1f77b4" if model == "clmbr" else "#ff7f0e"
                        ax.axhline(
                            y=baseline_mean,
                            color=baseline_color,
                            linestyle="--",
                            linewidth=2,
                            alpha=0.7,
                            label=f"{model_label} 768D Baseline",
                        )

            # customize subplot
            ax.set_xlabel("# of Dimensions", fontsize=12)
            ax.set_ylabel(f"{metric.upper()}", fontsize=12)
            ax.set_title(
                f"{TASK_GROUP_2_PAPER_NAME[task_group]}", fontsize=14, fontweight="bold"
            )
            ax.set_xscale("log")
            if dimensions:
                ax.set_xticks(dimensions)
                ax.set_xticklabels([str(d) for d in dimensions])
            ax.grid(True, alpha=0.3)

            # Show legend on middle subplot to indicate it applies to all figures
            if idx == 1:
                legend = ax.legend(
                    fontsize=10,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.95),
                    ncol=2,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                )

            # make the plot look more polished
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.5)
            ax.spines["bottom"].set_linewidth(0.5)

        plt.tight_layout()

        # save with the specific filename
        output_file = os.path.join(
            output_dir, f"clean_{metric}_vs_dimensions_by_category.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"saved: clean_{metric}_vs_dimensions_by_category.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_dir")
    parser.add_argument("--no_baseline", action="store_true")

    args = parser.parse_args()

    # load results
    results_df = pd.read_csv(args.results_csv)

    # Get unique tasks and models from the data
    tasks = results_df["task"].unique().tolist()
    models = results_df["model"].unique().tolist()

    # Load baseline results if provided
    baseline_results = {}
    include_baseline = not args.no_baseline
    if include_baseline and args.baseline_dir:
        print(f"Loading baseline results from {args.baseline_dir}")
        baseline_results = load_baseline_results(args.baseline_dir, tasks, models)
    else:
        include_baseline = False

    create_category_based_dimensionality_plots(
        results_df, baseline_results, args.output_dir, include_baseline
    )

    print("dimensionality plotting completed successfully!")


if __name__ == "__main__":
    main()
