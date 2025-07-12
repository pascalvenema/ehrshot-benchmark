#!/usr/bin/env python3


import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, kruskal, mannwhitneyu, friedmanchisquare
from scipy import stats
from statsmodels.stats.multitest import multipletests
import argparse
import os
import glob
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

LABELING_FUNCTION_2_PAPER_NAME = {
    "guo_los": "Long LOS",
    "guo_readmission": "30-day Readmission",
    "guo_icu": "ICU Admission",
    "new_pancan": "Pancreatic Cancer",
    "new_celiac": "Celiac",
    "new_lupus": "Lupus",
    "new_acutemi": "Acute MI",
    "new_hypertension": "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    "chexpert": "ChexPert",
}


def load_results_data(results_dir: str) -> pd.DataFrame:
    result_files = glob.glob(os.path.join(results_dir, "*/all_results.csv"))

    if not result_files:
        raise ValueError("no result files found")

    all_data = []
    for file in result_files:
        task = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df["labeling_function"] = task
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def analyze_rq1_1_kshot_aware(
    df_results: pd.DataFrame, score: str = "auroc", alpha: float = 0.05
) -> Dict:

    # get ClinicalBERT pooling data across all k values
    cb_data = df_results[
        (df_results["score"] == score)
        & (df_results["model"].str.contains("clinicalbert", na=False))
        & (df_results["model"].str.endswith("_clinicalbert_pool"))
        & (df_results["head"] == "lr_lbfgs")
    ].copy()

    cb_data["cb_type"] = cb_data["model"].str.extract(r"clinicalbert_(type\d)_")[0]
    cb_data = cb_data.dropna(subset=["cb_type"])

    k_values = sorted([k for k in cb_data["k"].unique() if k != -1]) + [-1]

    results = {"by_k": {}, "overall_across_k": {}, "k_trend_analysis": {}}

    # 1. analysis by each k value
    type_performance_by_k = {k: {} for k in k_values}

    for k_val in k_values:
        k_data = cb_data[cb_data["k"] == k_val]

        if len(k_data["cb_type"].unique()) < 2:
            continue

        # collect scores for each type
        type_groups = []
        type_names = []
        for cb_type in ["type1", "type2", "type3"]:
            type_scores = k_data[k_data["cb_type"] == cb_type]["value"].values
            if len(type_scores) > 0:
                type_groups.append(type_scores)
                type_names.append(cb_type)
                type_performance_by_k[k_val][cb_type] = np.mean(type_scores)

        if len(type_groups) >= 2:
            # kruskal-wallis for this k value
            kw_stat, kw_p = kruskal(*type_groups)

            # post-hoc pairwise comparisons
            pairwise_results = {}
            for i, type1 in enumerate(type_names):
                for j, type2 in enumerate(type_names[i + 1 :], i + 1):
                    stat, p_val = mannwhitneyu(
                        type_groups[i], type_groups[j], alternative="two-sided"
                    )
                    mean_diff = np.mean(type_groups[j]) - np.mean(type_groups[i])
                    pairwise_results[f"{type1}_vs_{type2}"] = {
                        "mean_diff": mean_diff,
                        "p_value": p_val,
                        "significant": p_val < alpha,
                    }

            results["by_k"][k_val] = {
                "kruskal_wallis": {
                    "statistic": kw_stat,
                    "p_value": kw_p,
                    "significant": kw_p < alpha,
                },
                "pairwise": pairwise_results,
            }

    # 2. trend analysis
    if len(type_performance_by_k) > 3:
        k_vals_for_trend = []
        type3_advantage = []

        for k_val in k_values:
            if (
                k_val != -1
                and "type1" in type_performance_by_k[k_val]
                and "type3" in type_performance_by_k[k_val]
            ):
                k_vals_for_trend.append(k_val)
                advantage = (
                    type_performance_by_k[k_val]["type3"]
                    - type_performance_by_k[k_val]["type1"]
                )
                type3_advantage.append(advantage)

        if len(k_vals_for_trend) > 3:
            corr, corr_p = stats.pearsonr(k_vals_for_trend, type3_advantage)

            if corr > 0:
                trend_desc = "type3 advantage increases with more training data"
            else:
                trend_desc = "type3 advantage decreases with more training data"

            results["k_trend_analysis"] = {
                "correlation": corr,
                "correlation_p": corr_p,
                "interpretation": trend_desc,
                "k_values": k_vals_for_trend,
                "advantages": type3_advantage,
            }

    # 3. overall friedman test across all k values
    friedman_data = []
    for task in cb_data["labeling_function"].unique():
        task_data = cb_data[cb_data["labeling_function"] == task]

        for k_val in k_values:
            k_task_data = task_data[task_data["k"] == k_val]

            type1_scores = k_task_data[k_task_data["cb_type"] == "type1"][
                "value"
            ].values
            type2_scores = k_task_data[k_task_data["cb_type"] == "type2"][
                "value"
            ].values
            type3_scores = k_task_data[k_task_data["cb_type"] == "type3"][
                "value"
            ].values

            if (
                len(type1_scores) > 0
                and len(type2_scores) > 0
                and len(type3_scores) > 0
            ):
                friedman_data.append(
                    [
                        np.mean(type1_scores),
                        np.mean(type2_scores),
                        np.mean(type3_scores),
                    ]
                )

    if len(friedman_data) > 5:
        friedman_data = np.array(friedman_data)
        friedman_stat, friedman_p = friedmanchisquare(
            friedman_data[:, 0],  # type1
            friedman_data[:, 1],  # type2
            friedman_data[:, 2],  # type3
        )

        type_means = {
            "type1": np.mean(friedman_data[:, 0]),
            "type2": np.mean(friedman_data[:, 1]),
            "type3": np.mean(friedman_data[:, 2]),
        }

        results["overall_across_k"]["friedman"] = {
            "statistic": friedman_stat,
            "p_value": friedman_p,
            "significant": friedman_p < alpha,
            "type_means": type_means,
        }

    return results


def analyze_rq1_2_kshot_aware(
    df_results: pd.DataFrame, score: str = "auroc", alpha: float = 0.05
) -> Dict:

    comparison_data = df_results[
        (df_results["score"] == score)
        & (df_results["head"] == "lr_lbfgs")
        & (df_results["model"].isin(["clmbr", "clinicalbert_type3_clinicalbert_pool"]))
    ].copy()

    k_values = sorted([k for k in comparison_data["k"].unique() if k != -1]) + [-1]

    results = {"by_k": {}, "k_trend_analysis": {}, "summary": {}}

    clmbr_advantages = []
    k_vals_for_trend = []

    # 1. analysis by each k value
    for k_val in k_values:
        k_data = comparison_data[comparison_data["k"] == k_val]

        clmbr_scores = []
        cb_scores = []

        for task in k_data["labeling_function"].unique():
            task_data = k_data[k_data["labeling_function"] == task]

            clmbr_task = task_data[task_data["model"] == "clmbr"]["value"].values
            cb_task = task_data[
                task_data["model"] == "clinicalbert_type3_clinicalbert_pool"
            ]["value"].values

            if len(clmbr_task) > 0 and len(cb_task) > 0:
                clmbr_scores.append(np.mean(clmbr_task))
                cb_scores.append(np.mean(cb_task))

        if len(clmbr_scores) > 1:
            clmbr_scores = np.array(clmbr_scores)
            cb_scores = np.array(cb_scores)

            wilcoxon_stat, wilcoxon_p = wilcoxon(
                clmbr_scores, cb_scores, alternative="two-sided"
            )
            mean_diff = np.mean(clmbr_scores - cb_scores)
            clmbr_wins = np.sum(clmbr_scores > cb_scores)

            sig_marker = (
                "***"
                if wilcoxon_p < 0.001
                else "**" if wilcoxon_p < 0.01 else "*" if wilcoxon_p < 0.05 else "ns"
            )

            print(
                f"\nk = {k_val:>3}: mean diff = {mean_diff:+.3f}, p = {wilcoxon_p:.4f} {sig_marker}, clmbr wins = {clmbr_wins}/{len(clmbr_scores)}"
            )

            results["by_k"][k_val] = {
                "mean_diff": mean_diff,
                "wilcoxon_p": wilcoxon_p,
                "clmbr_wins": int(clmbr_wins),
                "total_tasks": len(clmbr_scores),
                "significant": wilcoxon_p < alpha,
            }

            # store for trend analysis
            if k_val != -1:
                k_vals_for_trend.append(k_val)
                clmbr_advantages.append(mean_diff)

    # 2. trend analysis
    if len(k_vals_for_trend) > 3:
        corr, corr_p = stats.pearsonr(k_vals_for_trend, clmbr_advantages)

        if corr > 0:
            trend_desc = "clmbr advantage increases with more training data"
        else:
            trend_desc = "clmbr advantage decreases with more training data"

        results["k_trend_analysis"] = {
            "correlation": corr,
            "correlation_p": corr_p,
            "interpretation": trend_desc,
            "k_values": k_vals_for_trend,
            "advantages": clmbr_advantages,
        }

    # 3. summary statistics
    all_significant = sum(
        1 for k_res in results["by_k"].values() if k_res["significant"]
    )
    total_k_vals = len(results["by_k"])

    if results["by_k"]:
        avg_advantage = np.mean([res["mean_diff"] for res in results["by_k"].values()])
        avg_win_rate = np.mean(
            [res["clmbr_wins"] / res["total_tasks"] for res in results["by_k"].values()]
        )

        results["summary"] = {
            "significant_k_count": all_significant,
            "total_k_count": total_k_vals,
            "avg_advantage": avg_advantage,
            "avg_win_rate": avg_win_rate,
        }

    return results


def analyze_rq2_1_fair_comparison(
    df_results: pd.DataFrame, score: str = "auroc", alpha: float = 0.05
) -> Dict:

    linear_heads = ["lr_lbfgs"]
    nonlinear_heads = ["knn", "rf", "gbm"]

    # use k=-1 for this analysis 
    full_data = df_results[
        (df_results["score"] == score) & (df_results["k"] == -1)
    ].copy()

    results = {"by_model": {}, "summary": {}}

    model_types = ["clmbr", "clinicalbert_type3_clinicalbert_pool"]

    for model_type in model_types:
        model_data = full_data[full_data["model"] == model_type]

        if model_data.empty:
            continue

        tasks = model_data["labeling_function"].unique()

        linear_task_scores = []
        best_nonlinear_task_scores = []
        task_improvements = []
        best_heads_used = []

        for task in tasks:
            task_data = model_data[model_data["labeling_function"] == task]

            # linear performance
            linear_scores = task_data[task_data["head"].isin(linear_heads)][
                "value"
            ].values

            # find best non-linear performance
            best_nonlinear_score = -1
            best_nonlinear_head = None

            for head in nonlinear_heads:
                head_scores = task_data[task_data["head"] == head]["value"].values
                if len(head_scores) > 0:
                    head_mean = np.mean(head_scores)
                    if head_mean > best_nonlinear_score:
                        best_nonlinear_score = head_mean
                        best_nonlinear_head = head

            if len(linear_scores) > 0 and best_nonlinear_score > -1:
                linear_mean = np.mean(linear_scores)
                improvement = best_nonlinear_score - linear_mean

                linear_task_scores.append(linear_mean)
                best_nonlinear_task_scores.append(best_nonlinear_score)
                task_improvements.append(improvement)
                best_heads_used.append(best_nonlinear_head)

        # overall analysis
        if len(task_improvements) > 1:
            task_improvements = np.array(task_improvements)
            linear_scores_arr = np.array(linear_task_scores)
            nonlinear_scores_arr = np.array(best_nonlinear_task_scores)

            # paired tests: best non-linear vs linear
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                nonlinear_scores_arr, linear_scores_arr, alternative="greater"
            )

            mean_improvement = np.mean(task_improvements)
            positive_improvements = np.sum(task_improvements > 0)

            # effect size
            cohens_d = mean_improvement / np.std(task_improvements, ddof=1)

            # confidence interval
            ci_lower = np.percentile(task_improvements, 2.5)
            ci_upper = np.percentile(task_improvements, 97.5)

            # most commonly used best head
            from collections import Counter

            head_counts = Counter(best_heads_used)
            most_common_head = head_counts.most_common(1)[0]

            results["by_model"][model_type] = {
                "mean_improvement": mean_improvement,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "positive_count": int(positive_improvements),
                "total_tasks": len(task_improvements),
                "wilcoxon_p": wilcoxon_p,
                "cohens_d": cohens_d,
                "significant": wilcoxon_p < alpha,
                "best_head_counts": dict(head_counts),
                "most_common_head": most_common_head[0],
            }

    return results


def analyze_rq2_2_dimensionality_analysis(
    dimensionality_file: str, score: str = "auroc", alpha: float = 0.05
) -> Dict:
    if not os.path.exists(dimensionality_file):
        return {}

    try:
        df_dim = pd.read_csv(dimensionality_file)
    except Exception as e:
        return {}

    df_metric = df_dim[df_dim["metric"] == score].copy()

    if df_metric.empty:
        return {}

    results = {"dimension_effects": {}, "optimal_dimensions": {}, "trend_analysis": {}}

    models = ["clmbr", "clinicalbert_type3_clinicalbert_pool"]
    model_names = ["CLMBR", "ClinicalBERT"]
    methods = ["pca", "umap"]

    # 1. test if dimensionality significantly affects kNN performance
    for model, model_name in zip(models, model_names):
        model_data = df_metric[df_metric["model"] == model]

        if model_data.empty:
            continue

        results["dimension_effects"][model] = {}

        for method in methods:
            method_data = model_data[model_data["method"] == method]

            if method_data.empty:
                continue

            # group scores by dimension
            dimension_groups = []
            dimensions = sorted(method_data["dimension"].unique())

            for dim in dimensions:
                dim_scores = method_data[method_data["dimension"] == dim][
                    "score"
                ].values
                if len(dim_scores) > 0:
                    dimension_groups.append(dim_scores)

            if (
                len(dimension_groups) >= 3
            ):  # need at least 3 dimensions for Kruskal-Wallis
                kw_stat, kw_p = kruskal(*dimension_groups)

                # calculate means for each dimension
                dim_means = [np.mean(group) for group in dimension_groups]
                best_dim_idx = np.argmax(dim_means)
                best_dim = dimensions[best_dim_idx]
                best_score = dim_means[best_dim_idx]

                results["dimension_effects"][model][method] = {
                    "kruskal_wallis_stat": kw_stat,
                    "kruskal_wallis_p": kw_p,
                    "significant": kw_p < alpha,
                    "best_dimension": best_dim,
                    "best_score": best_score,
                    "dimension_means": dict(zip(dimensions, dim_means)),
                }

    # 2. trend analysis
    for model, model_name in zip(models, model_names):
        model_data = df_metric[df_metric["model"] == model]

        if model_data.empty:
            continue

        results["trend_analysis"][model] = {}

        for method in methods:
            method_data = model_data[model_data["method"] == method]

            if method_data.empty:
                continue

            # calculate mean performance across tasks for each dimension
            dim_performance = (
                method_data.groupby("dimension")["score"]
                .agg(["mean", "count"])
                .reset_index()
            )

            if len(dim_performance) >= 4:  # need enoug points for correlation
                dimensions = dim_performance["dimension"].values
                mean_scores = dim_performance["mean"].values

                # test for correlation
                corr, corr_p = stats.pearsonr(dimensions, mean_scores)

                # also test with log-transformed dimensions
                log_dims = np.log(dimensions)
                log_corr, log_corr_p = stats.pearsonr(log_dims, mean_scores)

                if abs(log_corr) > abs(corr):
                    best_corr, best_p = log_corr, log_corr_p
                    trend_type = "logarithmic"
                else:
                    best_corr, best_p = corr, corr_p
                    trend_type = "linear"

                if best_p < alpha:
                    if best_corr > 0:
                        trend_desc = (
                            f"performance increases with dimension ({trend_type})"
                        )
                    else:
                        trend_desc = (
                            f"performance decreases with dimension ({trend_type})"
                        )
                else:
                    trend_desc = "no significant trend"

                results["trend_analysis"][model][method] = {
                    "linear_correlation": corr,
                    "linear_p": corr_p,
                    "log_correlation": log_corr,
                    "log_p": log_corr_p,
                    "best_correlation": best_corr,
                    "best_p": best_p,
                    "trend_type": trend_type,
                    "significant": best_p < alpha,
                    "trend_description": trend_desc,
                }

    # 3. find optimal dimensions
    for model, model_name in zip(models, model_names):
        model_data = df_metric[df_metric["model"] == model]

        if model_data.empty:
            continue

        results["optimal_dimensions"][model] = {}

        for method in methods:
            method_data = model_data[model_data["method"] == method]

            if not method_data.empty:
                # find optimal dimension (highest mean performance across tasks)
                dim_performance = (
                    method_data.groupby("dimension")["score"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                best_row = dim_performance.loc[dim_performance["mean"].idxmax()]

                optimal_dim = int(best_row["dimension"])
                optimal_score = best_row["mean"]
                optimal_std = best_row["std"]
                n_tasks = int(best_row["count"])

                results["optimal_dimensions"][model][method] = {
                    "optimal_dimension": optimal_dim,
                    "optimal_score": optimal_score,
                    "score_std": optimal_std,
                    "n_tasks": n_tasks,
                }

    return results




