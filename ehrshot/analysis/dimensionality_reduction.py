#!/usr/bin/env python3


import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Set, Tuple
from loguru import logger
import sys
import json
import time
from datetime import datetime

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dimensionality_analysis")
)

from reducers import get_reducer
from evaluator import kNNEvaluator
from plotting import create_all_plots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    get_labels_and_features,
    get_patient_splits_by_idx,
    convert_multiclass_to_binary_labels,
)
from femr.labelers import load_labeled_patients


def load_existing_results(output_dir: str) -> Tuple[pd.DataFrame, Set[str]]:

    results_path = os.path.join(output_dir, "dimensionality_results.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoint_completed.json")

    existing_df = pd.DataFrame()
    completed_combinations = set()

    if os.path.exists(results_path):
        try:
            existing_df = pd.read_csv(results_path)
            logger.info(
                f"Loaded {len(existing_df)} existing results from {results_path}"
            )

            # get completed combinations
            for _, row in existing_df.iterrows():
                combo_key = (
                    f"{row['task']}_{row['model']}_{row['method']}_{row['dimension']}"
                )
                completed_combinations.add(combo_key)

        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")

    # load checkpoint file
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                completed_combinations.update(
                    checkpoint_data.get("completed_combinations", [])
                )
                logger.info("Loaded checkpoint")
        except Exception as e:
            logger.warning(f"coud not load checkpoint: {e}")

    return existing_df, completed_combinations


def save_checkpoint(
    output_dir: str,
    all_results: List[Dict],
    completed_combinations: Set[str],
    current_task: str = None,
    current_progress: Dict = None,
):

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, "dimensionality_results.csv")
        temp_path = results_path + ".tmp"

        try:
            results_df.to_csv(temp_path, index=False)
            os.rename(temp_path, results_path)  # Atomic operation on most filesystems
            logger.info("Saved  results ")
        except Exception as e:
            logger.error(f"failed to save results: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    checkpoint_path = os.path.join(output_dir, "checkpoint_completed.json")
    temp_checkpoint_path = checkpoint_path + ".tmp"

    checkpoint_data = {
        "completed_combinations": list(completed_combinations),
        "last_update": datetime.now().isoformat(),
        "total_results": len(all_results),
        "current_task": current_task,
        "current_progress": current_progress or {},
    }

    try:
        with open(temp_checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        os.rename(temp_checkpoint_path, checkpoint_path)
        logger.info(
            f"checkpoint saved: {len(completed_combinations)} combinations completed"
        )
    except Exception as e:
        logger.error(f"failed to save checkpoint: {e}")
        if os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)


def get_combination_key(task: str, model: str, method: str, dimension: int) -> str:
    return f"{task}_{model}_{method}_{dimension}"


def run_dimensionality_analysis(
    path_to_database: str,
    path_to_labels_dir: str,
    path_to_features_dir: str,
    path_to_split_csv: str,
    tasks: List[str] = None,
    models: List[str] = None,
    dimensions: List[int] = None,
    methods: List[str] = None,
    output_dir: str = "dimensionality_results",
    skip_baseline: bool = False,
) -> pd.DataFrame:

    start_time = time.time()

    # default parameters
    if tasks is None:
        # All 14 tasks (excluding chexpert)
        tasks = [
            # Operational outcomes
            "guo_los",
            "guo_readmission",
            "guo_icu",
            # Lab values
            "lab_thrombocytopenia",
            "lab_hyperkalemia",
            "lab_hypoglycemia",
            "lab_hyponatremia",
            "lab_anemia",
            # New diagnoses
            "new_hypertension",
            "new_hyperlipidemia",
            "new_pancan",
            "new_celiac",
            "new_lupus",
            "new_acutemi",
        ]

    if models is None:
        models = ["clmbr", "clinicalbert_type3_clinicalbert_pool"]

    if dimensions is None:
        dimensions = [2, 5, 10, 25, 50, 100, 200, 400]

    if methods is None:
        methods = [
            "pca",
            "umap",
        ]

    os.makedirs(output_dir, exist_ok=True)

    existing_results_df, completed_combinations = load_existing_results(output_dir)

    total_combinations = 0
    for task in tasks:
        for model in models:
            for method in methods:
                for dimension in dimensions:
                    total_combinations += 1
            if not skip_baseline:
                total_combinations += 1

    logger.info(f"total combinations to process: {total_combinations}")

    if skip_baseline:
        logger.info("skipping baseline evaluation ")
        # mrk all baseline combinations as completed to avoid counting them
        for task in tasks:
            for model in models:
                baseline_key = get_combination_key(task, model, "none", 768)
                completed_combinations.add(baseline_key)

    if not existing_results_df.empty:
        all_results = existing_results_df.to_dict("records")
        logger.info(f"Resuming with {len(all_results)} existing results")
    else:
        all_results = []

    evaluator = kNNEvaluator()

    logger.info(
        f"Starting dimensionality analysis for {len(tasks)} tasks, {len(models)} models"
    )
    logger.info(f"Testing dimensions: {dimensions}")
    logger.info(f"Using methods: {methods}")

    for task_idx, task_name in enumerate(tasks):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing task {task_idx+1}/{len(tasks)}: {task_name}")
        logger.info(f"{'='*50}")

        try:
            # load labeled patients for this task
            labeled_patients_path = os.path.join(
                path_to_labels_dir, task_name, "labeled_patients.csv"
            )
            labeled_patients = load_labeled_patients(labeled_patients_path)

            # get labels and features
            patient_ids, label_values, label_times, feature_matrixes = (
                get_labels_and_features(labeled_patients, path_to_features_dir)
            )

            # get train/val/test splits
            train_indices, val_indices, test_indices = get_patient_splits_by_idx(
                path_to_split_csv, np.array(patient_ids)
            )

            # process labels if needed
            if task_name.startswith("lab_"):
                label_values = convert_multiclass_to_binary_labels(
                    label_values, threshold=1
                )

            for model_idx, model in enumerate(models):

                # check if model features are available
                if model not in feature_matrixes:

                    continue

                try:
                    # get features for this model
                    feature_matrix = feature_matrixes[model]

                    # Split the data
                    X_train = feature_matrix[train_indices]
                    X_val = feature_matrix[val_indices]
                    X_test = feature_matrix[test_indices]
                    y_train = label_values[train_indices]
                    y_val = label_values[val_indices]
                    y_test = label_values[test_indices]

                    # Add baseline (no dimensionality reduction) evaluation, ONLY if not skipping
                    if not skip_baseline:
                        baseline_key = get_combination_key(
                            task_name, model, "none", X_train.shape[1]
                        )
                        if baseline_key not in completed_combinations:
                            baseline_results = evaluator.evaluate(
                                X_train, X_val, X_test, y_train, y_val, y_test
                            )

                            # store baseline results
                            for metric, score in baseline_results.items():
                                all_results.append(
                                    {
                                        "task": task_name,
                                        "model": model,
                                        "method": "none",
                                        "dimension": X_train.shape[1],
                                        "metric": metric,
                                        "score": score,
                                    }
                                )

                            completed_combinations.add(baseline_key)

                            # Save checkpoint after baseline
                            save_checkpoint(
                                output_dir,
                                all_results,
                                completed_combinations,
                                task_name,
                                {"model": model, "status": "baseline_completed"},
                            )

                    for method_idx, method in enumerate(methods):

                        for dim_idx, n_dims in enumerate(dimensions):
                            combination_key = get_combination_key(
                                task_name, model, method, n_dims
                            )

                            # skip if already completed
                            if combination_key in completed_combinations:

                                continue

                            if n_dims >= X_train.shape[1]:

                                completed_combinations.add(
                                    combination_key
                                )  # Mark as completed (skipped)
                                continue

                            try:
                                # get reducer for this method and dimension
                                reducer = get_reducer(
                                    method, n_components=n_dims, random_state=42
                                )

                                # fit only on training data to avoid data leakage
                                reducer.fit(X_train)

                                # transform all splits separately to preserve boundaries
                                X_train_reduced = reducer.transform(X_train)
                                X_val_reduced = reducer.transform(X_val)
                                X_test_reduced = reducer.transform(X_test)

                                # Evaluate with kNN using proper splits
                                results = evaluator.evaluate(
                                    X_train_reduced,
                                    X_val_reduced,
                                    X_test_reduced,
                                    y_train,
                                    y_val,
                                    y_test,
                                )

                                # Store results
                                for metric, score in results.items():
                                    all_results.append(
                                        {
                                            "task": task_name,
                                            "model": model,
                                            "method": method,
                                            "dimension": n_dims,
                                            "metric": metric,
                                            "score": score,
                                        }
                                    )

                                completed_combinations.add(combination_key)
                                logger.info(
                                    f"    Results: AUROC={results['auroc']:.3f}, AUPRC={results['auprc']:.3f}"
                                )

                                # Progress update

                                progress_pct = (
                                    len(completed_combinations)
                                    / total_combinations
                                    * 100
                                )

                                # save checkpoint after every combination
                                if (
                                    len(completed_combinations) % 5 == 0
                                ):  # save every 5 combinations
                                    save_checkpoint(
                                        output_dir,
                                        all_results,
                                        completed_combinations,
                                        task_name,
                                        {
                                            "model": model,
                                            "method": method,
                                            "dimension": n_dims,
                                            "progress_pct": progress_pct,
                                        },
                                    )

                            except Exception as e:
                                logger.error(f"Error {str(e)}")
                                # still mark as completed to avoid retrying failed combinations
                                completed_combinations.add(combination_key)
                                continue

                        # save checkpoint after each method
                        save_checkpoint(
                            output_dir,
                            all_results,
                            completed_combinations,
                            task_name,
                            {
                                "model": model,
                                "method": method,
                                "status": "method_completed",
                            },
                        )

                except Exception as e:
                    logger.error(f"error: {str(e)}")
                    continue

            # save checkpoint after each task completion
            save_checkpoint(
                output_dir,
                all_results,
                completed_combinations,
                task_name,
                {"status": "task_completed"},
            )
            logger.info(f"task {task_name} completed")

        except Exception as e:
            logger.error(f"Error : {str(e)}")
            continue

    # Final save
    save_checkpoint(
        output_dir,
        all_results,
        completed_combinations,
        "ALL",
        {"status": "analysis_completed"},
    )

    # convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        # Create plots and summary
        logger.info("Creating visualizations...")
        results_dir = os.path.join(
            os.path.dirname(path_to_database), "..", "results"
        )  # Assuming EHRSHOT_ASSETS structure
        create_all_plots(results_df, results_dir, output_dir)

        # Create summary table
        summary_path = os.path.join(output_dir, "dimensionality_summary.csv")
        summary_data = []
        for task in results_df["task"].unique():
            for model in results_df["model"].unique():
                for method in results_df["method"].unique():
                    subset = results_df[
                        (results_df["task"] == task)
                        & (results_df["model"] == model)
                        & (results_df["method"] == method)
                        & (results_df["metric"] == "auroc")
                    ]
                    if not subset.empty:
                        best_score = subset["score"].max()
                        best_dim = subset[subset["score"] == best_score][
                            "dimension"
                        ].iloc[0]
                        summary_data.append(
                            {
                                "task": task,
                                "model": model,
                                "method": method,
                                "best_auroc": best_score,
                                "best_dimension": best_dim,
                            }
                        )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)

    else:
        logger.error("No results generated")

    return results_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_database", required=True, type=str, help="path to the FEMR database"
    )
    parser.add_argument(
        "--path_to_labels_dir", required=True, type=str, help="path to labels directory"
    )
    parser.add_argument(
        "--path_to_features_dir",
        required=True,
        type=str,
        help="path to features directory",
    )
    parser.add_argument(
        "--path_to_split_csv", required=True, type=str, help="path to split CSV file"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            # Operational outcomes
            "guo_los",
            "guo_readmission",
            "guo_icu",
            # Lab values
            "lab_thrombocytopenia",
            "lab_hyperkalemia",
            "lab_hypoglycemia",
            "lab_hyponatremia",
            "lab_anemia",
            # New diagnoses
            "new_hypertension",
            "new_hyperlipidemia",
            "new_pancan",
            "new_celiac",
            "new_lupus",
            "new_acutemi",
        ],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clmbr", "clinicalbert_type3_clinicalbert_pool"],
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[2, 5, 10, 25, 50, 100, 200, 400],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "umap"],
    )
    parser.add_argument(
        "--output_dir",
        default="dimensionality_results",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
    )

    args = parser.parse_args()

    # Run analysis
    results_df = run_dimensionality_analysis(
        path_to_database=args.path_to_database,
        path_to_labels_dir=args.path_to_labels_dir,
        path_to_features_dir=args.path_to_features_dir,
        path_to_split_csv=args.path_to_split_csv,
        tasks=args.tasks,
        models=args.models,
        dimensions=args.dimensions,
        methods=args.methods,
        output_dir=args.output_dir,
        skip_baseline=args.skip_baseline,
    )

    if not results_df.empty:
        logger.info(" completed ")

    else:
        logger.error("analysis failed - no results generated")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
