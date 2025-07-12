import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Any
from loguru import logger
import sys
import os

sys.path.append(os.path.dirname(__file__))

from reducers import get_reducer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import KNN_PARAMS
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_module",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation.py"),
    )
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)

    tune_hyperparams = eval_module.tune_hyperparams
    run_evaluation = eval_module.run_evaluation

except ImportError as e:
    KNN_PARAMS = {
        "n_neighbors": [1, 3, 5],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "cosine"],
    }
    tune_hyperparams = None
    run_evaluation = None

N_JOBS = int(os.environ.get("SKLEARN_N_JOBS", 1))
logger.info(f"Using {N_JOBS} parallel jobs for scikit-learn operations")


class kNNEvaluator:

    def __init__(self, random_state: int = 42):

        self.random_state = random_state

    def evaluate(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:

        if len(np.unique(y_train)) < 2:
            return {"auroc": 0.0, "auprc": 0.0}

        if run_evaluation is None:
            return {"auroc": 0.0, "auprc": 0.0}

        try:

            model, scores = run_evaluation(
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                model_head="knn",
                n_jobs=N_JOBS,
            )

            return {"auroc": scores["auroc"], "auprc": scores["auprc"]}

        except Exception as e:
            logger.error(f"Error in main pipeline evaluation: {e}")
            return {"auroc": 0.0, "auprc": 0.0}


def evaluate_all_combinations(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dimensions_to_test: List[int],
    methods: List[str] = ["pca", "umap"],
) -> pd.DataFrame:

    evaluator = kNNEvaluator()
    results = []

    baseline_results = evaluator.evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    results.append(
        {
            "method": "none",
            "n_components": X_train.shape[1],
            "auroc": baseline_results["auroc"],
            "auprc": baseline_results["auprc"],
        }
    )

    for method in methods:

        for n_dims in dimensions_to_test:
            if n_dims >= X_train.shape[1]:
                continue

            try:
                # get reducer
                reducer = get_reducer(method, n_components=n_dims, random_state=42)

                reducer.fit(X_train)

                X_train_reduced = reducer.transform(X_train)
                X_val_reduced = reducer.transform(X_val)
                X_test_reduced = reducer.transform(X_test)

                eval_results = evaluator.evaluate(
                    X_train_reduced,
                    X_val_reduced,
                    X_test_reduced,
                    y_train,
                    y_val,
                    y_test,
                )

                results.append(
                    {
                        "method": method,
                        "n_components": n_dims,
                        "auroc": eval_results["auroc"],
                        "auprc": eval_results["auprc"],
                    }
                )

            except Exception as e:
                continue

    return pd.DataFrame(results)


def evaluate_knn_with_reduced_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    method: str,
    n_components: int,
    random_state: int = 42,
) -> Dict[str, float]:
    reducer = get_reducer(method, n_components=n_components, random_state=random_state)

    reducer.fit(X_train)

    X_train_reduced = reducer.transform(X_train)
    X_val_reduced = reducer.transform(X_val)
    X_test_reduced = reducer.transform(X_test)

    evaluator = kNNEvaluator(random_state=random_state)
    return evaluator.evaluate(
        X_train_reduced, X_val_reduced, X_test_reduced, y_train, y_val, y_test
    )


def run_dimensionality_experiment(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    task_name: str,
    dimensions_to_test: List[int] = None,
    methods: List[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:

    if dimensions_to_test is None:
        dimensions_to_test = [2, 5, 10, 25, 50]

    if methods is None:
        methods = ["pca", "umap"]  # Only PCA and UMAP are supported

    evaluator = kNNEvaluator(random_state=random_state)
    results = []

    baseline_results = evaluator.evaluate(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    results.append(
        {
            "task": task_name,
            "reduction_method": "none",
            "n_components": X_train.shape[1],
            "auroc": baseline_results["auroc"],
            "auprc": baseline_results["auprc"],
        }
    )

    # test each method and dimension combination
    for method in methods:

        for n_dims in dimensions_to_test:
            if n_dims >= X_train.shape[1]:
                continue

            try:
                eval_results = evaluate_knn_with_reduced_embeddings(
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    method,
                    n_dims,
                    random_state,
                )

                results.append(
                    {
                        "task": task_name,
                        "reduction_method": method,
                        "n_components": n_dims,
                        "auroc": eval_results["auroc"],
                        "auprc": eval_results["auprc"],
                    }
                )

            except Exception as e:
                continue

    return pd.DataFrame(results)
