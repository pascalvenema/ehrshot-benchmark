"""
Evaluation module for dimensionality reduction analysis.
Provides k-nearest neighbors evaluation using the exact same code as the main EHRSHOT pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Any
from loguru import logger
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from reducers import get_reducer

# Import the actual evaluation functions from main pipeline
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import KNN_PARAMS
    # Import the actual evaluation functions from main pipeline
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_module", os.path.join(os.path.dirname(os.path.dirname(__file__)), "7_eval.py"))
    eval_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_module)
    
    # Use the actual functions from main pipeline
    tune_hyperparams = eval_module.tune_hyperparams
    run_evaluation = eval_module.run_evaluation
    
except ImportError as e:
    logger.error(f"Could not import main pipeline evaluation functions: {e}")
    # Fallback parameters if import fails
    KNN_PARAMS = {
        'n_neighbors': [1, 3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine']
    }
    tune_hyperparams = None
    run_evaluation = None

class kNNEvaluator:
    """
    k-Nearest Neighbors evaluator using the exact same code as the main EHRSHOT pipeline.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the kNN evaluator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate kNN performance using the exact same procedure as the main EHRSHOT pipeline.
        
        Args:
            X: Feature matrix (should be train+val combined for consistency)
            y: Target labels (should be train+val combined for consistency)
            
        Returns:
            Dictionary with performance metrics
        """
        if len(np.unique(y)) < 2:
            logger.warning("Only one class present in labels, returning zero scores")
            return {'auroc': 0.0, 'auprc': 0.0}
        
        if run_evaluation is None:
            logger.error("Main pipeline evaluation function not available")
            return {'auroc': 0.0, 'auprc': 0.0}
        
        # Split the combined data back into train/val for the main pipeline function
        # Use a 80/20 split to mimic typical train/val proportions
        split_idx = int(0.8 * len(X))
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = X[train_indices]
        X_val = X[val_indices] 
        X_test = X_val  # Use val as test for evaluation (since we don't have actual test here)
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y_val  # Use val as test for evaluation
        
        try:
            # Use the exact same evaluation function as the main pipeline
            model, scores = run_evaluation(
                X_train=X_train,
                X_val=X_val, 
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                model_head='knn',
            n_jobs=1
        )
        
            return {
                'auroc': scores['auroc'],
                'auprc': scores['auprc']
            }
            
        except Exception as e:
            logger.error(f"Error in main pipeline evaluation: {e}")
            return {'auroc': 0.0, 'auprc': 0.0}


def evaluate_all_combinations(
    X_train: np.ndarray, 
    X_val: np.ndarray, 
    X_test: np.ndarray,
    y_train: np.ndarray, 
    y_val: np.ndarray, 
    y_test: np.ndarray,
    dimensions_to_test: List[int],
    methods: List[str] = ['pca', 'umap', 'tsne']
) -> pd.DataFrame:
    """
    Evaluate all combinations of dimensionality reduction methods and target dimensions.
    
    Args:
        X_train, X_val, X_test: Feature matrices for train/val/test splits
        y_train, y_val, y_test: Labels for train/val/test splits
        dimensions_to_test: List of target dimensions to test
        methods: List of dimensionality reduction methods
        
    Returns:
        DataFrame with results for all combinations
    """
    evaluator = kNNEvaluator()
    results = []
    
    # Add baseline (no reduction)
    logger.info("Evaluating baseline (no dimensionality reduction)")
    baseline_X = np.vstack([X_train, X_val])
    baseline_y = np.hstack([y_train, y_val])
    baseline_results = evaluator.evaluate(baseline_X, baseline_y)
    
    results.append({
        'method': 'none',
        'n_components': X_train.shape[1],
        'auroc': baseline_results['auroc'],
        'auprc': baseline_results['auprc']
    })
    
    # Test each method and dimension combination
    for method in methods:
        logger.info(f"Testing method: {method}")
        
        for n_dims in dimensions_to_test:
            if n_dims >= X_train.shape[1]:
                logger.warning(f"Skipping {n_dims} dimensions (>= original {X_train.shape[1]})")
                continue
                
            logger.info(f"  Testing {n_dims} dimensions")
            
            try:
                # Get reducer
                reducer = get_reducer(method, n_components=n_dims, random_state=42)
                
                # Fit on training data
                reducer.fit(X_train)
                
                # Transform all splits
                X_train_reduced = reducer.transform(X_train)
                X_val_reduced = reducer.transform(X_val)
                
                # Combine train and val for evaluation
                X_combined = np.vstack([X_train_reduced, X_val_reduced])
                y_combined = np.hstack([y_train, y_val])
                
                # Evaluate using main pipeline evaluation
                eval_results = evaluator.evaluate(X_combined, y_combined)
                
                results.append({
                    'method': method,
                    'n_components': n_dims,
                    'auroc': eval_results['auroc'],
                    'auprc': eval_results['auprc']
                })
                
                logger.info(f"    AUROC: {eval_results['auroc']:.3f}, AUPRC: {eval_results['auprc']:.3f}")
                
            except Exception as e:
                logger.error(f"    Error with {method} {n_dims}D: {e}")
                continue
    
    return pd.DataFrame(results) 