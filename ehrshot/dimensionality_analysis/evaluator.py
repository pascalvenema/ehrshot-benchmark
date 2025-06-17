"""
Evaluation module for dimensionality reduction analysis.
Provides k-nearest neighbors evaluation with proper hyperparameter tuning.
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

# Import kNN parameters from main utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from utils import KNN_PARAMS
except ImportError:
    # Fallback parameters if import fails
    KNN_PARAMS = {
        'n_neighbors': [1, 3, 5],
        'metric': ['euclidean', 'cosine']
    }

class kNNEvaluator:
    """
    k-Nearest Neighbors evaluator with cross-validation hyperparameter tuning.
    Uses the same hyperparameters as the main EHRSHOT pipeline.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize the kNN evaluator.
        
        Args:
            cv_folds: Number of cross-validation folds for hyperparameter tuning
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate kNN performance using cross-validation for hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with performance metrics
        """
        if len(np.unique(y)) < 2:
            logger.warning("Only one class present in labels, returning zero scores")
            return {'auroc': 0.0, 'auprc': 0.0}
        
        # Use stratified k-fold for hyperparameter selection
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        best_score = -1
        best_params = None
        
        # Grid search over hyperparameters
        for n_neighbors in KNN_PARAMS['n_neighbors']:
            for metric in KNN_PARAMS['metric']:
                scores = []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    try:
                        # Train kNN classifier
                        knn = KNeighborsClassifier(
                            n_neighbors=min(n_neighbors, len(X_train_fold) - 1),
                            metric=metric,
                            n_jobs=1
                        )
                        knn.fit(X_train_fold, y_train_fold)
                        
                        # Get predictions
                        y_pred_proba = knn.predict_proba(X_val_fold)[:, 1]
                        
                        # Calculate AUROC
                        auroc = roc_auc_score(y_val_fold, y_pred_proba)
                        scores.append(auroc)
                        
                    except Exception as e:
                        logger.warning(f"Error with n_neighbors={n_neighbors}, metric={metric}: {e}")
                        scores.append(0.0)
                
                # Average score across folds
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'n_neighbors': n_neighbors, 'metric': metric}
        
        if best_params is None:
            logger.error("No valid hyperparameters found")
            return {'auroc': 0.0, 'auprc': 0.0}
        
        # Train final model with best parameters
        knn = KNeighborsClassifier(
            n_neighbors=min(best_params['n_neighbors'], len(X) - 1),
            metric=best_params['metric'],
            n_jobs=1
        )
        
        # Use cross-validation to get final performance estimate
        final_aurocs = []
        final_auprcs = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            knn.fit(X_train_fold, y_train_fold)
            y_pred_proba = knn.predict_proba(X_test_fold)[:, 1]
            
            auroc = roc_auc_score(y_test_fold, y_pred_proba)
            auprc = average_precision_score(y_test_fold, y_pred_proba)
            
            final_aurocs.append(auroc)
            final_auprcs.append(auprc)
        
        return {
            'auroc': np.mean(final_aurocs),
            'auprc': np.mean(final_auprcs)
        }


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
                
                # Evaluate
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