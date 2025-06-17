"""
Evaluator for dimensionality reduction experiments.

This module handles the evaluation of different dimensionality reduction techniques
on the EHRSHOT benchmark, with a focus on k-NN performance improvements.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import lightgbm as lgb

from .reducers import create_reducer
from .utils import load_features, save_reduced_features
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import (
    get_labels_and_features, 
    get_patient_splits_by_idx,
    LR_PARAMS,
    RF_PARAMS, 
    KNN_PARAMS,
    XGB_PARAMS
)


class DimensionalityReductionEvaluator:
    """Evaluator for dimensionality reduction experiments."""
    
    def __init__(self, 
                 path_to_database: str,
                 path_to_labels_dir: str, 
                 path_to_features_dir: str,
                 path_to_split_csv: str,
                 path_to_output_dir: str,
                 num_threads: int = 1):
        """Initialize the evaluator.
        
        Args:
            path_to_database: Path to FEMR database
            path_to_labels_dir: Path to labels directory
            path_to_features_dir: Path to features directory
            path_to_split_csv: Path to split CSV file
            path_to_output_dir: Path to output directory
            num_threads: Number of threads for parallel processing
        """
        self.path_to_database = path_to_database
        self.path_to_labels_dir = path_to_labels_dir
        self.path_to_features_dir = path_to_features_dir
        self.path_to_split_csv = path_to_split_csv
        self.path_to_output_dir = path_to_output_dir
        self.num_threads = num_threads
        
        # Create output directory
        os.makedirs(path_to_output_dir, exist_ok=True)
        
        # Model configurations to test
        self.model_configs = [
            'clinicalbert_type3_clinicalbert_pool'
        ]
        
        # Prediction heads to test (focusing on k-NN)
        self.prediction_heads = ['knn', 'lr_lbfgs', 'rf']
        
        # Dimensionality reduction methods
        self.reduction_methods = ['pca', 'truncated_svd', 'factor_analysis']
        
        # Dimensions to test
        self.dimensions_to_test = [16, 32, 64, 128, 256, 384]
    
    def generate_reduced_features(self, 
                                 model_config: str,
                                 reduction_method: str,
                                 n_components: int,
                                 labeled_patients = None) -> str:
        """Generate reduced features for a given configuration.
        
        Args:
            model_config: Model configuration (e.g., 'clinicalbert_type3_clinicalbert_pool')
            reduction_method: Reduction method ('pca', 'truncated_svd', etc.)
            n_components: Number of components to keep
            labeled_patients: Labeled patients object (optional)
            
        Returns:
            Path to the saved reduced features file
        """
        logger.info(f"Generating reduced features: {model_config}, {reduction_method}, {n_components}")
        
        # Load original features
        original_features_path = os.path.join(self.path_to_features_dir, f'{model_config}_features.pkl')
        feature_matrix, patient_ids, label_times = load_features(original_features_path)
        
        # Create reducer
        reducer = create_reducer(reduction_method, n_components, random_state=42)
        
        # Fit reducer (for unsupervised methods, we don't need labels)
        reduced_matrix = reducer.fit_transform(feature_matrix)
        
        # Save reduced features
        output_filename = f'{model_config}_{reduction_method}_{n_components}d_features.pkl'
        output_path = os.path.join(self.path_to_output_dir, 'reduced_features', output_filename)
        
        reducer_info = {
            'method': reduction_method,
            'n_components': n_components,
            'original_dimensions': feature_matrix.shape[1],
            'explained_variance_ratio': reducer.get_explained_variance_ratio(),
            'feature_importance': reducer.get_feature_importance()
        }
        
        save_reduced_features(reduced_matrix, patient_ids, label_times, output_path, reducer_info)
        
        return output_path
    
    def evaluate_reduced_features(self,
                                 path_to_reduced_features: str,
                                 labeling_function: str,
                                 k_shot: int = -1) -> List[Dict[str, Any]]:
        """Evaluate reduced features on a specific task.
        
        Args:
            path_to_reduced_features: Path to reduced features file
            labeling_function: Labeling function to evaluate
            k_shot: Number of shots (-1 for full data)
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Evaluating reduced features on {labeling_function}")
        
        # Load labeled patients
        from femr.labelers import load_labeled_patients
        path_to_labels_file = os.path.join(self.path_to_labels_dir, f'{labeling_function}/labeled_patients.csv')
        labeled_patients = load_labeled_patients(path_to_labels_file)
        
        # Load features (original function expects features in a dict format)
        feature_matrixes = {}
        model_name = os.path.basename(path_to_reduced_features).replace('_features.pkl', '')
        
        with open(path_to_reduced_features, 'rb') as f:
            data = pickle.load(f)
            feature_matrix = data['data_matrix']
            feature_patient_ids = data['patient_ids']
            feature_times = data['labeling_time']
        
        # Get labels and features
        label_data = get_labels_and_features(labeled_patients, None)
        if len(label_data) == 3:
            label_patient_ids, label_values, label_times = label_data
        else:
            label_patient_ids, label_values, label_times, _ = label_data
        
        # Get splits
        train_idxs, val_idxs, test_idxs = get_patient_splits_by_idx(
            self.path_to_split_csv, label_patient_ids
        )
        
        # Align features with labels (simplified version)
        feature_patient_ids = feature_patient_ids.astype(label_patient_ids.dtype)
        feature_times = feature_times.astype(label_times.dtype)
        
        # Create alignment mask
        alignment_mask = np.zeros(len(label_patient_ids), dtype=bool)
        feature_indices = np.zeros(len(label_patient_ids), dtype=int)
        
        for i, (pid, time) in enumerate(zip(label_patient_ids, label_times)):
            match_mask = (feature_patient_ids == pid) & (feature_times == time)
            if np.any(match_mask):
                alignment_mask[i] = True
                feature_indices[i] = np.where(match_mask)[0][0]
        
        # Filter aligned data
        aligned_features = feature_matrix[feature_indices[alignment_mask]]
        aligned_labels = label_values[alignment_mask]
        
        # Convert train/val/test indices to numpy arrays
        train_idxs = np.array(train_idxs)
        val_idxs = np.array(val_idxs)
        test_idxs = np.array(test_idxs)
        
        # Apply alignment mask to splits
        aligned_train_idxs = train_idxs[alignment_mask[train_idxs]]
        aligned_val_idxs = val_idxs[alignment_mask[val_idxs]]
        aligned_test_idxs = test_idxs[alignment_mask[test_idxs]]
        
        # Apply k-shot sampling if needed
        if k_shot > 0:
            # Simple k-shot sampling
            pos_train_idxs = aligned_train_idxs[aligned_labels[aligned_train_idxs] == 1]
            neg_train_idxs = aligned_train_idxs[aligned_labels[aligned_train_idxs] == 0]
            
            np.random.seed(42)
            selected_pos = np.random.choice(pos_train_idxs, min(k_shot, len(pos_train_idxs)), replace=False)
            selected_neg = np.random.choice(neg_train_idxs, min(k_shot, len(neg_train_idxs)), replace=False)
            
            aligned_train_idxs = np.concatenate([selected_pos, selected_neg])
        
        # Prepare data splits
        X_train = aligned_features[aligned_train_idxs]
        X_val = aligned_features[aligned_val_idxs]
        X_test = aligned_features[aligned_test_idxs]
        y_train = aligned_labels[aligned_train_idxs]
        y_val = aligned_labels[aligned_val_idxs]
        y_test = aligned_labels[aligned_test_idxs]
        
        results = []
        
        # Evaluate each prediction head
        for head in self.prediction_heads:
            try:
                model, scores = self._train_and_evaluate_model(
                    X_train, X_val, X_test, y_train, y_val, y_test, head
                )
                
                for score_name, score_value in scores.items():
                    results.append({
                        'labeling_function': labeling_function,
                        'model': model_name,
                        'head': head,
                        'k': k_shot,
                        'score': score_name,
                        'value': score_value,
                        'n_train': len(X_train),
                        'n_val': len(X_val),
                        'n_test': len(X_test)
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating {head} on {labeling_function}: {e}")
                continue
        
        return results
    
    def _train_and_evaluate_model(self,
                                 X_train: np.ndarray,
                                 X_val: np.ndarray,
                                 X_test: np.ndarray,
                                 y_train: np.ndarray,
                                 y_val: np.ndarray,
                                 y_test: np.ndarray,
                                 model_head: str) -> Tuple[Any, Dict[str, float]]:
        """Train and evaluate a single model.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Label arrays
            model_head: Model head type
            
        Returns:
            Tuple of (model, scores_dict)
        """
        logger.debug(f"Training {model_head} model")
        
        # Handle different model types
        if model_head == 'knn':
            # k-Nearest Neighbors
            scaler = MaxAbsScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Adjust k based on training set size
            n_samples = X_train.shape[0]
            adjusted_knn_params = KNN_PARAMS.copy()
            adjusted_knn_params['n_neighbors'] = [k for k in KNN_PARAMS['n_neighbors'] if k <= n_samples]
            if not adjusted_knn_params['n_neighbors']:
                adjusted_knn_params['n_neighbors'] = [1]
            
            model = KNeighborsClassifier()
            model = self._tune_hyperparams(X_train_scaled, X_val_scaled, y_train, y_val, 
                                         model, adjusted_knn_params)
            
            X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
            
        elif model_head == 'lr_lbfgs':
            # Logistic Regression
            scaler = MaxAbsScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000)
            model = self._tune_hyperparams(X_train_scaled, X_val_scaled, y_train, y_val, 
                                         model, LR_PARAMS)
            
            X_train, X_val, X_test = X_train_scaled, X_val_scaled, X_test_scaled
            
        elif model_head == 'rf':
            # Random Forest
            model = RandomForestClassifier(random_state=42)
            rf_params = RF_PARAMS.copy()
            rf_params['min_samples_leaf'] = [1]
            rf_params['min_samples_split'] = [2]
            model = self._tune_hyperparams(X_train, X_val, y_train, y_val, model, rf_params)
            
        elif model_head == 'gbm':
            # LightGBM
            model = lgb.LGBMClassifier(random_state=42)
            gbm_params = XGB_PARAMS.copy()
            gbm_params['min_child_samples'] = [1]
            model = self._tune_hyperparams(X_train, X_val, y_train, y_val, model, gbm_params)
            
        else:
            raise ValueError(f"Unknown model head: {model_head}")
        
        # Get predictions
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        scores = {
            'auroc': roc_auc_score(y_test, y_test_proba),
            'auprc': average_precision_score(y_test, y_test_proba),
        }
        
        return model, scores
    
    def _tune_hyperparams(self, X_train, X_val, y_train, y_val, model, param_grid):
        """Tune hyperparameters using GridSearchCV."""
        from scipy.sparse import vstack, issparse
        
        # Concatenate train/val sets
        X = vstack([X_train, X_val]) if issparse(X_train) else np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)
        
        # Create predefined split
        test_fold = -np.ones(X.shape[0])
        test_fold[X_train.shape[0]:] = 0
        
        # Tune
        clf = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=1, 
                          cv=PredefinedSplit(test_fold), refit=False)
        clf.fit(X, y)
        
        # Refit on training data only
        best_model = model.__class__(**clf.best_params_)
        best_model.fit(X_train, y_train)
        
        return best_model
    
    def run_full_experiment(self, 
                           labeling_functions: List[str],
                           k_shots: List[int] = [-1]) -> pd.DataFrame:
        """Run the full dimensionality reduction experiment.
        
        Args:
            labeling_functions: List of labeling functions to evaluate
            k_shots: List of k-shot values to test
            
        Returns:
            DataFrame with all results
        """
        logger.info("Starting full dimensionality reduction experiment")
        
        all_results = []
        
        # For each model configuration
        for model_config in tqdm(self.model_configs, desc="Model configs"):
            
            # For each reduction method
            for reduction_method in tqdm(self.reduction_methods, desc="Reduction methods", leave=False):
                
                # For each dimension
                for n_components in tqdm(self.dimensions_to_test, desc="Dimensions", leave=False):
                    
                    try:
                        # Generate reduced features
                        reduced_features_path = self.generate_reduced_features(
                            model_config, reduction_method, n_components
                        )
                        
                        # For each labeling function
                        for labeling_function in tqdm(labeling_functions, desc="Tasks", leave=False):
                            
                            # For each k-shot value
                            for k_shot in k_shots:
                                
                                try:
                                    results = self.evaluate_reduced_features(
                                        reduced_features_path, labeling_function, k_shot
                                    )
                                    
                                    # Add metadata
                                    for result in results:
                                        result['reduction_method'] = reduction_method
                                        result['n_components'] = n_components
                                        result['original_model'] = model_config
                                    
                                    all_results.extend(results)
                                    
                                except Exception as e:
                                    logger.error(f"Error in task {labeling_function}, k={k_shot}: {e}")
                                    continue
                    
                    except Exception as e:
                        logger.error(f"Error generating features for {model_config}, {reduction_method}, {n_components}: {e}")
                        continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_path = os.path.join(self.path_to_output_dir, 'dimensionality_reduction_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        return results_df 