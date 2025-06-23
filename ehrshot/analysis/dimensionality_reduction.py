#!/usr/bin/env python3
"""
Dimensionality Reduction Analysis for EHRSHOT Benchmark

This script evaluates the impact of dimensionality reduction on kNN classification
performance using CLMBR and ClinicalBERT type3 clinicalbert_pool embeddings.
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from loguru import logger
import sys

# Add the dimensionality analysis module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dimensionality_analysis'))

from reducers import get_reducer
from evaluator import kNNEvaluator
from plotting import create_all_plots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_labels_and_features, get_patient_splits_by_idx, convert_multiclass_to_binary_labels
from femr.labelers import load_labeled_patients


def run_dimensionality_analysis(
    path_to_database: str,
    path_to_labels_dir: str,
    path_to_features_dir: str,
    path_to_split_csv: str,
    tasks: List[str] = None,
    models: List[str] = None,
    dimensions: List[int] = None,
    methods: List[str] = None,
    output_dir: str = "dimensionality_results"
) -> pd.DataFrame:
    """
    Run comprehensive dimensionality reduction analysis.
    
    Args:
        path_to_database: Path to the FEMR database
        path_to_labels_dir: Path to labels directory
        path_to_features_dir: Path to features directory
        path_to_split_csv: Path to split CSV file
        tasks: List of task names to analyze
        models: List of model names (clmbr, clinicalbert_type3_clinicalbert_pool)
        dimensions: List of target dimensions to test
        methods: List of dimensionality reduction methods
        output_dir: Directory to save results
        
    Returns:
        DataFrame with all results
    """
    # Default parameters
    if tasks is None:
        # All 14 tasks (excluding chexpert)
        tasks = [
            # Operational outcomes
            'guo_los', 'guo_readmission', 'guo_icu',
            # Lab values  
            'lab_thrombocytopenia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia',
            # New diagnoses
            'new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 'new_lupus', 'new_acutemi'
        ]
    
    if models is None:
        models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    
    if dimensions is None:
        dimensions = [2, 5, 10, 25, 50, 100, 200, 400]
    
    if methods is None:
        methods = ['pca', 'umap']  # Skip t-SNE for now due to transform issues
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    evaluator = kNNEvaluator()
    
    # Store all results
    all_results = []
    
    logger.info(f"Starting dimensionality analysis for {len(tasks)} tasks, {len(models)} models")
    logger.info(f"Testing dimensions: {dimensions}")
    logger.info(f"Using methods: {methods}")
    
    for task_name in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing task: {task_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Load labeled patients for this task
            labeled_patients_path = os.path.join(path_to_labels_dir, task_name, 'labeled_patients.csv')
            labeled_patients = load_labeled_patients(labeled_patients_path)
            logger.info(f"Loaded labeled patients for {task_name}")
            
            # Get labels and features
            patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(
                labeled_patients, path_to_features_dir
            )
            
            # Get train/val/test splits
            train_indices, val_indices, test_indices = get_patient_splits_by_idx(
                path_to_split_csv, np.array(patient_ids)
            )
            
            # Process labels if needed
            if task_name.startswith('lab_'):
                label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)
            
            for model in models:
                logger.info(f"\nProcessing model: {model}")
                
                # Check if model features are available
                if model not in feature_matrixes:
                    logger.warning(f"Features not available for model {model}, skipping")
                    continue
                
                try:
                    # Get features for this model
                    feature_matrix = feature_matrixes[model]
                    
                    # Split the data
                    X_train = feature_matrix[train_indices]
                    X_val = feature_matrix[val_indices] 
                    X_test = feature_matrix[test_indices]
                    y_train = label_values[train_indices]
                    y_val = label_values[val_indices]
                    y_test = label_values[test_indices]
                    
                    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                    logger.info(f"Label distribution - Train: {np.mean(y_train):.3f}, Val: {np.mean(y_val):.3f}, Test: {np.mean(y_test):.3f}")
                    
                    # Combine train and val for cross-validation
                    X_combined = np.vstack([X_train, X_val])
                    y_combined = np.hstack([y_train, y_val])
                    
                    for method in methods:
                        logger.info(f"\n  Testing method: {method}")
                        
                        for n_dims in dimensions:
                            if n_dims >= X_combined.shape[1]:
                                logger.warning(f"Skipping {n_dims} dimensions (>= original {X_combined.shape[1]})")
                                continue
                                
                            logger.info(f"    Reducing to {n_dims} dimensions...")
                            
                            try:
                                # Get reducer for this method and dimension
                                reducer = get_reducer(method, n_components=n_dims, random_state=42)
                                
                                # Apply dimensionality reduction
                                X_reduced = reducer.fit_transform(X_combined)
                                
                                logger.info(f"    Reduced shape: {X_reduced.shape}")
                                
                                # Evaluate with kNN
                                results = evaluator.evaluate(X_reduced, y_combined)
                                
                                # Store results
                                for metric, score in results.items():
                                    all_results.append({
                                        'task': task_name,
                                        'model': model,
                                        'method': method,
                                        'dimension': n_dims,
                                        'metric': metric,
                                        'score': score
                                    })
                                
                                logger.info(f"    Results: AUROC={results['auroc']:.3f}, AUPRC={results['auprc']:.3f}")
                                
                            except Exception as e:
                                logger.error(f"    Error with {method} {n_dims}D: {str(e)}")
                                continue
                
                except Exception as e:
                    logger.error(f"Error processing model {model} for task {task_name}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        # Save raw results
        results_path = os.path.join(output_dir, 'dimensionality_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Create plots and summary
        logger.info("Creating visualizations...")
        results_dir = os.path.join(os.path.dirname(path_to_database), '..', 'results')  # Assuming EHRSHOT_ASSETS structure
        create_all_plots(results_df, results_dir, output_dir)
    else:
        logger.error("No results generated!")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run dimensionality reduction analysis on EHRSHOT embeddings"
    )
    parser.add_argument(
        "--path_to_database", 
        required=True, 
        type=str,
        help="Path to the FEMR database"
    )
    parser.add_argument(
        "--path_to_labels_dir", 
        required=True, 
        type=str,
        help="Path to labels directory"
    )
    parser.add_argument(
        "--path_to_features_dir", 
        required=True, 
        type=str,
        help="Path to features directory"
    )
    parser.add_argument(
        "--path_to_split_csv", 
        required=True, 
        type=str,
        help="Path to split CSV file"
    )
    parser.add_argument(
        "--tasks", 
        nargs='+', 
        default=[
            # Operational outcomes
            'guo_los', 'guo_readmission', 'guo_icu',
            # Lab values  
            'lab_thrombocytopenia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia',
            # New diagnoses
            'new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 'new_lupus', 'new_acutemi'
        ],
        help="List of tasks to analyze"
    )
    parser.add_argument(
        "--models", 
        nargs='+', 
        default=['clmbr', 'clinicalbert_type3_clinicalbert_pool'],
        help="List of models to analyze"
    )
    parser.add_argument(
        "--dimensions", 
        nargs='+', 
        type=int,
        default=[2, 5, 10, 25, 50, 100, 200, 400],
        help="List of target dimensions to test"
    )
    parser.add_argument(
        "--methods", 
        nargs='+', 
        default=['pca', 'umap'],
        help="List of dimensionality reduction methods"
    )
    parser.add_argument(
        "--output_dir", 
        default="dimensionality_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add(
        os.path.join(args.output_dir, "dimensionality_analysis.log"),
        rotation="10 MB",
        level="INFO"
    )
    
    logger.info("Starting EHRSHOT dimensionality reduction analysis")
    logger.info(f"Arguments: {vars(args)}")
    
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
        output_dir=args.output_dir
    )
    
    if not results_df.empty:
        logger.info("Analysis completed successfully!")
        logger.info(f"Generated {len(results_df)} result entries")
        
        # Print summary
        logger.info("\nSummary by model and method:")
        for model in results_df['model'].unique():
            for method in results_df['method'].unique():
                subset = results_df[
                    (results_df['model'] == model) & 
                    (results_df['method'] == method) &
                    (results_df['metric'] == 'auroc')
                ]
                if not subset.empty:
                    mean_auroc = subset['score'].mean()
                    logger.info(f"  {model} + {method}: {mean_auroc:.3f} AUROC (avg)")
    else:
        logger.error("Analysis failed - no results generated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 