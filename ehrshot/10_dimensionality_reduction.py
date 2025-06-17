"""
Dimensionality Reduction Analysis for EHRSHOT Benchmark

This script analyzes the impact of dimensionality reduction on ClinicalBERT embeddings,
with a particular focus on k-NN performance improvements due to addressing the curse 
of dimensionality.

Usage:
    python3 10_dimensionality_reduction.py \
        --path_to_database '../EHRSHOT_ASSETS/femr/extract' \
        --path_to_labels_dir '../EHRSHOT_ASSETS/benchmark' \
        --path_to_features_dir '../EHRSHOT_ASSETS/features' \
        --path_to_split_csv '../EHRSHOT_ASSETS/splits/person_id_map.csv' \
        --path_to_output_dir '../EHRSHOT_ASSETS/dimensionality_reduction' \
        --labeling_functions 'guo_icu,guo_los,lab_anemia' \
        --reduction_methods 'pca,truncated_svd' \
        --dimensions '32,64,128,256' \
        --num_threads 10
"""

import argparse
import os
from typing import List
import pandas as pd
from loguru import logger
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(__file__))
from dimensionality_reduction import DimensionalityReductionEvaluator
from dimensionality_reduction.utils import (
    load_features, 
    plot_explained_variance,
    create_reduction_report,
    visualize_embeddings_2d
)
from utils import LABELING_FUNCTION_2_PAPER_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dimensionality Reduction Analysis for EHRSHOT")
    
    parser.add_argument("--path_to_database", required=True, type=str, 
                       help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, 
                       help="Path to directory containing labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, 
                       help="Path to directory containing features")
    parser.add_argument("--path_to_split_csv", required=True, type=str, 
                       help="Path to CSV file with patient splits")
    parser.add_argument("--path_to_output_dir", required=True, type=str, 
                       help="Path to output directory")
    
    parser.add_argument("--labeling_functions", type=str, 
                       default="guo_icu,guo_los,guo_readmission,lab_anemia,lab_hypoglycemia,new_hypertension",
                       help="Comma-separated list of labeling functions to test")
    parser.add_argument("--reduction_methods", type=str, 
                       default="pca,truncated_svd,factor_analysis",
                       help="Comma-separated list of reduction methods")
    parser.add_argument("--dimensions", type=str, 
                       default="16,32,64,128,256,384",
                       help="Comma-separated list of dimensions to test")
    parser.add_argument("--k_shots", type=str, 
                       default="-1",
                       help="Comma-separated list of k-shot values (-1 for full data)")
    
    parser.add_argument("--model_config", type=str, 
                       default="clinicalbert_type3_clinicalbert_pool",
                       help="Model configuration to analyze")
    parser.add_argument("--prediction_heads", type=str, 
                       default="knn,lr_lbfgs,rf",
                       help="Comma-separated list of prediction heads to test")
    
    parser.add_argument("--num_threads", type=int, default=1, 
                       help="Number of threads to use")
    parser.add_argument("--create_visualizations", action='store_true', 
                       help="Create 2D visualizations of embeddings")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, 
                       help="Force refresh all outputs")
    
    return parser.parse_args()


def analyze_original_embeddings(path_to_features_dir: str, 
                               model_config: str,
                               path_to_output_dir: str) -> None:
    """Analyze the original high-dimensional embeddings.
    
    Args:
        path_to_features_dir: Path to features directory
        model_config: Model configuration to analyze
        path_to_output_dir: Output directory for analysis
    """
    logger.info(f"Analyzing original embeddings for {model_config}")
    
    # Load original features
    features_path = os.path.join(path_to_features_dir, f'{model_config}_features.pkl')
    feature_matrix, patient_ids, label_times = load_features(features_path)
    
    logger.info(f"Original embedding statistics:")
    logger.info(f"  Shape: {feature_matrix.shape}")
    logger.info(f"  Mean: {feature_matrix.mean():.6f}")
    logger.info(f"  Std: {feature_matrix.std():.6f}")
    logger.info(f"  Min: {feature_matrix.min():.6f}")
    logger.info(f"  Max: {feature_matrix.max():.6f}")
    
    # Create explained variance analysis
    analysis_dir = os.path.join(path_to_output_dir, 'analysis')
    plot_explained_variance(feature_matrix, analysis_dir, max_components=300)
    
    logger.info("Original embedding analysis complete")


def create_visualizations(path_to_features_dir: str,
                         model_config: str, 
                         path_to_output_dir: str) -> None:
    """Create 2D visualizations of embeddings.
    
    Args:
        path_to_features_dir: Path to features directory
        model_config: Model configuration
        path_to_output_dir: Output directory
    """
    logger.info("Creating embedding visualizations")
    
    # Load original features
    features_path = os.path.join(path_to_features_dir, f'{model_config}_features.pkl')
    feature_matrix, patient_ids, label_times = load_features(features_path)
    
    # Subsample for visualization (t-SNE is expensive)
    n_samples = min(2000, feature_matrix.shape[0])
    indices = np.random.choice(feature_matrix.shape[0], n_samples, replace=False)
    X_subset = feature_matrix[indices]
    
    viz_dir = os.path.join(path_to_output_dir, 'visualizations')
    
    # PCA visualization
    visualize_embeddings_2d(
        X_subset, 
        method='pca',
        path_to_output_file=os.path.join(viz_dir, 'embeddings_2d_pca.png'),
        title=f'{model_config} - PCA Visualization'
    )
    
    # t-SNE visualization 
    visualize_embeddings_2d(
        X_subset,
        method='tsne', 
        path_to_output_file=os.path.join(viz_dir, 'embeddings_2d_tsne.png'),
        title=f'{model_config} - t-SNE Visualization'
    )
    
    logger.info("Visualization creation complete")


def main():
    args = parse_args()
    
    # Parse comma-separated arguments
    labeling_functions = [f.strip() for f in args.labeling_functions.split(',')]
    reduction_methods = [m.strip() for m in args.reduction_methods.split(',')]
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]
    k_shots = [int(k.strip()) for k in args.k_shots.split(',')]
    prediction_heads = [h.strip() for h in args.prediction_heads.split(',')]
    
    logger.info("Starting Dimensionality Reduction Analysis")
    logger.info(f"Labeling functions: {labeling_functions}")
    logger.info(f"Reduction methods: {reduction_methods}")
    logger.info(f"Dimensions to test: {dimensions}")
    logger.info(f"K-shot values: {k_shots}")
    logger.info(f"Prediction heads: {prediction_heads}")
    
    # Create output directory
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Step 1: Analyze original embeddings
    logger.info("=" * 60)
    logger.info("STEP 1: ANALYZING ORIGINAL EMBEDDINGS")
    logger.info("=" * 60)
    
    analyze_original_embeddings(
        args.path_to_features_dir, 
        args.model_config,
        args.path_to_output_dir
    )
    
    # Step 2: Create visualizations (if requested)
    if args.create_visualizations:
        logger.info("=" * 60)
        logger.info("STEP 2: CREATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        create_visualizations(
            args.path_to_features_dir,
            args.model_config,
            args.path_to_output_dir
        )
    
    # Step 3: Initialize evaluator
    logger.info("=" * 60)
    logger.info("STEP 3: INITIALIZING EVALUATOR")
    logger.info("=" * 60)
    
    evaluator = DimensionalityReductionEvaluator(
        path_to_database=args.path_to_database,
        path_to_labels_dir=args.path_to_labels_dir,
        path_to_features_dir=args.path_to_features_dir,
        path_to_split_csv=args.path_to_split_csv,
        path_to_output_dir=args.path_to_output_dir,
        num_threads=args.num_threads
    )
    
    # Override default configurations with command line arguments
    evaluator.model_configs = [args.model_config]
    evaluator.prediction_heads = prediction_heads
    evaluator.reduction_methods = reduction_methods
    evaluator.dimensions_to_test = dimensions
    
    # Step 4: Run dimensionality reduction experiments
    logger.info("=" * 60)
    logger.info("STEP 4: RUNNING DIMENSIONALITY REDUCTION EXPERIMENTS")
    logger.info("=" * 60)
    
    results_df = evaluator.run_full_experiment(
        labeling_functions=labeling_functions,
        k_shots=k_shots
    )
    
    logger.info(f"Experiment complete! Generated {len(results_df)} result records")
    
    # Step 5: Create comprehensive report
    logger.info("=" * 60)
    logger.info("STEP 5: CREATING ANALYSIS REPORT")
    logger.info("=" * 60)
    
    report_dir = os.path.join(args.path_to_output_dir, 'reports')
    create_reduction_report(results_df, report_dir)
    
    # Step 6: Print summary statistics
    logger.info("=" * 60)
    logger.info("STEP 6: SUMMARY STATISTICS")
    logger.info("=" * 60)
    
    print_summary_statistics(results_df)
    
    logger.success("Dimensionality Reduction Analysis Complete!")
    logger.info(f"All outputs saved to: {args.path_to_output_dir}")


def print_summary_statistics(results_df: pd.DataFrame) -> None:
    """Print summary statistics of the experiment results."""
    
    if results_df.empty:
        logger.warning("No results to summarize")
        return
    
    logger.info("SUMMARY STATISTICS")
    print("-" * 80)
    
    # Overall statistics
    print(f"Total result records: {len(results_df)}")
    print(f"Tasks evaluated: {len(results_df['labeling_function'].unique())}")
    print(f"Reduction methods tested: {len(results_df['reduction_method'].unique())}")
    print(f"Dimensions tested: {sorted(results_df['n_components'].unique())}")
    print(f"Prediction heads tested: {list(results_df['head'].unique())}")
    
    print("\n" + "="*50)
    print("BEST PERFORMANCE BY METRIC AND HEAD")
    print("="*50)
    
    # Focus on k-NN results (main interest)
    knn_results = results_df[results_df['head'] == 'knn'].copy()
    
    if not knn_results.empty:
        print("\n🔍 k-NN PERFORMANCE HIGHLIGHTS:")
        print("-" * 40)
        
        for score in ['auroc', 'auprc']:
            score_data = knn_results[knn_results['score'] == score]
            if not score_data.empty:
                best_config = score_data.loc[score_data['value'].idxmax()]
                print(f"\nBest k-NN {score.upper()}: {best_config['value']:.4f}")
                print(f"  Method: {best_config['reduction_method']}")
                print(f"  Dimensions: {best_config['n_components']}")
                print(f"  Task: {best_config['labeling_function']}")
                
                # Compare to full dimensionality performance if available
                full_dim_mask = (score_data['n_components'] == score_data['n_components'].max())
                if full_dim_mask.any():
                    full_dim_perf = score_data[full_dim_mask]['value'].mean()
                    improvement = best_config['value'] - full_dim_perf
                    print(f"  Improvement over full dims: {improvement:+.4f}")
    
    # Performance by dimensionality
    print("\n" + "="*50)
    print("AVERAGE PERFORMANCE BY DIMENSIONALITY (k-NN)")
    print("="*50)
    
    if not knn_results.empty:
        for score in ['auroc', 'auprc']:
            score_data = knn_results[knn_results['score'] == score]
            if not score_data.empty:
                print(f"\n{score.upper()} by Dimensions:")
                perf_by_dim = score_data.groupby('n_components')['value'].agg(['mean', 'std', 'count'])
                for dim, stats in perf_by_dim.iterrows():
                    print(f"  {dim:3d}D: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
    
    # Method comparison
    print("\n" + "="*50)
    print("AVERAGE PERFORMANCE BY REDUCTION METHOD (k-NN)")
    print("="*50)
    
    if not knn_results.empty:
        for score in ['auroc', 'auprc']:
            score_data = knn_results[knn_results['score'] == score]
            if not score_data.empty:
                print(f"\n{score.upper()} by Method:")
                perf_by_method = score_data.groupby('reduction_method')['value'].agg(['mean', 'std', 'count'])
                for method, stats in perf_by_method.iterrows():
                    print(f"  {method:15s}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues in other functions
    main() 