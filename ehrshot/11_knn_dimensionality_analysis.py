"""
k-NN Dimensionality Analysis

This script provides focused analysis of how dimensionality reduction affects
k-NN performance on clinical prediction tasks, addressing the curse of dimensionality.

This is a streamlined version that focuses specifically on answering RQ2.2:
"How does the dimensionality of the patient embeddings affect the relative 
performance of different prediction heads?"
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from loguru import logger
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(__file__))
from dimensionality_reduction.utils import load_features
from dimensionality_reduction.reducers import create_reducer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


def quick_knn_analysis(feature_matrix: np.ndarray, 
                      labels: np.ndarray,
                      dimensions_to_test: List[int] = [16, 32, 64, 128, 256, 384],
                      reduction_methods: List[str] = ['pca', 'truncated_svd']) -> pd.DataFrame:
    """Run a quick k-NN analysis across different dimensionalities.
    
    Args:
        feature_matrix: Original high-dimensional features
        labels: Binary labels
        dimensions_to_test: List of dimensions to test
        reduction_methods: List of reduction methods
        
    Returns:
        DataFrame with results
    """
    logger.info("Running quick k-NN dimensionality analysis")
    
    results = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples") 
    logger.info(f"Class balance - Train: {np.mean(y_train):.3f}, Test: {np.mean(y_test):.3f}")
    
    # Test original high-dimensional features
    logger.info("Testing original 768D features...")
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try different k values for k-NN
    k_values = [1, 3, 5, 7, 11, 15]
    k_values = [k for k in k_values if k <= len(X_train_scaled)]
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]
        
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        
        results.append({
            'method': 'original',
            'n_components': 768,
            'k_neighbors': k,
            'auroc': auroc,
            'auprc': auprc
        })
    
    # Test reduced dimensionalities
    for method in tqdm(reduction_methods, desc="Reduction methods"):
        for n_comp in tqdm(dimensions_to_test, desc="Dimensions", leave=False):
            if n_comp >= feature_matrix.shape[1]:
                continue
                
            try:
                # Create and fit reducer
                reducer = create_reducer(method, n_comp, random_state=42)
                X_train_reduced = reducer.fit_transform(X_train)
                X_test_reduced = reducer.transform(X_test)
                
                # Scale reduced features
                scaler = MaxAbsScaler()
                X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
                X_test_reduced_scaled = scaler.transform(X_test_reduced)
                
                # Test different k values
                for k in k_values:
                    if k <= len(X_train_reduced_scaled):
                        knn = KNeighborsClassifier(n_neighbors=k)
                        knn.fit(X_train_reduced_scaled, y_train)
                        y_pred_proba = knn.predict_proba(X_test_reduced_scaled)[:, 1]
                        
                        auroc = roc_auc_score(y_test, y_pred_proba)
                        auprc = average_precision_score(y_test, y_pred_proba)
                        
                        results.append({
                            'method': method,
                            'n_components': n_comp,
                            'k_neighbors': k,
                            'auroc': auroc,
                            'auprc': auprc
                        })
                        
            except Exception as e:
                logger.error(f"Error with {method} {n_comp}D: {e}")
                continue
    
    return pd.DataFrame(results)


def plot_knn_dimensionality_analysis(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualizations of k-NN dimensionality analysis."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for score in ['auroc', 'auprc']:
        # Best k for each dimension
        best_k_results = results_df.loc[results_df.groupby(['method', 'n_components'])[score].idxmax()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Performance vs Dimensions (best k for each)
        for method in best_k_results['method'].unique():
            method_data = best_k_results[best_k_results['method'] == method].sort_values('n_components')
            
            ax1.plot(method_data['n_components'], method_data[score], 'o-', 
                    label=method, linewidth=2, markersize=6)
            
            # Add k values as text annotations
            for _, row in method_data.iterrows():
                ax1.annotate(f"k={int(row['k_neighbors'])}", 
                           (row['n_components'], row[score]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        ax1.set_xlabel('Number of Dimensions')
        ax1.set_ylabel(f'{score.upper()} Score')
        ax1.set_title(f'k-NN Performance vs Dimensionality (Best k)\n{score.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Heatmap of performance by k and dimensions (for best method)
        best_method = best_k_results.groupby('method')[score].mean().idxmax()
        heatmap_data = results_df[results_df['method'] == best_method]
        
        pivot_data = heatmap_data.pivot(index='k_neighbors', columns='n_components', values=score)
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax2)
        ax2.set_title(f'k-NN Performance Heatmap ({best_method})\n{score.upper()}')
        ax2.set_xlabel('Number of Dimensions')
        ax2.set_ylabel('k (Number of Neighbors)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'knn_dimensionality_analysis_{score}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def analyze_curse_of_dimensionality(results_df: pd.DataFrame) -> None:
    """Analyze evidence for curse of dimensionality in k-NN performance."""
    
    logger.info("🔍 CURSE OF DIMENSIONALITY ANALYSIS")
    print("=" * 60)
    
    for score in ['auroc', 'auprc']:
        print(f"\n📊 {score.upper()} Analysis:")
        print("-" * 30)
        
        # Find best performance for each dimensionality
        best_by_dim = results_df.loc[results_df.groupby('n_components')[score].idxmax()]
        best_by_dim = best_by_dim.sort_values('n_components')
        
        # Original performance (768D)
        original_perf = best_by_dim[best_by_dim['n_components'] == 768][score].iloc[0] if 768 in best_by_dim['n_components'].values else None
        
        if original_perf is not None:
            print(f"Original 768D performance: {original_perf:.4f}")
        
        # Best reduced dimensionality performance
        reduced_dims = best_by_dim[best_by_dim['n_components'] < 768]
        if not reduced_dims.empty:
            best_reduced = reduced_dims.loc[reduced_dims[score].idxmax()]
            print(f"Best reduced performance: {best_reduced[score]:.4f} ({int(best_reduced['n_components'])}D, {best_reduced['method']}, k={int(best_reduced['k_neighbors'])})")
            
            if original_perf is not None:
                improvement = best_reduced[score] - original_perf
                print(f"Improvement over 768D: {improvement:+.4f} ({improvement/original_perf*100:+.1f}%)")
        
        # Optimal dimensionality analysis
        print(f"\nTop 5 dimensionalities for {score.upper()}:")
        top_dims = best_by_dim.nlargest(5, score)
        for i, (_, row) in enumerate(top_dims.iterrows(), 1):
            print(f"  {i}. {int(row['n_components'])}D: {row[score]:.4f} ({row['method']}, k={int(row['k_neighbors'])})")


def main():
    parser = argparse.ArgumentParser(description="k-NN Dimensionality Analysis")
    parser.add_argument("--path_to_features", required=True, type=str,
                       help="Path to ClinicalBERT features file")
    parser.add_argument("--path_to_labels", required=True, type=str,
                       help="Path to labels file (CSV with patient_id and label columns)")
    parser.add_argument("--path_to_output_dir", required=True, type=str,
                       help="Path to output directory")
    parser.add_argument("--dimensions", type=str, default="16,32,64,128,256,384",
                       help="Comma-separated dimensions to test")
    parser.add_argument("--reduction_methods", type=str, default="pca,truncated_svd",
                       help="Comma-separated reduction methods")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum samples to use (for faster analysis)")
    
    args = parser.parse_args()
    
    # Parse arguments
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]
    methods = [m.strip() for m in args.reduction_methods.split(',')]
    
    logger.info("Starting k-NN Dimensionality Analysis")
    logger.info(f"Dimensions to test: {dimensions}")
    logger.info(f"Reduction methods: {methods}")
    
    # Load features
    feature_matrix, patient_ids, label_times = load_features(args.path_to_features)
    
    # Load labels (simplified - assumes CSV with patient_id and label columns)
    labels_df = pd.read_csv(args.path_to_labels)
    
    # For this quick analysis, we'll create synthetic labels based on feature characteristics
    # In practice, you'd align with actual task labels
    if 'label' not in labels_df.columns:
        logger.warning("No 'label' column found, creating synthetic binary labels")
        # Create labels based on first principal component (just for demonstration)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(feature_matrix)
        labels = (pc1.flatten() > np.median(pc1)).astype(int)
    else:
        # Align labels with features (simplified alignment)
        labels = labels_df['label'].values[:len(feature_matrix)]
    
    # Subsample if needed
    if len(feature_matrix) > args.max_samples:
        logger.info(f"Subsampling to {args.max_samples} samples")
        indices = np.random.choice(len(feature_matrix), args.max_samples, replace=False)
        feature_matrix = feature_matrix[indices]
        labels = labels[indices]
    
    logger.info(f"Using {len(feature_matrix)} samples with {len(labels)} labels")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Run analysis
    results_df = quick_knn_analysis(
        feature_matrix, labels, dimensions, methods
    )
    
    # Save results
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    results_path = os.path.join(args.path_to_output_dir, 'knn_dimensionality_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Create visualizations
    plot_knn_dimensionality_analysis(results_df, args.path_to_output_dir)
    
    # Print analysis
    analyze_curse_of_dimensionality(results_df)
    
    logger.success("k-NN Dimensionality Analysis Complete!")


if __name__ == "__main__":
    main() 