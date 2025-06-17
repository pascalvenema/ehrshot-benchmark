"""
Utility functions for dimensionality reduction analysis.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def load_features(path_to_features_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features from pickle file.
    
    Args:
        path_to_features_file: Path to the feature pickle file
        
    Returns:
        Tuple of (feature_matrix, patient_ids, label_times)
    """
    logger.info(f"Loading features from {path_to_features_file}")
    
    with open(path_to_features_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        feature_matrix = data['data_matrix']
        patient_ids = data['patient_ids'] 
        label_times = data['labeling_time']
    else:
        # Legacy format: tuple
        feature_matrix = data[0]
        patient_ids = data[1]
        label_times = data[3]  # Skip label_values at index 2
    
    logger.info(f"Loaded features: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
    
    return feature_matrix, patient_ids, label_times


def save_reduced_features(feature_matrix: np.ndarray, 
                         patient_ids: np.ndarray,
                         label_times: np.ndarray, 
                         path_to_output_file: str,
                         reducer_info: Dict[str, Any] = None) -> None:
    """Save reduced features to pickle file.
    
    Args:
        feature_matrix: Reduced feature matrix
        patient_ids: Patient IDs
        label_times: Label times
        path_to_output_file: Output file path
        reducer_info: Information about the reduction method used
    """
    logger.info(f"Saving reduced features to {path_to_output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(path_to_output_file), exist_ok=True)
    
    # Save in the same format as original features
    data = {
        'data_matrix': feature_matrix,
        'patient_ids': patient_ids,
        'labeling_time': label_times,
        'reducer_info': reducer_info,
        'creation_time': datetime.datetime.now().isoformat()
    }
    
    with open(path_to_output_file, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved reduced features: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")


def get_optimal_dimensions(X: np.ndarray, 
                          y: Optional[np.ndarray] = None,
                          method: str = 'elbow',
                          max_components: int = 200) -> List[int]:
    """Determine optimal number of dimensions using various heuristics.
    
    Args:
        X: Input feature matrix
        y: Labels (optional, used for some methods)
        method: Method to use ('elbow', 'variance_90', 'variance_95', 'variance_99')
        max_components: Maximum number of components to consider
        
    Returns:
        List of recommended dimensions
    """
    logger.info(f"Determining optimal dimensions using method: {method}")
    
    n_samples, n_features = X.shape
    max_components = min(max_components, n_samples - 1, n_features)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'elbow':
        # Use elbow method with PCA
        components_range = np.logspace(1, np.log10(max_components), 20, dtype=int)
        components_range = np.unique(components_range)
        
        explained_variances = []
        for n_comp in components_range:
            pca = PCA(n_components=n_comp)
            pca.fit(X_scaled)
            explained_variances.append(np.sum(pca.explained_variance_ratio_))
        
        # Find elbow point (simple method)
        diffs = np.diff(explained_variances)
        elbow_idx = np.argmax(diffs < 0.01)  # Where improvement becomes marginal
        optimal_dims = [components_range[elbow_idx]]
        
    elif method.startswith('variance_'):
        # Use variance threshold
        threshold = float(method.split('_')[1]) / 100.0
        
        pca = PCA(n_components=max_components)
        pca.fit(X_scaled)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= threshold) + 1
        optimal_dims = [n_components]
        
        logger.info(f"Components for {threshold:.1%} variance: {n_components}")
        
    else:
        # Default: provide common dimensions
        optimal_dims = [32, 64, 128, 256]
        optimal_dims = [d for d in optimal_dims if d < n_features]
    
    logger.info(f"Recommended dimensions: {optimal_dims}")
    return optimal_dims


def plot_explained_variance(X: np.ndarray, 
                           path_to_output_dir: str,
                           max_components: int = 300) -> None:
    """Plot explained variance ratio for PCA components.
    
    Args:
        X: Input feature matrix
        path_to_output_dir: Directory to save plot
        max_components: Maximum number of components to plot
    """
    logger.info("Creating explained variance plot")
    
    n_samples, n_features = X.shape
    max_components = min(max_components, n_samples - 1, n_features)
    
    # Fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=max_components)
    pca.fit(X_scaled)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual explained variance
    ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'b-', alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'r-', linewidth=2)
    ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% variance')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
    ax2.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(path_to_output_dir, exist_ok=True)
    plt.savefig(os.path.join(path_to_output_dir, 'explained_variance_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Explained variance plot saved to {path_to_output_dir}")


def plot_dimensionality_comparison(results_df: pd.DataFrame, 
                                  path_to_output_dir: str,
                                  score: str = 'auroc') -> None:
    """Plot performance comparison across different dimensionalities.
    
    Args:
        results_df: DataFrame with results from different dimensionalities
        path_to_output_dir: Directory to save plot
        score: Score to plot ('auroc' or 'auprc')
    """
    logger.info(f"Creating dimensionality comparison plot for {score}")
    
    # Filter for the specified score
    df_filtered = results_df[results_df['score'] == score].copy()
    
    if df_filtered.empty:
        logger.warning(f"No data found for score: {score}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for each reduction method
    for method in df_filtered['reduction_method'].unique():
        method_data = df_filtered[df_filtered['reduction_method'] == method]
        
        # Group by dimensionality and calculate mean/std
        grouped = method_data.groupby('n_components')['value'].agg(['mean', 'std'])
        
        ax.plot(grouped.index, grouped['mean'], 'o-', label=method, linewidth=2, markersize=6)
        ax.fill_between(grouped.index, 
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.2)
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel(f'{score.upper()} Score')
    ax.set_title(f'Performance vs Dimensionality ({score.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale for x-axis if range is large
    if df_filtered['n_components'].max() / df_filtered['n_components'].min() > 10:
        ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(path_to_output_dir, exist_ok=True)
    plt.savefig(os.path.join(path_to_output_dir, f'dimensionality_comparison_{score}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dimensionality comparison plot saved to {path_to_output_dir}")


def create_reduction_report(results_df: pd.DataFrame, 
                           path_to_output_dir: str) -> None:
    """Create a comprehensive report of dimensionality reduction results.
    
    Args:
        results_df: DataFrame with all results
        path_to_output_dir: Directory to save report
    """
    logger.info("Creating dimensionality reduction report")
    
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Check if results_df is empty or missing required columns
    if results_df.empty:
        logger.warning("No results to create report from")
        return
        
    if 'score' not in results_df.columns:
        logger.warning("Results DataFrame missing 'score' column")
        return
    
    # Summary statistics
    summary_stats = []
    
    for score in ['auroc', 'auprc']:
        score_data = results_df[results_df['score'] == score]
        
        if score_data.empty:
            continue
            
        for method in score_data['reduction_method'].unique():
            method_data = score_data[score_data['reduction_method'] == method]
            
            for n_comp in method_data['n_components'].unique():
                comp_data = method_data[method_data['n_components'] == n_comp]
                
                summary_stats.append({
                    'score': score,
                    'reduction_method': method,
                    'n_components': n_comp,
                    'mean_performance': comp_data['value'].mean(),
                    'std_performance': comp_data['value'].std(),
                    'min_performance': comp_data['value'].min(),
                    'max_performance': comp_data['value'].max(),
                    'n_tasks': len(comp_data['labeling_function'].unique())
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary table
    summary_df.to_csv(os.path.join(path_to_output_dir, 'dimensionality_reduction_summary.csv'), 
                     index=False)
    
    # Create best performing combinations table
    best_combinations = []
    
    for score in ['auroc', 'auprc']:
        score_data = results_df[results_df['score'] == score]
        
        if score_data.empty:
            continue
            
        for task in score_data['task'].unique():
            task_data = score_data[score_data['task'] == task]
            
            # Find best combination for this task
            best_idx = task_data['value'].idxmax()
            best_row = task_data.loc[best_idx]
            
            best_combinations.append({
                'task': task,
                'score': score,
                'best_method': best_row['reduction_method'],
                'best_n_components': best_row['n_components'],
                'best_performance': best_row['value'],
                'original_dims': 768,  # ClinicalBERT embedding size
                'reduction_ratio': best_row['n_components'] / 768
            })
    
    best_df = pd.DataFrame(best_combinations)
    best_df.to_csv(os.path.join(path_to_output_dir, 'best_dimensionality_combinations.csv'), 
                  index=False)
    
    # Create plots
    for score in ['auroc', 'auprc']:
        if score in results_df['score'].unique():
            plot_dimensionality_comparison(results_df, path_to_output_dir, score)
    
    logger.info(f"Dimensionality reduction report saved to {path_to_output_dir}")


def visualize_embeddings_2d(X: np.ndarray, 
                           y: Optional[np.ndarray] = None,
                           method: str = 'pca',
                           path_to_output_file: str = None,
                           title: str = "2D Embedding Visualization") -> None:
    """Create 2D visualization of high-dimensional embeddings.
    
    Args:
        X: High-dimensional embeddings
        y: Labels for coloring (optional)
        method: Reduction method ('pca' or 'tsne')
        path_to_output_file: Where to save the plot
        title: Plot title
    """
    logger.info(f"Creating 2D visualization using {method}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        xlabel = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)'
        ylabel = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)'
    elif method == 'tsne':
        # Use PCA preprocessing for t-SNE if data is very high-dimensional
        if X.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
        
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = reducer.fit_transform(X_scaled)
        xlabel = 't-SNE Dimension 1'
        ylabel = 't-SNE Dimension 2'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if y is not None:
        # Color by labels
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter)
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=20)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if path_to_output_file:
        os.makedirs(os.path.dirname(path_to_output_file), exist_ok=True)
        plt.savefig(path_to_output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"2D visualization saved to {path_to_output_file}")
    else:
        plt.show()


def analyze_feature_importance(X: np.ndarray, 
                              y: np.ndarray,
                              method: str = 'pca',
                              n_components: int = 100) -> np.ndarray:
    """Analyze which original features are most important after reduction.
    
    Args:
        X: Original feature matrix
        y: Labels
        method: Reduction method
        n_components: Number of components
        
    Returns:
        Feature importance scores
    """
    from .reducers import create_reducer
    
    logger.info(f"Analyzing feature importance using {method}")
    
    reducer = create_reducer(method, n_components)
    reducer.fit(X, y)
    
    if hasattr(reducer.reducer, 'components_'):
        # For PCA-like methods, compute feature importance as sum of absolute loadings
        components = reducer.reducer.components_
        feature_importance = np.sum(np.abs(components), axis=0)
    elif hasattr(reducer.reducer, 'scores_'):
        # For feature selection methods
        feature_importance = reducer.reducer.scores_
    else:
        logger.warning(f"Cannot compute feature importance for method: {method}")
        feature_importance = np.ones(X.shape[1])
    
    return feature_importance 