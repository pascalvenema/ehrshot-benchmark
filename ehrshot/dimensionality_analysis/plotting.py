"""
Plotting Module for Dimensionality Reduction Analysis

This module creates visualizations to show how kNN performance changes
with different dimensionality reduction techniques and dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import os
from loguru import logger


def plot_dimensionality_results(
    results_df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create comprehensive plots of dimensionality reduction results.
    
    Args:
        results_df: DataFrame with results from dimensionality experiments
        output_dir: Directory to save plots
        figsize: Figure size for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter out failed results
    df_clean = results_df[~results_df['auroc'].isna()].copy()
    
    if df_clean.empty:
        logger.warning("No valid results to plot")
        return
    
    # 1. Performance vs Dimensions plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot for each metric
    for i, metric in enumerate(['auroc', 'auprc']):
        ax = axes[i, 0]
        
        # Plot each method
        for method in df_clean['reduction_method'].unique():
            if method == 'none':
                # Show baseline as horizontal line
                baseline_score = df_clean[df_clean['reduction_method'] == 'none'][metric].iloc[0]
                ax.axhline(y=baseline_score, color='black', linestyle='--', 
                          label=f'Baseline (768D)', alpha=0.7, linewidth=2)
            else:
                method_data = df_clean[df_clean['reduction_method'] == method]
                ax.plot(method_data['n_components'], method_data[metric], 
                       marker='o', label=method.upper(), linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of Dimensions')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'{metric.upper()} vs Dimensionality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Use actual dimension numbers on x-axis
        unique_dims = sorted(df_clean['n_components'].unique())
        ax.set_xticks(unique_dims)
        ax.set_xticklabels([str(d) for d in unique_dims])
    
    # 2. Performance improvement over baseline
    for i, metric in enumerate(['auroc', 'auprc']):
        ax = axes[i, 1]
        
        baseline_score = df_clean[df_clean['reduction_method'] == 'none'][metric].iloc[0]
        
        for method in df_clean['reduction_method'].unique():
            if method != 'none':
                method_data = df_clean[df_clean['reduction_method'] == method]
                improvement = method_data[metric] - baseline_score
                ax.plot(method_data['n_components'], improvement, 
                       marker='o', label=method.upper(), linewidth=2, markersize=6)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax.set_xlabel('Number of Dimensions')
        ax.set_ylabel(f'{metric.upper()} Improvement over Baseline')
        ax.set_title(f'{metric.upper()} Improvement vs Dimensionality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Use actual dimension numbers on x-axis
        unique_dims = sorted(df_clean['n_components'].unique())
        ax.set_xticks(unique_dims)
        ax.set_xticklabels([str(d) for d in unique_dims])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimensionality_analysis_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed comparison for each task (if multiple tasks)
    if len(df_clean['task'].unique()) > 1:
        plot_multi_task_comparison(df_clean, output_dir)
    
    # 4. Best dimensions heatmap
    plot_best_dimensions_heatmap(df_clean, output_dir)
    
    logger.info(f"Plots saved to {output_dir}")


def plot_multi_task_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot comparison across multiple tasks."""
    tasks = df['task'].unique()
    n_tasks = len(tasks)
    
    fig, axes = plt.subplots(2, n_tasks, figsize=(5*n_tasks, 10))
    if n_tasks == 1:
        axes = axes.reshape(2, 1)
    
    for i, task in enumerate(tasks):
        task_data = df[df['task'] == task]
        
        for j, metric in enumerate(['auroc', 'auprc']):
            ax = axes[j, i]
            
            # Plot baseline
            baseline_score = task_data[task_data['reduction_method'] == 'none'][metric].iloc[0]
            ax.axhline(y=baseline_score, color='black', linestyle='--', 
                      label='Baseline', alpha=0.7, linewidth=2)
            
            # Plot each method
            for method in task_data['reduction_method'].unique():
                if method != 'none':
                    method_data = task_data[task_data['reduction_method'] == method]
                    ax.plot(method_data['n_components'], method_data[metric], 
                           marker='o', label=method.upper(), linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Dimensions')
            ax.set_ylabel(f'{metric.upper()} Score')
            ax.set_title(f'{task} - {metric.upper()}')
            if i == 0:  # Only show legend on first plot
                ax.legend()
            ax.grid(True, alpha=0.3)
            # Use actual dimension numbers on x-axis
            unique_dims = sorted(task_data['n_components'].unique())
            ax.set_xticks(unique_dims)
            ax.set_xticklabels([str(d) for d in unique_dims])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_task_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_best_dimensions_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Create heatmap showing best dimensions for each method."""
    
    # Find best dimension for each method/task/metric combination
    best_dims = []
    
    for task in df['task'].unique():
        for metric in ['auroc', 'auprc']:
            task_data = df[(df['task'] == task) & (df['reduction_method'] != 'none')]
            
            if task_data.empty:
                continue
                
            for method in task_data['reduction_method'].unique():
                method_data = task_data[task_data['reduction_method'] == method]
                if not method_data.empty:
                    best_idx = method_data[metric].idxmax()
                    best_row = method_data.loc[best_idx]
                    
                    best_dims.append({
                        'task': task,
                        'metric': metric,
                        'method': method,
                        'best_dims': best_row['n_components'],
                        'best_score': best_row[metric]
                    })
    
    if not best_dims:
        logger.warning("No data for heatmap")
        return
        
    best_df = pd.DataFrame(best_dims)
    
    # Create pivot table for heatmap
    pivot_dims = best_df.pivot_table(
        index=['task', 'metric'], 
        columns='method', 
        values='best_dims', 
        aggfunc='first'
    )
    
    pivot_scores = best_df.pivot_table(
        index=['task', 'metric'], 
        columns='method', 
        values='best_score', 
        aggfunc='first'
    )
    
    # Plot best dimensions heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    sns.heatmap(pivot_dims, annot=True, fmt='.0f', cmap='viridis', 
                ax=ax1, cbar_kws={'label': 'Best Number of Dimensions'})
    ax1.set_title('Best Dimensions for Each Method')
    ax1.set_xlabel('Reduction Method')
    ax1.set_ylabel('Task / Metric')
    
    sns.heatmap(pivot_scores, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=ax2, cbar_kws={'label': 'Best Score'})
    ax2.set_title('Best Scores Achieved')
    ax2.set_xlabel('Reduction Method')
    ax2.set_ylabel('Task / Metric')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_dimensions_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(results_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Create a summary table of results with baseline comparisons."""
    # Calculate mean performance across tasks for each method/dimension/model combination
    summary_data = []
    
    for model in results_df['model'].unique():
        for method in results_df['method'].unique():
            for dimension in sorted(results_df['dimension'].unique()):
                for metric in results_df['metric'].unique():
                    subset = results_df[
                        (results_df['model'] == model) &
                        (results_df['method'] == method) &
                        (results_df['dimension'] == dimension) &
                        (results_df['metric'] == metric)
                    ]
                    
                    if not subset.empty:
                        mean_score = subset['score'].mean()
                        std_score = subset['score'].std()
                        
                        summary_data.append({
                            'model': model,
                            'method': method,
                            'dimension': dimension,
                            'metric': metric,
                            'mean_score': mean_score,
                            'std_score': std_score,
                            'n_tasks': len(subset)
                        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if not summary_df.empty:
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Summary table saved to {output_path}")
    
    return summary_df

def load_baseline_results(results_dir: str, tasks: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load baseline kNN results from existing CSV files.
    
    Returns:
        Dict[task][model][metric] = score
    """
    baseline_results = {}
    
    for task in tasks:
        csv_path = os.path.join(results_dir, f"{task}/all_results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: No results file found for {task}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Filter for kNN results with full data (k=-1)
        knn_baseline = df[
            (df['head'] == 'knn') & 
            (df['k'] == -1) &
            (df['replicate'] == 0)  # Use first replicate
        ]
        
        task_results = {}
        
        # Get CLMBR kNN baseline
        clmbr_data = knn_baseline[knn_baseline['model'] == 'clmbr']
        if not clmbr_data.empty:
            task_results['clmbr'] = {}
            for _, row in clmbr_data.iterrows():
                task_results['clmbr'][row['score']] = row['value']
        
        # Get ClinicalBERT type3 clinicalbert_pool kNN baseline
        cb_data = knn_baseline[knn_baseline['model'] == 'clinicalbert_type3_clinicalbert_pool']
        if not cb_data.empty:
            task_results['clinicalbert_type3_clinicalbert_pool'] = {}
            for _, row in cb_data.iterrows():
                task_results['clinicalbert_type3_clinicalbert_pool'][row['score']] = row['value']
        
        baseline_results[task] = task_results
    
    return baseline_results

def plot_dimensionality_analysis_comparison(
    results_df: pd.DataFrame,
    baseline_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    title: str = "Dimensionality Reduction Analysis"
) -> None:
    """
    Create comparison plots for dimensionality reduction analysis with separate subplots
    for CLMBR and ClinicalBERT type3 clinicalbert_pool.
    """
    # Create figure with 2x2 subplots (AUROC/AUPRC for CLMBR and ClinicalBERT)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Define colors for methods
    method_colors = {
        'pca': '#1f77b4',
        'tsne': '#ff7f0e', 
        'umap': '#2ca02c'
    }
    
    # Define models to plot
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    metrics = ['auroc', 'auprc']
    metric_labels = ['AUROC', 'AUPRC']
    
    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[metric_idx, model_idx]
            
            # Filter results for this model
            if model == 'clmbr':
                model_data = results_df[results_df['model'] == 'clmbr']
                baseline_key = 'clmbr'
            else:
                model_data = results_df[results_df['model'] == 'clinicalbert_type3_clinicalbert_pool']
                baseline_key = 'clinicalbert_type3_clinicalbert_pool'
            
            if model_data.empty:
                ax.text(0.5, 0.5, f'No {model_label} data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{model_label} - {metric_label}')
                continue
            
            # Get unique tasks and dimensions
            tasks = model_data['task'].unique()
            dimensions = sorted(model_data['dimension'].unique())
            methods = sorted(model_data['method'].unique())
            
            # Calculate baselines across all tasks
            baseline_scores = []
            for task in tasks:
                if task in baseline_results and baseline_key in baseline_results[task]:
                    if metric in baseline_results[task][baseline_key]:
                        baseline_scores.append(baseline_results[task][baseline_key][metric])
            
            if baseline_scores:
                baseline_mean = np.mean(baseline_scores)
                baseline_std = np.std(baseline_scores)
            else:
                baseline_mean = 0.5  # Default if no baseline found
                baseline_std = 0
            
            # Plot baseline as horizontal line
            ax.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Original 768D Baseline\n({baseline_mean:.3f}±{baseline_std:.3f})')
            ax.fill_between(range(len(dimensions)), 
                          baseline_mean - baseline_std, 
                          baseline_mean + baseline_std,
                          color='red', alpha=0.1)
            
            # Plot each method
            x_positions = np.arange(len(dimensions))
            
            for method in methods:
                method_data = model_data[model_data['method'] == method]
                
                if method_data.empty:
                    continue
                
                # Calculate mean and std across tasks for each dimension
                mean_scores = []
                std_scores = []
                
                for dim in dimensions:
                    dim_scores = method_data[
                        (method_data['dimension'] == dim) & 
                        (method_data['metric'] == metric)
                    ]['score'].values
                    
                    if len(dim_scores) > 0:
                        mean_scores.append(np.mean(dim_scores))
                        std_scores.append(np.std(dim_scores))
                    else:
                        mean_scores.append(0)
                        std_scores.append(0)
                
                # Plot line with error bars
                ax.errorbar(x_positions, mean_scores, yerr=std_scores,
                           color=method_colors.get(method, 'gray'),
                           marker='o', linewidth=2, markersize=8,
                           label=f'{method.upper()}', capsize=5)
            
            # Customize axes
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(d) for d in dimensions])
            ax.set_xlabel('Reduced Dimensions')
            ax.set_ylabel(f'{metric_label} Score')
            ax.set_title(f'{model_label} - {metric_label}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Set reasonable y-limits based on metric
            if metric == 'auroc':
                ax.set_ylim(0.45, 0.75)
            else:  # auprc
                ax.set_ylim(0.0, 0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")

def plot_combined_dimensionality_analysis(
    results_df: pd.DataFrame,
    baseline_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    title: str = "Combined Dimensionality Reduction Analysis"
) -> None:
    """
    Create combined plots showing both CLMBR and ClinicalBERT type3 on the same axes.
    """
    # Create figure with 1x2 subplots (AUROC and AUPRC)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Define colors for model+method combinations
    colors = {
        ('clmbr', 'pca'): '#1f77b4',
        ('clmbr', 'tsne'): '#ff7f0e',
        ('clmbr', 'umap'): '#2ca02c',
        ('clinicalbert_type3_clinicalbert_pool', 'pca'): '#d62728',
        ('clinicalbert_type3_clinicalbert_pool', 'tsne'): '#9467bd',
        ('clinicalbert_type3_clinicalbert_pool', 'umap'): '#8c564b'
    }
    
    # Define line styles
    linestyles = {
        'clmbr': '-',
        'clinicalbert_type3_clinicalbert_pool': '--'
    }
    
    metrics = ['auroc', 'auprc']
    metric_labels = ['AUROC', 'AUPRC']
    
    for metric_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[metric_idx]
        
        # Get unique dimensions from all data
        dimensions = sorted(results_df['dimension'].unique())
        x_positions = np.arange(len(dimensions))
        
        # Plot baselines for both models
        models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
        model_labels = ['CLMBR', 'ClinicalBERT Type3']
        baseline_colors = ['blue', 'red']
        
        for model, model_label, bl_color in zip(models, model_labels, baseline_colors):
            # Calculate baseline across all tasks
            baseline_scores = []
            tasks = results_df['task'].unique()
            
            for task in tasks:
                if (task in baseline_results and 
                    model in baseline_results[task] and 
                    metric in baseline_results[task][model]):
                    baseline_scores.append(baseline_results[task][model][metric])
            
            if baseline_scores:
                baseline_mean = np.mean(baseline_scores)
                baseline_std = np.std(baseline_scores)
                
                ax.axhline(y=baseline_mean, color=bl_color, linestyle=':', linewidth=2, 
                          alpha=0.7, label=f'{model_label} 768D Baseline ({baseline_mean:.3f})')
        
        # Plot each model+method combination
        for model in models:
            model_data = results_df[results_df['model'] == model]
            if model_data.empty:
                continue
                
            methods = sorted(model_data['method'].unique())
            model_label = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT Type3'
            
            for method in methods:
                method_data = model_data[model_data['method'] == method]
                
                if method_data.empty:
                    continue
                
                # Calculate mean and std across tasks for each dimension
                mean_scores = []
                std_scores = []
                
                for dim in dimensions:
                    dim_scores = method_data[
                        (method_data['dimension'] == dim) & 
                        (method_data['metric'] == metric)
                    ]['score'].values
                    
                    if len(dim_scores) > 0:
                        mean_scores.append(np.mean(dim_scores))
                        std_scores.append(np.std(dim_scores))
                    else:
                        mean_scores.append(0)
                        std_scores.append(0)
                
                # Plot line with error bars
                color = colors.get((model, method), 'gray')
                linestyle = linestyles.get(model, '-')
                
                ax.errorbar(x_positions, mean_scores, yerr=std_scores,
                           color=color, linestyle=linestyle,
                           marker='o', linewidth=2, markersize=6,
                           label=f'{model_label} {method.upper()}', capsize=3)
        
        # Customize axes
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.set_xlabel('Reduced Dimensions')
        ax.set_ylabel(f'{metric_label} Score')
        ax.set_title(f'{metric_label} Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Set reasonable y-limits based on metric
        if metric == 'auroc':
            ax.set_ylim(0.45, 0.75)
        else:  # auprc
            ax.set_ylim(0.0, 0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to {output_path}")

def plot_best_dimensions_heatmap(results_df: pd.DataFrame, output_path: str) -> None:
    """Create heatmap showing best performing dimensions for each task/method combination."""
    # Find best dimension for each task/method/model/metric combination
    best_dims = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            for method in results_df['method'].unique():
                for metric in results_df['metric'].unique():
                    subset = results_df[
                        (results_df['task'] == task) &
                        (results_df['model'] == model) &
                        (results_df['method'] == method) &
                        (results_df['metric'] == metric)
                    ]
                    
                    if not subset.empty:
                        best_idx = subset['score'].idxmax()
                        best_row = subset.loc[best_idx]
                        best_dims.append({
                            'task': task,
                            'model': model,
                            'method': method,
                            'metric': metric,
                            'best_dimension': best_row['dimension'],
                            'best_score': best_row['score']
                        })
    
    if not best_dims:
        print("No data available for heatmap")
        return
    
    best_dims_df = pd.DataFrame(best_dims)
    
    # Create separate heatmaps for each metric
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    for i, metric in enumerate(['auroc', 'auprc']):
        metric_data = best_dims_df[best_dims_df['metric'] == metric]
        
        if metric_data.empty:
            continue
        
        # Pivot for heatmap
        pivot_data = metric_data.pivot_table(
            index=['model', 'method'], 
            columns='task',
            values='best_dimension',
            aggfunc='first'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='g', cmap='RdYlBu_r', 
                   ax=axes[i], cbar_kws={'label': 'Best Dimension'})
        axes[i].set_title(f'Best Dimensions for {metric.upper()}')
        axes[i].set_ylabel('Model + Method')
        axes[i].set_xlabel('Task')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best dimensions heatmap saved to {output_path}")

def create_all_plots(results_df: pd.DataFrame, results_dir: str, output_dir: str) -> None:
    """Create all dimensionality analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique tasks from results
    tasks = results_df['task'].unique()
    
    # Load baseline results
    print("Loading baseline kNN results...")
    baseline_results = load_baseline_results(results_dir, tasks)
    
    # Create comparison plots (separate subplots)
    plot_dimensionality_analysis_comparison(
        results_df, baseline_results,
        os.path.join(output_dir, "dimensionality_analysis_comparison.png"),
        "Dimensionality Reduction Analysis - Model Comparison"
    )
    
    # Create combined plots (same axes)  
    plot_combined_dimensionality_analysis(
        results_df, baseline_results,
        os.path.join(output_dir, "dimensionality_analysis_combined.png"),
        "Dimensionality Reduction Analysis - Combined View"
    )
    
    # Create best dimensions heatmap
    plot_best_dimensions_heatmap(
        results_df,
        os.path.join(output_dir, "best_dimensions_heatmap.png")
    )
    
    # Create summary table
    create_summary_table(
        results_df,
        os.path.join(output_dir, "dimensionality_summary.csv")
    )
    
    print(f"All plots saved to {output_dir}") 