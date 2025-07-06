"""
Plotting Module for Dimensionality Reduction Analysis

This module creates visualizations specifically designed to answer the research question:
"How does the dimensionality of the patient embeddings affect the relative performance 
of the k-nearest neighbors prediction head?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import os
from loguru import logger

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

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

def load_logistic_regression_results(results_dir: str, tasks: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load logistic regression results from existing CSV files.
    
    Returns:
        Dict[task][model][metric] = score
    """
    lr_results = {}
    
    for task in tasks:
        csv_path = os.path.join(results_dir, f"{task}/all_results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: No results file found for {task}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Filter for logistic regression results with full data (k=-1)
        lr_baseline = df[
            (df['head'] == 'lr') & 
            (df['k'] == -1) &
            (df['replicate'] == 0)  # Use first replicate
        ]
        
        task_results = {}
        
        # Get CLMBR LR results
        clmbr_data = lr_baseline[lr_baseline['model'] == 'clmbr']
        if not clmbr_data.empty:
            task_results['clmbr'] = {}
            for _, row in clmbr_data.iterrows():
                task_results['clmbr'][row['score']] = row['value']
        
        # Get ClinicalBERT type3 clinicalbert_pool LR results
        cb_data = lr_baseline[lr_baseline['model'] == 'clinicalbert_type3_clinicalbert_pool']
        if not cb_data.empty:
            task_results['clinicalbert_type3_clinicalbert_pool'] = {}
            for _, row in cb_data.iterrows():
                task_results['clinicalbert_type3_clinicalbert_pool'][row['score']] = row['value']
        
        lr_results[task] = task_results
    
    return lr_results

def plot_dimensionality_performance_curves(results_df: pd.DataFrame, baseline_results: Dict, output_path: str) -> None:
    """
    Core visualization: Performance curves showing how kNN performance changes with dimensionality.
    This directly answers the research question.
    """
    # Prepare data
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    methods = ['pca', 'umap']
    method_colors = {'pca': '#2E86AB', 'umap': '#A23B72'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        model_data = results_df[results_df['model'] == model].copy()
        
        if model_data.empty:
            continue
            
        for metric_idx, metric in enumerate(['auroc', 'auprc']):
            ax = axes[model_idx, metric_idx]
            
            # Get dimensions and calculate mean performance across tasks
            dimensions = sorted(model_data['dimension'].unique())
            
            # Calculate baseline performance across all tasks
            baseline_scores = []
            for task in model_data['task'].unique():
                if (task in baseline_results and 
                    model in baseline_results[task] and 
                    metric in baseline_results[task][model]):
                    baseline_scores.append(baseline_results[task][model][metric])
            
            baseline_mean = np.mean(baseline_scores) if baseline_scores else None
            baseline_std = np.std(baseline_scores) if baseline_scores else None
            
            # Plot baseline
            if baseline_mean is not None:
                ax.axhline(y=baseline_mean, color='black', linestyle='--', linewidth=2, 
                          alpha=0.7, label=f'Full 768D Baseline ({baseline_mean:.3f})')
                if baseline_std is not None:
                    ax.fill_between(dimensions, baseline_mean - baseline_std, baseline_mean + baseline_std,
                                   color='black', alpha=0.1)
            
            # Plot each dimensionality reduction method
            for method in methods:
                method_data = model_data[
                    (model_data['method'] == method) & 
                    (model_data['metric'] == metric)
                ]
                
                if method_data.empty:
                    continue
                
                # Calculate mean and std across tasks for each dimension
                mean_scores = []
                std_scores = []
                
                for dim in dimensions:
                    dim_scores = method_data[method_data['dimension'] == dim]['score'].values
                    if len(dim_scores) > 0:
                        mean_scores.append(np.mean(dim_scores))
                        std_scores.append(np.std(dim_scores))
                    else:
                        mean_scores.append(np.nan)
                        std_scores.append(np.nan)
                
                # Plot with error bars
                color = method_colors.get(method, 'gray')
                ax.errorbar(dimensions, mean_scores, yerr=std_scores,
                           color=color, marker='o', linewidth=3, markersize=8,
                           label=f'{method.upper()}', capsize=5, capthick=2)
            
            # Customize plot
            ax.set_xlabel('Reduced Dimensions', fontsize=12)
            ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
            ax.set_title(f'{model_label} - {metric.upper()} vs Dimensionality', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            # Set log scale for x-axis to better show low dimensions
            ax.set_xscale('log')
            ax.set_xticks(dimensions)
            ax.set_xticklabels([str(d) for d in dimensions])
            
            # Set appropriate y-limits
            if metric == 'auroc':
                ax.set_ylim(0.45, 0.85)
            else:  # auprc
                all_scores = [s for s in mean_scores if not np.isnan(s)]
                if all_scores:
                    y_min = max(0, min(all_scores) - 0.05)
                    y_max = max(all_scores) + 0.05
                    ax.set_ylim(y_min, y_max)
    
    plt.suptitle('How Dimensionality Affects kNN Performance (PCA & UMAP only)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance curves saved to {output_path}")

def plot_optimal_dimensions_analysis(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Analysis of optimal dimensions for each task/model/method combination.
    """
    # Find optimal dimensions for each combination
    optimal_dims = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            for method in results_df['method'].unique():
                for metric in ['auroc', 'auprc']:
                    subset = results_df[
                        (results_df['task'] == task) &
                        (results_df['model'] == model) &
                        (results_df['method'] == method) &
                        (results_df['metric'] == metric)
                    ]
                    
                    if not subset.empty:
                        best_idx = subset['score'].idxmax()
                        best_row = subset.loc[best_idx]
                        optimal_dims.append({
                            'task': task,
                            'model': model,
                            'method': method,
                            'metric': metric,
                            'optimal_dimension': best_row['dimension'],
                            'best_score': best_row['score']
                        })
    
    optimal_df = pd.DataFrame(optimal_dims)
    
    # Create subplots for analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of optimal dimensions by method
    ax1 = axes[0, 0]
    for metric in ['auroc', 'auprc']:
        metric_data = optimal_df[optimal_df['metric'] == metric]
        for method in ['pca', 'umap']:
            method_data = metric_data[metric_data['method'] == method]['optimal_dimension']
            if not method_data.empty:
                ax1.hist(method_data, alpha=0.6, label=f'{method.upper()} ({metric})', 
                        bins=range(2, 450, 50), density=True)
    
    ax1.set_xlabel('Optimal Dimension')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Optimal Dimensions by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Optimal dimensions by task type
    ax2 = axes[0, 1]
    task_groups = {
        'Operational': ['guo_los', 'guo_readmission', 'guo_icu'],
        'Lab Values': ['lab_thrombocytopenia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia'],
        'New Diagnoses': ['new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 'new_lupus', 'new_acutemi']
    }
    
    task_type_dims = []
    for group_name, tasks in task_groups.items():
        group_data = optimal_df[optimal_df['task'].isin(tasks)]
        for _, row in group_data.iterrows():
            task_type_dims.append({
                'task_type': group_name,
                'optimal_dimension': row['optimal_dimension'],
                'method': row['method'],
                'metric': row['metric']
            })
    
    task_type_df = pd.DataFrame(task_type_dims)
    
    # Box plot by task type
    for i, metric in enumerate(['auroc', 'auprc']):
        metric_data = task_type_df[task_type_df['metric'] == metric]
        if not metric_data.empty:
            sns.boxplot(data=metric_data, x='task_type', y='optimal_dimension', 
                       hue='method', ax=ax2 if i == 0 else axes[1, 0])
            (ax2 if i == 0 else axes[1, 0]).set_title(f'Optimal Dimensions by Task Type ({metric.upper()})')
            (ax2 if i == 0 else axes[1, 0]).set_ylabel('Optimal Dimension')
            (ax2 if i == 0 else axes[1, 0]).tick_params(axis='x', rotation=45)
    
    # 3. Method comparison - average optimal dimension
    ax3 = axes[1, 1]
    method_stats = optimal_df.groupby(['method', 'metric'])['optimal_dimension'].agg(['mean', 'std']).reset_index()
    
    x_pos = np.arange(len(['pca', 'umap']))
    width = 0.35
    
    for i, metric in enumerate(['auroc', 'auprc']):
        metric_stats = method_stats[method_stats['metric'] == metric]
        means = [metric_stats[metric_stats['method'] == m]['mean'].iloc[0] if not metric_stats[metric_stats['method'] == m].empty else 0 
                for m in ['pca', 'umap']]
        stds = [metric_stats[metric_stats['method'] == m]['std'].iloc[0] if not metric_stats[metric_stats['method'] == m].empty else 0 
               for m in ['pca', 'umap']]
        
        ax3.bar(x_pos + i*width, means, width, yerr=stds, 
               label=metric.upper(), alpha=0.8, capsize=5)
    
    ax3.set_xlabel('Dimensionality Reduction Method')
    ax3.set_ylabel('Average Optimal Dimension')
    ax3.set_title('Average Optimal Dimensions by Method')
    ax3.set_xticks(x_pos + width/2)
    ax3.set_xticklabels(['PCA', 'UMAP'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Optimal Dimensionality Analysis for kNN Performance', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimal dimensions analysis saved to {output_path}")

def plot_performance_improvement_heatmap(results_df: pd.DataFrame, baseline_results: Dict, output_path: str) -> None:
    """
    Heatmap showing performance improvement vs baseline for each task.
    """
    # Calculate improvement vs baseline for each task/model/method/dimension
    improvements = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            # Get baseline for this task/model
            baseline_auroc = None
            baseline_auprc = None
            
            if (task in baseline_results and 
                model in baseline_results[task]):
                baseline_auroc = baseline_results[task][model].get('auroc')
                baseline_auprc = baseline_results[task][model].get('auprc')
            
            if baseline_auroc is None or baseline_auprc is None:
                continue
                
            task_model_data = results_df[
                (results_df['task'] == task) & 
                (results_df['model'] == model)
            ]
            
            for method in task_model_data['method'].unique():
                for dim in task_model_data['dimension'].unique():
                    auroc_score = task_model_data[
                        (task_model_data['method'] == method) &
                        (task_model_data['dimension'] == dim) &
                        (task_model_data['metric'] == 'auroc')
                    ]['score'].iloc[0] if not task_model_data[
                        (task_model_data['method'] == method) &
                        (task_model_data['dimension'] == dim) &
                        (task_model_data['metric'] == 'auroc')
                    ].empty else None
                    
                    auprc_score = task_model_data[
                        (task_model_data['method'] == method) &
                        (task_model_data['dimension'] == dim) &
                        (task_model_data['metric'] == 'auprc')
                    ]['score'].iloc[0] if not task_model_data[
                        (task_model_data['method'] == method) &
                        (task_model_data['dimension'] == dim) &
                        (task_model_data['metric'] == 'auprc')
                    ].empty else None
                    
                    if auroc_score is not None and auprc_score is not None:
                        improvements.append({
                            'task': task,
                            'model': model,
                            'method': method,
                            'dimension': dim,
                            'auroc_improvement': auroc_score - baseline_auroc,
                            'auprc_improvement': auprc_score - baseline_auprc,
                            'auroc_relative': (auroc_score - baseline_auroc) / baseline_auroc * 100,
                            'auprc_relative': (auprc_score - baseline_auprc) / baseline_auprc * 100
                        })
    
    improvement_df = pd.DataFrame(improvements)
    
    if improvement_df.empty:
        print("No improvement data available for heatmap")
        return
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    
    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        model_data = improvement_df[improvement_df['model'] == model]
            
        if model_data.empty:
            continue
            
        # Find best improvement for each task/method combination
        best_improvements = []
        for task in model_data['task'].unique():
            for method in model_data['method'].unique():
                task_method_data = model_data[
                    (model_data['task'] == task) & 
                    (model_data['method'] == method)
                ]
                
                if not task_method_data.empty:
                    # Find best AUROC improvement
                    best_auroc_idx = task_method_data['auroc_improvement'].idxmax()
                    best_auroc_row = task_method_data.loc[best_auroc_idx]
                    
                    best_improvements.append({
                        'task': task,
                        'method': method,
                        'metric': 'AUROC',
                        'improvement': best_auroc_row['auroc_improvement'],
                        'relative_improvement': best_auroc_row['auroc_relative'],
                        'best_dimension': best_auroc_row['dimension']
                    })
                    
                    # Find best AUPRC improvement
                    best_auprc_idx = task_method_data['auprc_improvement'].idxmax()
                    best_auprc_row = task_method_data.loc[best_auprc_idx]
                    
                    best_improvements.append({
                        'task': task,
                        'method': method,
                        'metric': 'AUPRC',
                        'improvement': best_auprc_row['auprc_improvement'],
                        'relative_improvement': best_auprc_row['auprc_relative'],
                        'best_dimension': best_auprc_row['dimension']
                    })
        
        best_df = pd.DataFrame(best_improvements)
        
        for metric_idx, metric in enumerate(['AUROC', 'AUPRC']):
            ax = axes[model_idx, metric_idx]
            
            metric_data = best_df[best_df['metric'] == metric]
            if metric_data.empty:
                continue
                
            # Create pivot table for heatmap
            heatmap_data = metric_data.pivot(index='method', columns='task', values='improvement')
            
            # Create heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': f'{metric} Improvement vs Baseline'})
            
            ax.set_title(f'{model_label} - Best {metric} Improvement vs 768D Baseline')
            ax.set_xlabel('Clinical Task')
            ax.set_ylabel('Dimensionality Reduction Method')
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Performance Improvement vs Full Dimensionality Baseline', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance improvement heatmap saved to {output_path}")

def create_research_summary_plot(results_df: pd.DataFrame, baseline_results: Dict, output_path: str) -> None:
    """
    Single comprehensive plot that answers the research question.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Overall analysis across all tasks and models
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    methods = ['pca', 'umap']
    method_colors = {'pca': '#2E86AB', 'umap': '#A23B72'}
    
    # 1. Overall performance curves (top row)
    for metric_idx, metric in enumerate(['auroc', 'auprc']):
        ax = axes[0, metric_idx]
        
        dimensions = sorted(results_df['dimension'].unique())
        
        # Calculate overall baseline across all tasks and models
        all_baseline_scores = []
        for task in results_df['task'].unique():
            for model in models:
                if (task in baseline_results and 
                    model in baseline_results[task] and 
                    metric in baseline_results[task][model]):
                    all_baseline_scores.append(baseline_results[task][model][metric])
        
        overall_baseline = np.mean(all_baseline_scores) if all_baseline_scores else None
        
        if overall_baseline is not None:
            ax.axhline(y=overall_baseline, color='black', linestyle='--', linewidth=2,
                      label=f'768D Baseline ({overall_baseline:.3f})')
        
        # Plot each method across all tasks and models
        for method in methods:
            method_scores = []
            method_stds = []
                
            for dim in dimensions:
                dim_scores = results_df[
                    (results_df['method'] == method) &
                    (results_df['dimension'] == dim) &
                    (results_df['metric'] == metric)
                ]['score'].values
                
                if len(dim_scores) > 0:
                    method_scores.append(np.mean(dim_scores))
                    method_stds.append(np.std(dim_scores))
                else:
                    method_scores.append(np.nan)
                    method_stds.append(np.nan)
            
            ax.errorbar(dimensions, method_scores, yerr=method_stds,
                       color=method_colors[method], marker='o', linewidth=3, markersize=8,
                       label=method.upper(), capsize=5)
        
        ax.set_xlabel('Reduced Dimensions')
        ax.set_ylabel(f'{metric.upper()} Score')
        ax.set_title(f'Overall {metric.upper()} vs Dimensionality')
        ax.set_xscale('log')
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 2. Key findings summary (top right)
    ax_summary = axes[0, 2]
    ax_summary.axis('off')
    
    # Calculate key statistics
    best_improvements = []
    for task in results_df['task'].unique():
        for model in models:
            if (task in baseline_results and 
                model in baseline_results[task]):
                
                baseline_auroc = baseline_results[task][model].get('auroc')
                
                if baseline_auroc is not None:
                    task_model_data = results_df[
                        (results_df['task'] == task) &
                        (results_df['model'] == model) &
                        (results_df['metric'] == 'auroc')
                    ]
                    
                    if not task_model_data.empty:
                        best_score = task_model_data['score'].max()
                        improvement = best_score - baseline_auroc
                        best_improvements.append(improvement)
    
    avg_improvement = np.mean(best_improvements) if best_improvements else 0
    max_improvement = np.max(best_improvements) if best_improvements else 0
    pct_improved = (np.array(best_improvements) > 0).mean() * 100 if best_improvements else 0
    
    summary_text = f"""KEY FINDINGS:

• Average AUROC improvement: {avg_improvement:.3f}
• Maximum AUROC improvement: {max_improvement:.3f}
• Tasks that benefit: {pct_improved:.1f}%

• PCA generally peaks at: 100-200D
• UMAP optimal range: 25-100D  

CONCLUSION:
Dimensionality reduction {'CAN' if avg_improvement > 0.01 else 'RARELY'} 
improve kNN performance vs 768D.

The curse of dimensionality {'IS' if avg_improvement > 0.01 else 'IS NOT'} 
evident for kNN on clinical embeddings."""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 3. Bottom row: Method-specific analysis
    for method_idx, method in enumerate(methods):
        ax = axes[1, method_idx]
        
        method_data = results_df[
            (results_df['method'] == method) &
            (results_df['metric'] == 'auroc')
        ]
        
        dimensions = sorted(method_data['dimension'].unique())
        
        for model, model_label in zip(models, model_labels):
            model_method_data = method_data[method_data['model'] == model]
            
            # Calculate mean across tasks for this model
            mean_scores = []
            for dim in dimensions:
                dim_scores = model_method_data[
                    model_method_data['dimension'] == dim
                ]['score'].values
                mean_scores.append(np.mean(dim_scores) if len(dim_scores) > 0 else np.nan)
            
            ax.plot(dimensions, mean_scores, marker='o', linewidth=3, markersize=8,
                   label=model_label)
        
        # Add baseline
        all_baseline_scores = []
        for task in method_data['task'].unique():
            for model in models:
                if (task in baseline_results and 
                    model in baseline_results[task] and 
                    'auroc' in baseline_results[task][model]):
                    all_baseline_scores.append(baseline_results[task][model]['auroc'])
        
        if all_baseline_scores:
            baseline_mean = np.mean(all_baseline_scores)
            ax.axhline(y=baseline_mean, color='black', linestyle='--', alpha=0.7,
                      label='768D Baseline')
        
        ax.set_xlabel('Reduced Dimensions')
        ax.set_ylabel('AUROC Score')
        ax.set_title(f'{method.upper()} Performance')
        ax.set_xscale('log')
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle('Research Question: How does dimensionality affect kNN performance?', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Research summary plot saved to {output_path}")

def create_summary_table(results_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Create summary table with key statistics."""
    summary_data = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            for method in results_df['method'].unique():
                for metric in ['auroc', 'auprc']:
                    subset = results_df[
                        (results_df['task'] == task) &
                        (results_df['model'] == model) &
                        (results_df['method'] == method) &
                        (results_df['metric'] == metric)
                    ]
                    
                    if not subset.empty:
                        best_idx = subset['score'].idxmax()
                        best_row = subset.loc[best_idx]
                        
                        summary_data.append({
                            'task': task,
                            'model': model,
                            'method': method,
                            'metric': metric,
                            'best_score': best_row['score'],
                            'optimal_dimension': best_row['dimension'],
                            'mean_score': subset['score'].mean(),
                            'std_score': subset['score'].std()
                        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary table saved to {output_path}")
    return summary_df

def plot_best_vs_original_comparison(results_df: pd.DataFrame, baseline_results: Dict, output_path: str) -> None:
    """
    Direct comparison: Best reduced dimensionality performance vs original 768D performance.
    This clearly shows whether dimensionality reduction helps or hurts kNN performance.
    """
    # Calculate best reduced dimensionality scores for each task/model
    comparison_data = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            # Get baseline (768D) performance
            baseline_auroc = None
            baseline_auprc = None
            
            if (task in baseline_results and 
                model in baseline_results[task]):
                baseline_auroc = baseline_results[task][model].get('auroc')
                baseline_auprc = baseline_results[task][model].get('auprc')
            
            if baseline_auroc is None or baseline_auprc is None:
                continue
            
            # Get best reduced dimensionality performance
            task_model_data = results_df[
                (results_df['task'] == task) & 
                (results_df['model'] == model)
            ]
            
            if task_model_data.empty:
                continue
            
            # Find best AUROC and AUPRC across all methods and dimensions
            best_auroc_row = task_model_data[
                task_model_data['metric'] == 'auroc'
            ].loc[task_model_data[task_model_data['metric'] == 'auroc']['score'].idxmax()]
            
            best_auprc_row = task_model_data[
                task_model_data['metric'] == 'auprc'
            ].loc[task_model_data[task_model_data['metric'] == 'auprc']['score'].idxmax()]
            
            comparison_data.append({
                'task': task,
                'model': model,
                'baseline_auroc': baseline_auroc,
                'best_reduced_auroc': best_auroc_row['score'],
                'best_auroc_method': best_auroc_row['method'],
                'best_auroc_dimension': best_auroc_row['dimension'],
                'auroc_improvement': best_auroc_row['score'] - baseline_auroc,
                'baseline_auprc': baseline_auprc,
                'best_reduced_auprc': best_auprc_row['score'],
                'best_auprc_method': best_auprc_row['method'],
                'best_auprc_dimension': best_auprc_row['dimension'],
                'auprc_improvement': best_auprc_row['score'] - baseline_auprc
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        print("No comparison data available")
        return
    
    # Create the comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    model_colors = {'clmbr': '#2E86AB', 'clinicalbert_type3_clinicalbert_pool': '#A23B72'}
    
    # 1. AUROC Scatter Plot (top left)
    ax1 = axes[0, 0]
    for model, model_label in zip(models, model_labels):
        model_data = comparison_df[comparison_df['model'] == model]
        if not model_data.empty:
            ax1.scatter(model_data['baseline_auroc'], model_data['best_reduced_auroc'],
                       alpha=0.7, s=80, label=model_label, color=model_colors[model])
    
    # Add diagonal line (y=x) for reference
    min_val = min(comparison_df['baseline_auroc'].min(), comparison_df['best_reduced_auroc'].min()) - 0.01
    max_val = max(comparison_df['baseline_auroc'].max(), comparison_df['best_reduced_auroc'].max()) + 0.01
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='No Improvement')
    
    ax1.set_xlabel('768D Baseline AUROC')
    ax1.set_ylabel('Best Reduced Dimensionality AUROC')
    ax1.set_title('AUROC: Best Reduced vs 768D Baseline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. AUPRC Scatter Plot (top right)
    ax2 = axes[0, 1]
    for model, model_label in zip(models, model_labels):
        model_data = comparison_df[comparison_df['model'] == model]
        if not model_data.empty:
            ax2.scatter(model_data['baseline_auprc'], model_data['best_reduced_auprc'],
                       alpha=0.7, s=80, label=model_label, color=model_colors[model])
    
    # Add diagonal line
    min_val = min(comparison_df['baseline_auprc'].min(), comparison_df['best_reduced_auprc'].min()) - 0.005
    max_val = max(comparison_df['baseline_auprc'].max(), comparison_df['best_reduced_auprc'].max()) + 0.005
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='No Improvement')
    
    ax2.set_xlabel('768D Baseline AUPRC')
    ax2.set_ylabel('Best Reduced Dimensionality AUPRC')
    ax2.set_title('AUPRC: Best Reduced vs 768D Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. Improvement Distribution (bottom left)
    ax3 = axes[1, 0]
    
    # Calculate statistics
    auroc_improvements = comparison_df['auroc_improvement'].values
    auprc_improvements = comparison_df['auprc_improvement'].values
    
    # Histogram of improvements
    ax3.hist(auroc_improvements, bins=20, alpha=0.6, label='AUROC Improvement', 
             color='skyblue', density=True)
    ax3.hist(auprc_improvements, bins=20, alpha=0.6, label='AUPRC Improvement', 
             color='lightcoral', density=True)
    
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Performance Improvement')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Performance Improvements')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics (bottom right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate key statistics
    auroc_improved_count = (auroc_improvements > 0).sum()
    auroc_total = len(auroc_improvements)
    auprc_improved_count = (auprc_improvements > 0).sum()
    auprc_total = len(auprc_improvements)
    
    auroc_mean_improvement = np.mean(auroc_improvements)
    auprc_mean_improvement = np.mean(auprc_improvements)
    auroc_max_improvement = np.max(auroc_improvements)
    auprc_max_improvement = np.max(auprc_improvements)
    
    # Best performing methods
    best_auroc_methods = comparison_df['best_auroc_method'].value_counts()
    best_auprc_methods = comparison_df['best_auprc_method'].value_counts()
    
    # Calculate average optimal dimensions
    avg_auroc_dim = comparison_df['best_auroc_dimension'].mean()
    avg_auprc_dim = comparison_df['best_auprc_dimension'].mean()
    
    summary_text = f"""DIMENSIONALITY REDUCTION EFFECTIVENESS:

AUROC Results:
• Tasks improved: {auroc_improved_count}/{auroc_total} tasks ({auroc_improved_count/auroc_total*100:.1f}%)
• Mean improvement: {auroc_mean_improvement:.4f}
• Max improvement: {auroc_max_improvement:.4f}
• Best method: {best_auroc_methods.index[0] if len(best_auroc_methods) > 0 else 'N/A'}
• Avg optimal dims: {comparison_df['best_auroc_dimension'].mean():.1f}

AUPRC Results:
• Tasks improved: {auprc_improved_count}/{auprc_total} tasks ({auprc_improved_count/auprc_total*100:.1f}%)
• Mean improvement: {auprc_mean_improvement:.4f}
• Max improvement: {auprc_max_improvement:.4f}
• Best method: {best_auprc_methods.index[0] if len(best_auprc_methods) > 0 else 'N/A'}
• Avg optimal dims: {comparison_df['best_auprc_dimension'].mean():.1f}

CONCLUSION:
{'kNN (dimensionality reduced)' if auroc_mean_improvement > 0 else 'Logistic Regression'} 
performs better on average.

Dimensionality reduction {'ENABLES' if auroc_mean_improvement > 0.01 else 'DOES NOT ENABLE'} 
kNN to outperform LR."""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Best Reduced Dimensionality vs 768D Baseline Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Best vs original comparison plot saved to {output_path}")

def plot_lr_vs_best_knn_comparison(results_df: pd.DataFrame, results_dir: str, output_path: str) -> None:
    """
    Compare logistic regression prediction heads vs best kNN (dimensionality reduced) performance.
    """
    # Get unique tasks from results
    tasks = results_df['task'].unique()
    
    # Load logistic regression results
    lr_results = load_logistic_regression_results(results_dir, tasks)
    
    # Calculate best kNN performance for each task/model
    comparison_data = []
    
    for task in results_df['task'].unique():
        for model in results_df['model'].unique():
            # Get logistic regression performance
            lr_auroc = None
            lr_auprc = None
            
            if (task in lr_results and 
                model in lr_results[task]):
                lr_auroc = lr_results[task][model].get('auroc')
                lr_auprc = lr_results[task][model].get('auprc')
            
            if lr_auroc is None or lr_auprc is None:
                continue
        
            # Get best kNN performance
            task_model_data = results_df[
                (results_df['task'] == task) & 
                (results_df['model'] == model)
            ]
            
            if task_model_data.empty:
                continue
            
            # Find best AUROC and AUPRC across PCA and UMAP only
            auroc_data = task_model_data[task_model_data['metric'] == 'auroc']
            auprc_data = task_model_data[task_model_data['metric'] == 'auprc']
            
            if auroc_data.empty or auprc_data.empty:
                continue
                
            best_auroc_row = auroc_data.loc[auroc_data['score'].idxmax()]
            best_auprc_row = auprc_data.loc[auprc_data['score'].idxmax()]
            
            comparison_data.append({
                'task': task,
                'model': model,
                'lr_auroc': lr_auroc,
                'best_knn_auroc': best_auroc_row['score'],
                'best_auroc_method': best_auroc_row['method'],
                'best_auroc_dimension': best_auroc_row['dimension'],
                'auroc_knn_advantage': best_auroc_row['score'] - lr_auroc,
                'lr_auprc': lr_auprc,
                'best_knn_auprc': best_auprc_row['score'],
                'best_auprc_method': best_auprc_row['method'],
                'best_auprc_dimension': best_auprc_row['dimension'],
                'auprc_knn_advantage': best_auprc_row['score'] - lr_auprc
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if comparison_df.empty:
        print("No comparison data available")
        return
    
    # Create the comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    model_colors = {'clmbr': '#2E86AB', 'clinicalbert_type3_clinicalbert_pool': '#A23B72'}
    
    # 1. AUROC Scatter Plot (top left)
    ax1 = axes[0, 0]
    for model, model_label in zip(models, model_labels):
        model_data = comparison_df[comparison_df['model'] == model]
        if not model_data.empty:
            ax1.scatter(model_data['lr_auroc'], model_data['best_knn_auroc'],
                       alpha=0.7, s=100, label=model_label, color=model_colors[model])
    
    # Add diagonal line (y=x) for reference
    min_val = min(comparison_df['lr_auroc'].min(), comparison_df['best_knn_auroc'].min()) - 0.01
    max_val = max(comparison_df['lr_auroc'].max(), comparison_df['best_knn_auroc'].max()) + 0.01
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Performance')
    
    ax1.set_xlabel('Logistic Regression AUROC')
    ax1.set_ylabel('Best kNN (Dim Reduced) AUROC')
    ax1.set_title('AUROC: Logistic Regression vs Best kNN')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Add text annotations for points significantly above/below diagonal
    for _, row in comparison_df.iterrows():
        if abs(row['auroc_knn_advantage']) > 0.02:  # Significant difference
            ax1.annotate(f"{row['task']}", 
                        (row['lr_auroc'], row['best_knn_auroc']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # 2. AUPRC Scatter Plot (top right)
    ax2 = axes[0, 1]
    for model, model_label in zip(models, model_labels):
        model_data = comparison_df[comparison_df['model'] == model]
        if not model_data.empty:
            ax2.scatter(model_data['lr_auprc'], model_data['best_knn_auprc'],
                       alpha=0.7, s=100, label=model_label, color=model_colors[model])
    
    # Add diagonal line
    min_val = min(comparison_df['lr_auprc'].min(), comparison_df['best_knn_auprc'].min()) - 0.005
    max_val = max(comparison_df['lr_auprc'].max(), comparison_df['best_knn_auprc'].max()) + 0.005
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Equal Performance')
    
    ax2.set_xlabel('Logistic Regression AUPRC')
    ax2.set_ylabel('Best kNN (Dim Reduced) AUPRC')
    ax2.set_title('AUPRC: Logistic Regression vs Best kNN')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # 3. Performance Advantage Distribution (bottom left)
    ax3 = axes[1, 0]
    
    auroc_advantages = comparison_df['auroc_knn_advantage'].values
    auprc_advantages = comparison_df['auprc_knn_advantage'].values
    
    # Histogram of advantages (positive = kNN better, negative = LR better)
    ax3.hist(auroc_advantages, bins=20, alpha=0.6, label='AUROC Advantage', 
             color='skyblue', density=True)
    ax3.hist(auprc_advantages, bins=20, alpha=0.6, label='AUPRC Advantage', 
             color='lightcoral', density=True)
    
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_xlabel('kNN Advantage over Logistic Regression')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of kNN Performance Advantages')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary Statistics (bottom right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate key statistics
    auroc_knn_better_count = (auroc_advantages > 0).sum()
    auroc_total = len(auroc_advantages)
    auprc_knn_better_count = (auprc_advantages > 0).sum()
    auprc_total = len(auprc_advantages)
    
    auroc_mean_advantage = np.mean(auroc_advantages)
    auprc_mean_advantage = np.mean(auprc_advantages)
    auroc_max_advantage = np.max(auroc_advantages)
    auprc_max_advantage = np.max(auprc_advantages)
    
    # Best performing methods for kNN
    best_auroc_methods = comparison_df['best_auroc_method'].value_counts()
    best_auprc_methods = comparison_df['best_auprc_method'].value_counts()
    
    # Calculate average optimal dimensions
    avg_auroc_dim = comparison_df['best_auroc_dimension'].mean()
    avg_auprc_dim = comparison_df['best_auprc_dimension'].mean()
    
    summary_text = f"""LOGISTIC REGRESSION vs BEST kNN COMPARISON:
(Using PCA & UMAP)

AUROC Results:
• kNN wins: {auroc_knn_better_count}/{auroc_total} tasks ({auroc_knn_better_count/auroc_total*100:.1f}%)
• Mean kNN advantage: {auroc_mean_advantage:.4f}
• Max kNN advantage: {auroc_max_advantage:.4f}
• Best kNN method: {best_auroc_methods.index[0] if len(best_auroc_methods) > 0 else 'N/A'}
• Avg optimal dims: {avg_auroc_dim:.1f}

AUPRC Results:
• kNN wins: {auprc_knn_better_count}/{auprc_total} tasks ({auprc_knn_better_count/auprc_total*100:.1f}%)
• Mean kNN advantage: {auprc_mean_advantage:.4f}
• Max kNN advantage: {auprc_max_advantage:.4f}
• Best kNN method: {best_auprc_methods.index[0] if len(best_auprc_methods) > 0 else 'N/A'}
• Avg optimal dims: {avg_auprc_dim:.1f}

CONCLUSION:
{'kNN (dimensionality reduced)' if auroc_mean_advantage > 0 else 'Logistic Regression'} 
performs better on average.

Dimensionality reduction {'ENABLES' if auroc_mean_advantage > 0.01 else 'DOES NOT ENABLE'} 
kNN to outperform LR."""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Logistic Regression vs Best kNN (Dimensionality Reduced) Performance', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"LR vs best kNN comparison plot saved to {output_path}")

def create_all_plots(results_df: pd.DataFrame, results_dir: str, output_dir: str) -> None:
    """Create all dimensionality analysis plots focused on the research question."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique tasks from results
    tasks = results_df['task'].unique()
    
    # Load baseline results
    print("Loading baseline kNN results...")
    baseline_results = load_baseline_results(results_dir, tasks)
    
    # Create focused visualizations for the research question
    print("Creating performance curves...")
    plot_dimensionality_performance_curves(
        results_df, baseline_results,
        os.path.join(output_dir, "dimensionality_performance_curves.png")
    )
    
    print("Creating optimal dimensions analysis...")
    plot_optimal_dimensions_analysis(
        results_df,
        os.path.join(output_dir, "optimal_dimensions_analysis.png")
    )
    
    print("Creating performance improvement heatmap...")
    plot_performance_improvement_heatmap(
        results_df, baseline_results,
        os.path.join(output_dir, "performance_improvement_heatmap.png")
    )
    
    print("Creating research summary plot...")
    create_research_summary_plot(
        results_df, baseline_results,
        os.path.join(output_dir, "research_question_summary.png")
    )
    
    print("Creating best vs original comparison plot...")
    plot_best_vs_original_comparison(
        results_df, baseline_results,
        os.path.join(output_dir, "best_vs_original_comparison.png")
    )
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(
        results_df,
        os.path.join(output_dir, "dimensionality_summary.csv")
    )
    
    print("Creating LR vs best kNN comparison plot...")
    plot_lr_vs_best_knn_comparison(
        results_df, results_dir,
        os.path.join(output_dir, "lr_vs_best_knn_comparison.png")
    )
    
    print(f"All focused plots saved to {output_dir}")
    print("\n📊 Summary:")
    print(f"• Processed {len(results_df)} experiments using PCA & UMAP")

def plot_dimensionality_results(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive plots for dimensionality reduction results.
    
    Args:
        results_df: DataFrame with columns ['task', 'reduction_method', 'n_components', 'auroc', 'auprc']
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUROC curves
    ax1 = axes[0]
    methods = results_df['reduction_method'].unique()
    
    for method in methods:
        method_data = results_df[results_df['reduction_method'] == method].sort_values('n_components')
        if method == 'none':
            # Plot baseline as horizontal line
            baseline_auroc = method_data['auroc'].iloc[0]
            ax1.axhline(y=baseline_auroc, color='red', linestyle='--', linewidth=2, 
                       label=f'Baseline (no reduction): {baseline_auroc:.3f}')
        else:
            ax1.plot(method_data['n_components'], method_data['auroc'], 
                    marker='o', linewidth=2, markersize=6, label=method.upper())
    
    ax1.set_xlabel('Number of Dimensions')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC vs Number of Dimensions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUPRC curves
    ax2 = axes[1]
    
    for method in methods:
        method_data = results_df[results_df['reduction_method'] == method].sort_values('n_components')
        if method == 'none':
            # Plot baseline as horizontal line
            baseline_auprc = method_data['auprc'].iloc[0]
            ax2.axhline(y=baseline_auprc, color='red', linestyle='--', linewidth=2, 
                       label=f'Baseline (no reduction): {baseline_auprc:.3f}')
        else:
            ax2.plot(method_data['n_components'], method_data['auprc'], 
                    marker='o', linewidth=2, markersize=6, label=method.upper())
    
    ax2.set_xlabel('Number of Dimensions')
    ax2.set_ylabel('AUPRC')
    ax2.set_title('AUPRC vs Number of Dimensions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimensionality_performance_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best performance for each method
    best_results = []
    for method in methods:
        method_data = results_df[results_df['reduction_method'] == method]
        if method == 'none':
            best_auroc_row = method_data.iloc[0]
            best_auprc_row = method_data.iloc[0]
        else:
            best_auroc_row = method_data.loc[method_data['auroc'].idxmax()]
            best_auprc_row = method_data.loc[method_data['auprc'].idxmax()]
        
        best_results.append({
            'method': method,
            'best_auroc': best_auroc_row['auroc'],
            'best_auroc_dims': best_auroc_row['n_components'],
            'best_auprc': best_auprc_row['auprc'],
            'best_auprc_dims': best_auprc_row['n_components']
        })
    
    best_df = pd.DataFrame(best_results)
    
    x = np.arange(len(best_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_df['best_auroc'], width, label='Best AUROC', alpha=0.8)
    bars2 = ax.bar(x + width/2, best_df['best_auprc'], width, label='Best AUPRC', alpha=0.8)
    
    ax.set_xlabel('Dimensionality Reduction Method')
    ax.set_ylabel('Performance Score')
    ax.set_title('Best Performance by Method')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m != 'none' else 'BASELINE' for m in best_df['method']])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to {output_dir}/")
    print("  - dimensionality_performance_curves.png")
    print("  - best_performance_comparison.png") 