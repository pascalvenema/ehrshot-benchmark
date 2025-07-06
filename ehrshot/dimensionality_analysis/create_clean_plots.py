#!/usr/bin/env python3
"""
Create clean dimensionality plots with separated metrics.
Generates the specific plots: clean_auprc_vs_dimensions_separated.png and clean_auroc_vs_dimensions_separated.png
Also generates per-task plots: clean_auroc_vs_dimensions_per_task.png and clean_auprc_vs_dimensions_per_task.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import math

def create_clean_dimensionality_plots(results_df: pd.DataFrame, baseline_results: Dict, output_dir: str) -> None:
    """Create clean dimensionality plots with separated metrics for the user's specific needs."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating clean dimensionality plots...")
    
    # Create separate plots for AUROC and AUPRC
    for metric in ['auroc', 'auprc']:
        
        # Create figure for this metric
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
        model_labels = ['CLMBR', 'ClinicalBERT Type3']
        
        for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
            ax = axes[model_idx]
            
            # Filter data for this model and metric
            model_data = results_df[
                (results_df['model'] == model) &
                (results_df['metric'] == metric)
            ]
            
            if model_data.empty:
                ax.set_title(f'{model_label} - No Data')
                continue
            
            # Get dimensions tested
            dimensions = sorted(model_data['dimension'].unique())
            
            # Plot each reduction method
            methods = ['pca', 'umap']  # Exclude t-SNE for clean plots
            method_colors = {'pca': '#1f77b4', 'umap': '#ff7f0e'}
            
            for method in methods:
                method_data = model_data[model_data['method'] == method]
                
                if not method_data.empty:
                    # Calculate mean performance across tasks for each dimension
                    grouped = method_data.groupby('dimension')['score'].agg(['mean', 'std']).reset_index()
                    
                    ax.plot(grouped['dimension'], grouped['mean'], 
                           color=method_colors[method], label=method.upper(), 
                           linewidth=3, marker='o', markersize=8)
                    
                    # Add confidence bands
                    ax.fill_between(grouped['dimension'], 
                                   grouped['mean'] - grouped['std'], 
                                   grouped['mean'] + grouped['std'],
                                   color=method_colors[method], alpha=0.2)
            
            # Add baseline if available
            if baseline_results:
                all_baseline_scores = []
                for task in model_data['task'].unique():
                    if (task in baseline_results and 
                        model in baseline_results[task] and 
                        metric in baseline_results[task][model]):
                        all_baseline_scores.append(baseline_results[task][model][metric])
                
                if all_baseline_scores:
                    baseline_mean = np.mean(all_baseline_scores)
                    ax.axhline(y=baseline_mean, color='black', linestyle='--', 
                              linewidth=2, alpha=0.7, label='768D Baseline')
            
            # Customize subplot
            ax.set_xlabel('Reduced Dimensions', fontsize=12)
            ax.set_ylabel(f'{metric.upper()}', fontsize=12)
            ax.set_title(f'{model_label} - {metric.upper()} vs Dimensions', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.set_xticks(dimensions)
            ax.set_xticklabels([str(d) for d in dimensions])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.suptitle(f'Dimensionality Reduction Performance - {metric.upper()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save with the specific filename the user wants
        output_file = os.path.join(output_dir, f"clean_{metric}_vs_dimensions_separated.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: clean_{metric}_vs_dimensions_separated.png")

def create_per_task_dimensionality_plots(results_df: pd.DataFrame, baseline_results: Dict, output_dir: str) -> None:
    """Create dimensionality plots showing performance for each individual task."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating per-task dimensionality plots...")
    
    # Get all unique tasks
    tasks = sorted(results_df['task'].unique())
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    model_labels = ['CLMBR', 'ClinicalBERT Type3']
    methods = ['pca', 'umap']
    method_colors = {'pca': '#1f77b4', 'umap': '#ff7f0e'}
    
    # Create separate plots for AUROC and AUPRC
    for metric in ['auroc', 'auprc']:
        # Calculate subplot grid size - we want tasks in rows, models in columns
        n_tasks = len(tasks)
        n_models = len(models)
        
        # Create figure with subplots for each task-model combination
        fig, axes = plt.subplots(n_tasks, n_models, figsize=(8 * n_models, 4 * n_tasks))
        
        # Handle single task case
        if n_tasks == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for task_idx, task in enumerate(tasks):
            for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
                if n_tasks == 1 and n_models == 1:
                    ax = axes
                elif n_tasks == 1:
                    ax = axes[model_idx]
                elif n_models == 1:
                    ax = axes[task_idx]
                else:
                    ax = axes[task_idx, model_idx]
                
                # Filter data for this task, model and metric
                task_model_data = results_df[
                    (results_df['task'] == task) &
                    (results_df['model'] == model) &
                    (results_df['metric'] == metric)
                ]
                
                if task_model_data.empty:
                    ax.set_title(f'{task}\n{model_label} - No Data')
                    ax.set_xlabel('Reduced Dimensions')
                    ax.set_ylabel(f'{metric.upper()}')
                    continue
                
                # Get dimensions tested for this task/model
                dimensions = sorted(task_model_data['dimension'].unique())
                
                # Plot each reduction method
                for method in methods:
                    method_data = task_model_data[task_model_data['method'] == method]
                    
                    if not method_data.empty:
                        # Get scores for each dimension
                        scores = []
                        for dim in dimensions:
                            dim_data = method_data[method_data['dimension'] == dim]
                            if not dim_data.empty:
                                scores.append(dim_data['score'].iloc[0])
                            else:
                                scores.append(np.nan)
                        
                        # Plot the line
                        ax.plot(dimensions, scores, 
                               color=method_colors[method], label=method.upper(), 
                               linewidth=2, marker='o', markersize=6)
                
                # Add baseline if available
                if (baseline_results and task in baseline_results and 
                    model in baseline_results[task] and 
                    metric in baseline_results[task][model]):
                    baseline_score = baseline_results[task][model][metric]
                    ax.axhline(y=baseline_score, color='black', linestyle='--', 
                              linewidth=1.5, alpha=0.7, label='768D Baseline')
                
                # Customize subplot
                ax.set_xlabel('Reduced Dimensions', fontsize=10)
                ax.set_ylabel(f'{metric.upper()}', fontsize=10)
                ax.set_title(f'{task}\n{model_label}', fontsize=11, fontweight='bold')
                ax.set_xscale('log')
                if dimensions:
                    ax.set_xticks(dimensions)
                    ax.set_xticklabels([str(d) for d in dimensions], fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Only show legend on first subplot to avoid clutter
                if task_idx == 0 and model_idx == 0:
                    ax.legend(fontsize=9)
        
        plt.suptitle(f'Per-Task Dimensionality Reduction Performance - {metric.upper()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the per-task plot
        output_file = os.path.join(output_dir, f"clean_{metric}_vs_dimensions_per_task.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: clean_{metric}_vs_dimensions_per_task.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create clean dimensionality plots")
    parser.add_argument("--results_csv", required=True, help="Path to dimensionality results CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    parser.add_argument("--baseline_dir", help="Directory containing baseline results (optional)")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_csv}")
    results_df = pd.read_csv(args.results_csv)
    
    # Load baseline results if provided
    baseline_results = {}
    if args.baseline_dir:
        print(f"Loading baseline results from {args.baseline_dir}")
        # This would need to be implemented based on your baseline results format
        # For now, we'll use an empty dict
    
    # Create averaged plots (existing functionality)
    create_clean_dimensionality_plots(results_df, baseline_results, args.output_dir)
    
    # Create per-task plots (new functionality)
    create_per_task_dimensionality_plots(results_df, baseline_results, args.output_dir)
    
    print("✅ Clean dimensionality plots created successfully!")

if __name__ == "__main__":
    main() 