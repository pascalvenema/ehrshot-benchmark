#!/usr/bin/env python3
"""
Standalone Dimensionality Reduction Plotting Module

This module generates clean dimensionality plots independently from the analysis pipeline.
It reads pre-computed dimensionality reduction results and creates publication-ready visualizations.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

try:
    from ..utils import LABELING_FUNCTION_2_PAPER_NAME
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import LABELING_FUNCTION_2_PAPER_NAME

def load_baseline_results(results_dir: str, tasks: List[str], models: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load baseline kNN results from existing CSV files.
    
    Returns:
        Dict[task][model][metric] = score
    """
    baseline_results = {}
    
    for task in tasks:
        csv_path = os.path.join(results_dir, f"{task}/all_results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: No baseline results file found for {task}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Filter for kNN results with full data (k=-1)
        knn_baseline = df[
            (df['head'] == 'knn') & 
            (df['k'] == -1) &
            (df['replicate'] == 0)  # Use first replicate
        ]
        
        task_results = {}
        
        for model in models:
            model_data = knn_baseline[knn_baseline['model'] == model]
            if not model_data.empty:
                task_results[model] = {}
                for _, row in model_data.iterrows():
                    task_results[model][row['score']] = row['value']
        
        baseline_results[task] = task_results
    
    return baseline_results

def create_clean_dimensionality_plots(results_df: pd.DataFrame, 
                                    baseline_results: Dict, 
                                    output_dir: str,
                                    include_baseline: bool = True) -> None:
    """Create clean dimensionality plots with all models and methods on the same plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating clean dimensionality plots...")
    
    # Create separate plots for AUROC and AUPRC
    for metric in ['auroc', 'auprc']:
        
        # Create single figure for this metric - all lines on same plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
        model_labels = ['CLMBR', 'ClinicalBERT Type3']
        methods = ['pca', 'umap']  # Exclude t-SNE for clean plots
        
        # Define colors for each model-method combination
        colors = {
            ('clmbr', 'pca'): '#2E86AB',      # Blue for CLMBR PCA
            ('clmbr', 'umap'): '#A23B72',     # Purple for CLMBR UMAP
            ('clinicalbert_type3_clinicalbert_pool', 'pca'): '#F18F01',   # Orange for ClinicalBERT PCA
            ('clinicalbert_type3_clinicalbert_pool', 'umap'): '#C73E1D'   # Red for ClinicalBERT UMAP
        }
        
        # Get all dimensions across both models
        all_dimensions = set()
        for model in models:
            model_data = results_df[
                (results_df['model'] == model) &
                (results_df['metric'] == metric)
            ]
            if not model_data.empty:
                all_dimensions.update(model_data['dimension'].unique())
        
        dimensions = sorted(list(all_dimensions))
        
        # Plot each model-method combination
        for model, model_label in zip(models, model_labels):
            for method in methods:
                # Filter data for this model, method, and metric
                method_data = results_df[
                    (results_df['model'] == model) &
                    (results_df['method'] == method) &
                    (results_df['metric'] == metric)
                ]
                
                if not method_data.empty:
                    # Calculate mean performance across tasks for each dimension
                    grouped = method_data.groupby('dimension')['score'].agg(['mean', 'std']).reset_index()
                    
                    color = colors.get((model, method), 'gray')
                    label = f'{model_label} {method.upper()}'
                    
                    ax.errorbar(grouped['dimension'], grouped['mean'], yerr=grouped['std'],
                               color=color, marker='o', linewidth=3, markersize=8,
                               label=label, capsize=5, capthick=2, alpha=0.9)
        
        # Add baselines if available and requested
        if include_baseline and baseline_results:
            for model, model_label in zip(models, model_labels):
                all_baseline_scores = []
                
                # Get baseline scores for this model across all tasks
                for task in results_df[results_df['model'] == model]['task'].unique():
                    if (task in baseline_results and 
                        model in baseline_results[task] and 
                        metric in baseline_results[task][model]):
                        all_baseline_scores.append(baseline_results[task][model][metric])
                
                if all_baseline_scores:
                    baseline_mean = np.mean(all_baseline_scores)
                    baseline_color = '#2E86AB' if model == 'clmbr' else '#F18F01'  # Match model colors
                    ax.axhline(y=baseline_mean, color=baseline_color, linestyle='--', 
                              linewidth=2, alpha=0.7, 
                              label=f'{model_label} 768D Baseline ({baseline_mean:.3f})')
        
        # Customize plot
        ax.set_xlabel('Reduced Dimensions', fontsize=14)
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=14)
        ax.set_title(f'Dimensionality Reduction Performance - {metric.upper()}', 
                    fontsize=16, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(d) for d in dimensions])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Make the plot look more polished
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout()
        
        # Save with the specific filename
        output_file = os.path.join(output_dir, f"clean_{metric}_vs_dimensions_separated.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: clean_{metric}_vs_dimensions_separated.png")

def main():
    parser = argparse.ArgumentParser(description="Generate clean dimensionality reduction plots")
    parser.add_argument("--results_csv", required=True, 
                       help="Path to dimensionality results CSV file")
    parser.add_argument("--output_dir", required=True, 
                       help="Directory to save plots")
    parser.add_argument("--baseline_dir", 
                       help="Directory containing baseline results (optional)")
    parser.add_argument("--no_baseline", action="store_true",
                       help="Skip baseline comparison lines")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_csv}")
    results_df = pd.read_csv(args.results_csv)
    
    # Get unique tasks and models from the data
    tasks = results_df['task'].unique().tolist()
    models = results_df['model'].unique().tolist()
    
    # Load baseline results if provided
    baseline_results = {}
    include_baseline = not args.no_baseline
    if include_baseline and args.baseline_dir:
        print(f"Loading baseline results from {args.baseline_dir}")
        baseline_results = load_baseline_results(args.baseline_dir, tasks, models)
    else:
        include_baseline = False
    
    # Create clean separated plots (the main ones requested)
    create_clean_dimensionality_plots(results_df, baseline_results, args.output_dir, include_baseline)
    
    print("✅ Dimensionality plotting completed successfully!")

if __name__ == "__main__":
    main() 