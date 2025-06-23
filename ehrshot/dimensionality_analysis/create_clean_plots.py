#!/usr/bin/env python3
"""
Create clean dimensionality plots with separated metrics.
Generates the specific plots: clean_auprc_vs_dimensions_separated.png and clean_auroc_vs_dimensions_separated.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

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
    
    # Create plots
    create_clean_dimensionality_plots(results_df, baseline_results, args.output_dir)
    
    print("✅ Clean dimensionality plots created successfully!")

if __name__ == "__main__":
    main() 