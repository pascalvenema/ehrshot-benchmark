#!/usr/bin/env python3
"""
Generate comparison plots for CLMBR vs ClinicalBERT Type 3 (clinicalbert_pool) across all prediction heads.
This script creates a 2×3 grid showing CLMBR (top row) vs ClinicalBERT Type 3 (bottom row) across task groups.
"""

import os
import sys
import argparse
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the current directory to the path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    MODEL_2_INFO,
    HEAD_2_INFO,
)

def plot_all_heads_comparison(df_results: pd.DataFrame, 
                            score: str, 
                            path_to_output_dir: str):
    """Compare CLMBR vs ClinicalBERT Type 3 (clinicalbert_pool) with 2x3 grid layout"""
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    n_groups = len(task_groups)
    
    # Create a 2x3 layout: top row = CLMBR, bottom row = ClinicalBERT Type 3
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Get shared prediction heads between CLMBR and ClinicalBERT Type 3
    clmbr_heads = set(MODEL_2_INFO['clmbr']['heads'])
    cb_type3_heads = set(MODEL_2_INFO['clinicalbert_type3_clinicalbert_pool']['heads'])
    shared_heads = sorted(list(clmbr_heads.intersection(cb_type3_heads)))
    
    print(f"📊 Comparing across shared prediction heads: {shared_heads}")
    
    # Define colors for each head (consistent across both models)
    head_colors = {
        'lr_lbfgs': '#1f77b4',  # blue
        'knn': '#ff7f0e',       # orange  
        'rf': '#2ca02c',        # green
        'gbm': '#d62728'        # red
    }
    
    # Define markers for each head
    head_markers = {
        'lr_lbfgs': 's',  # square
        'knn': '^',       # triangle up
        'rf': 'o',        # circle
        'gbm': 'D'        # diamond
    }
    
    # Models to plot
    models = [
        ('clmbr', 'CLMBR'),
        ('clinicalbert_type3_clinicalbert_pool', 'ClinicalBERT Type 3')
    ]
    
    for model_idx, (model_name, model_label) in enumerate(models):
        for task_idx, task_group in enumerate(task_groups):
            ax = axes[model_idx, task_idx]
            labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
            
            # Get task-specific data
            task_data = df_results[
                (df_results['score'] == score) & 
                (df_results['labeling_function'].isin(labeling_functions))
            ]
            
            # Get all `k` shots tested and handle "All" data point
            ks = sorted(task_data['k'].unique().tolist())
            x_tick_labels = [str(k) for k in ks]
            if -1 in ks:
                ks.remove(-1)
                full_data_k = 2 * max(ks) if ks else 256  # fallback if no few-shot data
                ks.append(full_data_k)
                x_tick_labels = [str(k) if k != full_data_k else 'All' for k in ks]
                # Update the data to use fake k value for plotting
                task_data = task_data.copy()
                task_data.loc[task_data['k'] == -1, 'k'] = full_data_k
            else:
                full_data_k = None
            
            # Plot each head for this model
            for head in shared_heads:
                # Get data for this model-head combination
                model_head_data = task_data[
                    (task_data['model'] == model_name) &
                    (task_data['head'] == head)
                ]
                
                if not model_head_data.empty:
                    grouped = model_head_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                    
                    # Create label for legend
                    head_label = HEAD_2_INFO[head]['label']
                    
                    ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                               color=head_colors[head], 
                               linestyle='-',
                               marker=head_markers[head], 
                               label=head_label,
                               linewidth=2.5, markersize=7, alpha=0.9)
            
            # Customize subplot
            ax.set_xlabel('# of Train Examples per Class', fontsize=12)
            ax.set_ylabel(f'{score.upper()}', fontsize=12)
            
            # Set title: model name for first row, task group for all
            if model_idx == 0:  # Top row
                ax.set_title(f'{model_label}\n{TASK_GROUP_2_PAPER_NAME[task_group]}', 
                           fontsize=14, fontweight='bold')
            else:  # Bottom row
                ax.set_title(f'{model_label}\n{TASK_GROUP_2_PAPER_NAME[task_group]}', 
                           fontsize=14, fontweight='bold')
            
            ax.set_xscale('log')
            ax.set_xticks(ks)
            ax.set_xticklabels(x_tick_labels)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            # Make the plot look more polished
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save to output directory
    output_file = os.path.join(path_to_output_dir, f"clmbr_vs_clinicalbert_type3_by_model_{score}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"✅ Saved: {output_file}")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for CLMBR vs ClinicalBERT Type 3 with 2x3 layout")
    parser.add_argument("--path_to_results_dir", required=True, type=str, 
                       help="Path to directory containing results (e.g., EHRSHOT_ASSETS/results)")
    parser.add_argument("--path_to_output_dir", required=True, type=str, 
                       help="Path to directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Load all results
    print("Loading results...")
    dfs: List[pd.DataFrame] = []
    for labeling_function in tqdm(LABELING_FUNCTION_2_PAPER_NAME.keys()):
        path_to_csv = os.path.join(args.path_to_results_dir, f"{labeling_function}/all_results.csv")
        if not os.path.exists(path_to_csv): 
            print(f"⚠️  Skipping {labeling_function} - file not found")
            continue
        dfs.append(pd.read_csv(path_to_csv))
    
    if not dfs:
        print("❌ No results files found!")
        return
    
    df_results = pd.concat(dfs, ignore_index=True)
    print(f"📊 Loaded {len(df_results)} result rows")
    
    # Check what models and heads are available
    available_models = df_results['model'].unique()
    print(f"📋 Available models: {sorted(available_models)}")
    
    clmbr_data = df_results[df_results['model'] == 'clmbr']
    cb_type3_data = df_results[df_results['model'] == 'clinicalbert_type3_clinicalbert_pool']
    
    if clmbr_data.empty:
        print("⚠️  No CLMBR data found!")
    else:
        print(f"📋 CLMBR heads available: {sorted(clmbr_data['head'].unique())}")
    
    if cb_type3_data.empty:
        print("⚠️  No ClinicalBERT Type 3 (clinicalbert_pool) data found!")
    else:
        print(f"📋 ClinicalBERT Type 3 heads available: {sorted(cb_type3_data['head'].unique())}")
    
    # Generate plots for both AUROC and AUPRC
    print("\n🎨 Generating 2×3 comparison plots (models by task groups)...")
    
    for score in ['auroc', 'auprc']:
        if score in df_results['score'].unique():
            print(f"  📈 Creating {score.upper()} plot...")
            plot_all_heads_comparison(df_results, score, args.path_to_output_dir)
        else:
            print(f"  ⚠️  {score.upper()} not found in results")
    
    print(f"\n✅ All plots saved to: {args.path_to_output_dir}")

if __name__ == "__main__":
    main() 