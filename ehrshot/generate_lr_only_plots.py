#!/usr/bin/env python3
"""
Generate LR-only embedding comparison plots.
This script creates plots similar to embedding_comparison_shared_heads but focuses only on 
the logistic regression prediction head with an optimized layout.
"""

import os
import sys
import argparse
from typing import List
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
)

def plot_embedding_comparison_lr_only(df_results: pd.DataFrame, 
                                    score: str, 
                                    path_to_output_dir: str):
    """Compare CLMBR vs ClinicalBERT embeddings using only LR prediction head with optimized layout"""
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    n_groups = len(task_groups)
    
    # Create a more efficient layout for 3 task groups
    if n_groups == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        # Fallback to square layout for other numbers
        n_cols = int(np.ceil(np.sqrt(n_groups)))
        n_rows = int(np.ceil(n_groups / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        axes = axes.flatten()
    
    # Only use LR prediction head
    lr_head = 'lr_lbfgs'
    
    # Define colors and styles
    clmbr_style = {'color': 'blue', 'linestyle': '-', 'marker': 's', 'label': 'CLMBR + LR'}
    cb_style = {'color': 'lightcoral', 'linestyle': '-', 'marker': '^', 'label': 'ClinicalBERT Type3 + LR'}
    
    for idx, task_group in enumerate(task_groups):
        ax = axes[idx] if n_groups > 1 else axes
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Get task-specific data to determine k values
        task_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions))
        ]
        
        # Get all `k` shots tested and handle "All" data point like in plot.py
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
        
        # Plot CLMBR performance with LR head
        clmbr_data = task_data[
            (task_data['model'] == 'clmbr') &
            (task_data['head'] == lr_head)
        ]
        
        if not clmbr_data.empty:
            clmbr_grouped = clmbr_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            ax.errorbar(clmbr_grouped['k'], clmbr_grouped['mean'], yerr=clmbr_grouped['std'], 
                       color=clmbr_style['color'], 
                       linestyle=clmbr_style['linestyle'],
                       marker=clmbr_style['marker'], 
                       label=clmbr_style['label'],
                       linewidth=2.5, markersize=8, alpha=0.9)
        
        # Plot ClinicalBERT Type3 with clinicalbert pooling performance with LR head
        cb_data = task_data[
            (task_data['model'] == 'clinicalbert_type3_clinicalbert_pool') &
            (task_data['head'] == lr_head)
        ]
        
        if not cb_data.empty:
            cb_grouped = cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(cb_grouped['k'], cb_grouped['mean'], yerr=cb_grouped['std'], 
                       color=cb_style['color'],
                       linestyle=cb_style['linestyle'],
                       marker=cb_style['marker'],
                       label=cb_style['label'],
                       linewidth=2.5, markersize=8, alpha=0.9)
        
        ax.set_xlabel('# of Train Examples per Class', fontsize=12)
        ax.set_ylabel(f'{score.upper()}', fontsize=12)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=14, fontweight='bold')
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
    
    # Hide unused subplots if any
    if n_groups > 1 and hasattr(axes, '__len__'):
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
    
    # Remove title as requested
    plt.tight_layout()
    
    # Save to output directory
    output_file = os.path.join(path_to_output_dir, f"embedding_comparison_lr_only_{score}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"✅ Saved: {output_file}")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate LR-only embedding comparison plots")
    parser.add_argument("--path_to_results_dir", required=True, type=str, 
                       help="Path to directory containing results (e.g., EHRSHOT_ASSETS/results)")
    parser.add_argument("--path_to_output_dir", required=True, type=str, 
                       help="Path to directory to save plots")
    
    args = parser.parse_args()
    
    # Create clinicalbert subdirectory (consistent with existing plot organization)
    clinicalbert_dir = os.path.join(args.path_to_output_dir, 'clinicalbert')
    os.makedirs(clinicalbert_dir, exist_ok=True)
    
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
    
    # Generate plots for both AUROC and AUPRC
    print("\n🎨 Generating LR-only embedding comparison plots...")
    
    for score in ['auroc', 'auprc']:
        if score in df_results['score'].unique():
            print(f"  📈 Creating {score.upper()} plot...")
            plot_embedding_comparison_lr_only(df_results, score, clinicalbert_dir)
        else:
            print(f"  ⚠️  {score.upper()} not found in results")
    
    print(f"\n✅ All plots saved to: {clinicalbert_dir}")

if __name__ == "__main__":
    main() 