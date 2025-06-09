#!/usr/bin/env python3
"""
ClinicalBERT Analysis Script
Focused analysis and visualization of ClinicalBERT results in the EHRSHOT benchmark.
"""

import os
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    MODEL_2_INFO,
)

def analyze_clinicalbert_coverage(df_results: pd.DataFrame):
    """Analyze the coverage of ClinicalBERT models across tasks"""
    print("🧠 CLINICALBERT COVERAGE ANALYSIS")
    print("=" * 50)
    
    # Get all ClinicalBERT models
    cb_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    print(f"Total ClinicalBERT models found: {len(cb_models)}")
    print(f"Models: {sorted(cb_models)[:5]}{'...' if len(cb_models) > 5 else ''}")
    
    # Analyze by type
    type1_models = [m for m in cb_models if 'type1' in m]
    type2_models = [m for m in cb_models if 'type2' in m]
    type3_models = [m for m in cb_models if 'type3' in m]
    
    print(f"\nType 1 models: {len(type1_models)}")
    print(f"Type 2 models: {len(type2_models)}")
    print(f"Type 3 models: {len(type3_models)}")
    
    # Analyze coverage by task
    tasks = df_results['labeling_function'].unique()
    for task in sorted(tasks):
        task_cb_models = df_results[
            (df_results['labeling_function'] == task) & 
            (df_results['model'].str.contains('clinicalbert'))
        ]['model'].unique()
        
        print(f"\n{LABELING_FUNCTION_2_PAPER_NAME.get(task, task)}: {len(task_cb_models)} CB models")

def create_clinicalbert_summary_table(df_results: pd.DataFrame, score: str = 'auroc'):
    """Create summary table of best ClinicalBERT performance"""
    print(f"\n📊 CLINICALBERT PERFORMANCE SUMMARY ({score.upper()})")
    print("=" * 80)
    
    # Filter for ClinicalBERT and full data
    cb_data = df_results[
        (df_results['model'].str.contains('clinicalbert')) &
        (df_results['score'] == score) &
        (df_results['k'] == -1)
    ]
    
    if cb_data.empty:
        print("No ClinicalBERT data found")
        return
    
    # Group by task and find best performance
    summary_data = []
    
    for task in sorted(cb_data['labeling_function'].unique()):
        task_data = cb_data[cb_data['labeling_function'] == task]
        best_perf = task_data.groupby(['model', 'head'])['value'].mean()
        best_combo = best_perf.idxmax()
        best_score = best_perf.max()
        
        # Get baseline performance for comparison
        baseline_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['k'] == -1) &
            (df_results['model'].isin(['clmbr', 'count']))
        ]
        
        if not baseline_data.empty:
            best_baseline = baseline_data.groupby(['model', 'head'])['value'].mean().max()
            improvement = best_score - best_baseline
        else:
            best_baseline = 0
            improvement = 0
        
        summary_data.append({
            'Task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
            'Best CB Model': f"{best_combo[0].replace('clinicalbert_', '')}+{best_combo[1]}",
            'CB Score': f"{best_score:.3f}",
            'Best Baseline': f"{best_baseline:.3f}",
            'Improvement': f"{improvement:+.3f}"
        })
    
    # Create DataFrame and print
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    return summary_df

def plot_clinicalbert_type_performance_heatmap(df_results: pd.DataFrame, 
                                              score: str, 
                                              path_to_output_dir: str):
    """Create heatmap showing performance of different ClinicalBERT types across tasks"""
    cb_data = df_results[
        (df_results['model'].str.contains('clinicalbert')) &
        (df_results['score'] == score) &
        (df_results['k'] == -1)
    ]
    
    if cb_data.empty:
        print(f"No ClinicalBERT data for {score}")
        return
    
    # Create matrix of performance by type and task
    tasks = sorted(cb_data['labeling_function'].unique())
    cb_types = ['type1', 'type2', 'type3']
    
    performance_matrix = np.zeros((len(cb_types), len(tasks)))
    
    for i, cb_type in enumerate(cb_types):
        for j, task in enumerate(tasks):
            type_data = cb_data[
                (cb_data['labeling_function'] == task) &
                (cb_data['model'].str.contains(cb_type))
            ]
            
            if not type_data.empty:
                # Get best performance for this type on this task
                best_perf = type_data.groupby(['model', 'head'])['value'].mean().max()
                performance_matrix[i, j] = best_perf
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create labels
    task_labels = [LABELING_FUNCTION_2_PAPER_NAME.get(task, task) for task in tasks]
    type_labels = [f'ClinicalBERT {t.upper()}' for t in cb_types]
    
    im = ax.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(cb_types)))
    ax.set_xticklabels(task_labels, rotation=45, ha='right')
    ax.set_yticklabels(type_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(f'{score.upper()} Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(cb_types)):
        for j in range(len(tasks)):
            if performance_matrix[i, j] > 0:
                text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                              ha="center", va="center", color="white", fontweight='bold')
    
    ax.set_title(f'ClinicalBERT Performance Heatmap - {score.upper()}', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_heatmap_{score}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="ClinicalBERT Analysis and Visualization")
    parser.add_argument("--path_to_results_dir", required=True, type=str, 
                       help="Path to directory containing results")
    parser.add_argument("--path_to_output_dir", required=True, type=str, 
                       help="Path to save analysis outputs")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Load all results
    print("Loading results...")
    dfs: List[pd.DataFrame] = []
    for idx, labeling_function in tqdm(enumerate(LABELING_FUNCTION_2_PAPER_NAME.keys())):
        path_to_csv = os.path.join(args.path_to_results_dir, f"{labeling_function}/all_results.csv")
        if not os.path.exists(path_to_csv): 
            print(f"Skipping {labeling_function} - no results file")
            continue
        dfs.append(pd.read_csv(path_to_csv))
    
    if not dfs:
        print("No results files found!")
        return
    
    df_results = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_results)} result rows")
    
    # Run analyses
    analyze_clinicalbert_coverage(df_results)
    
    for score in ['auroc', 'auprc']:
        if score in df_results['score'].unique():
            summary_df = create_clinicalbert_summary_table(df_results, score)
            if summary_df is not None:
                summary_df.to_csv(os.path.join(args.path_to_output_dir, f"clinicalbert_summary_{score}.csv"), 
                                 index=False)
            
            plot_clinicalbert_type_performance_heatmap(df_results, score, args.path_to_output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.path_to_output_dir}")

if __name__ == "__main__":
    main() 