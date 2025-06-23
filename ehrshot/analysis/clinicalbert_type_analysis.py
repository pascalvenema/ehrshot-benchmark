#!/usr/bin/env python3
"""
ClinicalBERT Type Analysis: Examining the Impact of Text Formatting on Performance

Research Question: To what extent does manually transforming structured EHR data 
into more human-readable text improve ClinicalBERT performance on clinical prediction tasks?

Type 1: Raw tabular data with clinical codes
Type 2: Codes replaced with descriptions from medical ontologies  
Type 3: Data aggregated by day and converted to natural sentences using logic rules
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Not needed
from typing import List, Dict, Tuple
from tqdm import tqdm

# Import from the main plotting script
import sys
sys.path.append('.')
from importlib import import_module
plotting_module = import_module('8_make_results_plots')
LABELING_FUNCTION_2_PAPER_NAME = plotting_module.LABELING_FUNCTION_2_PAPER_NAME
TASK_GROUP_2_LABELING_FUNCTION = plotting_module.TASK_GROUP_2_LABELING_FUNCTION
TASK_GROUP_2_PAPER_NAME = plotting_module.TASK_GROUP_2_PAPER_NAME

def analyze_clinicalbert_types(df_results: pd.DataFrame, score: str = 'auroc'):
    """Analyze performance differences between ClinicalBERT types"""
    
    print(f"\n🔬 CLINICALBERT TYPE ANALYSIS: Impact of Text Formatting ({score.upper()})")
    print("=" * 80)
    print("Type 1: Raw codes | Type 2: Code descriptions | Type 3: Natural sentences")
    print("-" * 80)
    
    # Get ClinicalBERT models by type
    cb_models = [m for m in df_results['model'].unique() if 'clinicalbert' in m]
    type1_models = [m for m in cb_models if 'type1' in m]
    type2_models = [m for m in cb_models if 'type2' in m]
    type3_models = [m for m in cb_models if 'type3' in m]
    
    results = {}
    
    # Analyze each task
    for task in sorted(df_results['labeling_function'].unique()):
        task_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['k'] == -1)  # Full data
        ]
        
        if task_data.empty:
            continue
            
        # Calculate average performance for each type across all pooling strategies
        type1_perf = task_data[task_data['model'].isin(type1_models)]['value'].mean()
        type2_perf = task_data[task_data['model'].isin(type2_models)]['value'].mean()
        type3_perf = task_data[task_data['model'].isin(type3_models)]['value'].mean()
        
        # Calculate best performance for each type
        type1_best = task_data[task_data['model'].isin(type1_models)]['value'].max() if not task_data[task_data['model'].isin(type1_models)].empty else np.nan
        type2_best = task_data[task_data['model'].isin(type2_models)]['value'].max() if not task_data[task_data['model'].isin(type2_models)].empty else np.nan
        type3_best = task_data[task_data['model'].isin(type3_models)]['value'].max() if not task_data[task_data['model'].isin(type3_models)].empty else np.nan
        
        results[task] = {
            'type1_avg': type1_perf,
            'type2_avg': type2_perf,
            'type3_avg': type3_perf,
            'type1_best': type1_best,
            'type2_best': type2_best,
            'type3_best': type3_best,
            'type2_vs_type1_avg': type2_perf - type1_perf,
            'type3_vs_type1_avg': type3_perf - type1_perf,
            'type3_vs_type2_avg': type3_perf - type2_perf,
            'type2_vs_type1_best': type2_best - type1_best,
            'type3_vs_type1_best': type3_best - type1_best,
            'type3_vs_type2_best': type3_best - type2_best
        }
        
        task_name = LABELING_FUNCTION_2_PAPER_NAME.get(task, task)[:18]
        print(f"{task_name:<20} T1:{type1_perf:>6.3f} T2:{type2_perf:>6.3f} T3:{type3_perf:>6.3f} "
              f"Δ(2-1):{type2_perf-type1_perf:>+6.3f} Δ(3-1):{type3_perf-type1_perf:>+6.3f}")
    
    # Summary statistics
    if results:
        type2_improvements = [r['type2_vs_type1_avg'] for r in results.values() if not np.isnan(r['type2_vs_type1_avg'])]
        type3_improvements = [r['type3_vs_type1_avg'] for r in results.values() if not np.isnan(r['type3_vs_type1_avg'])]
        type3_vs_type2 = [r['type3_vs_type2_avg'] for r in results.values() if not np.isnan(r['type3_vs_type2_avg'])]
        
        print(f"\nSummary (Average Performance):")
        print(f"Type 2 vs Type 1 improvement:")
        print(f"  Mean: {np.mean(type2_improvements):+.3f}")
        print(f"  Tasks improved: {sum(1 for x in type2_improvements if x > 0)}/{len(type2_improvements)} ({100*sum(1 for x in type2_improvements if x > 0)/len(type2_improvements):.1f}%)")
        
        print(f"Type 3 vs Type 1 improvement:")
        print(f"  Mean: {np.mean(type3_improvements):+.3f}")  
        print(f"  Tasks improved: {sum(1 for x in type3_improvements if x > 0)}/{len(type3_improvements)} ({100*sum(1 for x in type3_improvements if x > 0)/len(type3_improvements):.1f}%)")
        
        print(f"Type 3 vs Type 2 improvement:")
        print(f"  Mean: {np.mean(type3_vs_type2):+.3f}")
        print(f"  Tasks improved: {sum(1 for x in type3_vs_type2 if x > 0)}/{len(type3_vs_type2)} ({100*sum(1 for x in type3_vs_type2 if x > 0)/len(type3_vs_type2):.1f}%)")
    
    return results

def plot_clinicalbert_type_comparison(df_results: pd.DataFrame, 
                                    score: str, 
                                    path_to_output_dir: str):
    """Create comprehensive plots comparing ClinicalBERT types"""
    
    # Plot 1: Learning curves by type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    colors = {'type1': '#d62728', 'type2': '#ff7f0e', 'type3': '#2ca02c'}
    type_names = {'type1': 'Type 1 (Raw Codes)', 'type2': 'Type 2 (Descriptions)', 'type3': 'Type 3 (Sentences)'}
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        for cb_type in ['type1', 'type2', 'type3']:
            type_models = [m for m in df_results['model'].unique() if f'clinicalbert_{cb_type}' in m]
            
            type_data = df_results[
                (df_results['score'] == score) & 
                (df_results['labeling_function'].isin(labeling_functions)) &
                (df_results['model'].isin(type_models))
            ]
            
            if not type_data.empty:
                grouped = type_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                           color=colors[cb_type], label=type_names[cb_type],
                           linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=10)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    fig.suptitle(f'ClinicalBERT Types Learning Curves - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_types_learning_curves_{score}.png"), dpi=300)
    plt.close()
    
    # Plot 2: Task-by-task performance comparison
    valid_tasks = []
    for task in df_results['labeling_function'].unique():
        task_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['k'] == -1)
        ]
        cb_data = task_data[task_data['model'].str.contains('clinicalbert', na=False)]
        if not cb_data.empty:
            valid_tasks.append(task)
    
    n_tasks = len(valid_tasks)
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, task in enumerate(valid_tasks):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        task_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['k'] == -1)
        ]
        
        type_perfs = []
        type_labels = []
        type_colors = []
        
        for cb_type in ['type1', 'type2', 'type3']:
            type_models = [m for m in task_data['model'].unique() if f'clinicalbert_{cb_type}' in m]
            type_data = task_data[task_data['model'].isin(type_models)]
            
            if not type_data.empty:
                # Use best performance for each type
                best_perf = type_data['value'].max()
                type_perfs.append(best_perf)
                type_labels.append(type_names[cb_type])
                type_colors.append(colors[cb_type])
        
        if type_perfs:
            bars = ax.bar(range(len(type_perfs)), type_perfs, color=type_colors, alpha=0.7)
            
            # Add value labels on bars
            for i, (bar, perf) in enumerate(zip(bars, type_perfs)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{perf:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(range(len(type_labels)))
        ax.set_xticklabels([label.split('(')[0].strip() for label in type_labels], rotation=45)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(LABELING_FUNCTION_2_PAPER_NAME.get(task, task), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_tasks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'ClinicalBERT Types Performance by Task - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_types_task_comparison_{score}.png"), dpi=300)
    plt.close()
    
    # Plot 3: Improvement analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate improvements for each task
    improvements_data = []
    
    for task in valid_tasks:
        task_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['k'] == -1)
        ]
        
        type1_perf = task_data[task_data['model'].str.contains('type1', na=False)]['value'].mean()
        type2_perf = task_data[task_data['model'].str.contains('type2', na=False)]['value'].mean()
        type3_perf = task_data[task_data['model'].str.contains('type3', na=False)]['value'].mean()
        
        if not (np.isnan(type1_perf) or np.isnan(type2_perf) or np.isnan(type3_perf)):
            improvements_data.append({
                'Task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
                'Type 2 vs Type 1': type2_perf - type1_perf,
                'Type 3 vs Type 1': type3_perf - type1_perf,
                'Type 3 vs Type 2': type3_perf - type2_perf
            })
    
    if improvements_data:
        improvements_df = pd.DataFrame(improvements_data)
        
        # Box plot of improvements
        improvement_cols = ['Type 2 vs Type 1', 'Type 3 vs Type 1', 'Type 3 vs Type 2']
        bp = ax1.boxplot([improvements_df[col].values for col in improvement_cols], 
                        labels=['T2 vs T1', 'T3 vs T1', 'T3 vs T2'], patch_artist=True)
        
        box_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel(f'{score.upper()} Improvement', fontsize=12)
        ax1.set_title('Distribution of Performance Improvements', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of Type 3 vs Type 1 improvements
        ax2.hist(improvements_df['Type 3 vs Type 1'], bins=10, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.axvline(x=improvements_df['Type 3 vs Type 1'].mean(), color='blue', linestyle='-', alpha=0.8, linewidth=2)
        ax2.set_xlabel(f'{score.upper()} Improvement (Type 3 vs Type 1)', fontsize=12)
        ax2.set_ylabel('Number of Tasks', fontsize=12)
        ax2.set_title('Distribution of Type 3 vs Type 1 Improvements', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_imp = improvements_df['Type 3 vs Type 1'].mean()
        positive_tasks = sum(improvements_df['Type 3 vs Type 1'] > 0)
        total_tasks = len(improvements_df)
        
        ax2.text(0.05, 0.95, f'Mean: {mean_imp:+.3f}\nPositive: {positive_tasks}/{total_tasks} ({100*positive_tasks/total_tasks:.1f}%)', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    fig.suptitle(f'ClinicalBERT Type Improvement Analysis - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_types_improvement_analysis_{score}.png"), dpi=300)
    plt.close()
    
    return fig

def create_clinicalbert_type_summary_table(df_results: pd.DataFrame, output_dir: str):
    """Create detailed CSV table for ClinicalBERT type analysis"""
    
    results = []
    
    for score in ['auroc', 'auprc']:
        for task in sorted(df_results['labeling_function'].unique()):
            task_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)
            ]
            
            if task_data.empty:
                continue
            
            # Get performance for each type
            type_results = {}
            for cb_type in ['type1', 'type2', 'type3']:
                type_models = [m for m in task_data['model'].unique() if f'clinicalbert_{cb_type}' in m]
                type_data = task_data[task_data['model'].isin(type_models)]
                
                if not type_data.empty:
                    type_results[f'{cb_type}_mean'] = type_data['value'].mean()
                    type_results[f'{cb_type}_std'] = type_data['value'].std()
                    type_results[f'{cb_type}_best'] = type_data['value'].max()
                    type_results[f'{cb_type}_worst'] = type_data['value'].min()
                    
                    # Best model for this type
                    best_idx = type_data['value'].idxmax()
                    type_results[f'{cb_type}_best_model'] = type_data.loc[best_idx, 'model']
                else:
                    for suffix in ['_mean', '_std', '_best', '_worst', '_best_model']:
                        type_results[f'{cb_type}{suffix}'] = np.nan
            
            if any(not np.isnan(v) for k, v in type_results.items() if isinstance(v, (int, float))):
                results.append({
                    'task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
                    'score': score.upper(),
                    **type_results,
                    'type2_vs_type1_mean': type_results['type2_mean'] - type_results['type1_mean'],
                    'type3_vs_type1_mean': type_results['type3_mean'] - type_results['type1_mean'],
                    'type3_vs_type2_mean': type_results['type3_mean'] - type_results['type2_mean'],
                    'type2_vs_type1_best': type_results['type2_best'] - type_results['type1_best'],
                    'type3_vs_type1_best': type_results['type3_best'] - type_results['type1_best'],
                    'type3_vs_type2_best': type_results['type3_best'] - type_results['type2_best']
                })
    
    # Save table
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'clinicalbert_type_analysis_detailed.csv'), index=False)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="ClinicalBERT Type Analysis")
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
    for labeling_function in tqdm(LABELING_FUNCTION_2_PAPER_NAME.keys()):
        path_to_csv = os.path.join(args.path_to_results_dir, f"{labeling_function}/all_results.csv")
        if not os.path.exists(path_to_csv): 
            continue
        dfs.append(pd.read_csv(path_to_csv))
    
    if not dfs:
        print("No results files found!")
        return
    
    df_results = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df_results)} result rows")
    
    # Run analyses for both metrics
    for score in ['auroc', 'auprc']:
        if score in df_results['score'].unique():
            analyze_clinicalbert_types(df_results, score)
            plot_clinicalbert_type_comparison(df_results, score, args.path_to_output_dir)
    
    # Create detailed table
    results_df = create_clinicalbert_type_summary_table(df_results, args.path_to_output_dir)
    
    print(f"\n✅ ClinicalBERT Type Analysis complete! Results saved to {args.path_to_output_dir}")
    print(f"📊 Files created:")
    print(f"  - clinicalbert_types_learning_curves_auroc.png")
    print(f"  - clinicalbert_types_learning_curves_auprc.png")
    print(f"  - clinicalbert_types_task_comparison_auroc.png")
    print(f"  - clinicalbert_types_task_comparison_auprc.png")
    print(f"  - clinicalbert_types_improvement_analysis_auroc.png")
    print(f"  - clinicalbert_types_improvement_analysis_auprc.png")
    print(f"  - clinicalbert_type_analysis_detailed.csv")

if __name__ == "__main__":
    main() 