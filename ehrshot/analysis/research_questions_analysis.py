#!/usr/bin/env python3
"""
Research Questions Analysis Script

This script addresses specific research questions:
1. How does CLMBR compare to ClinicalBERT when using shared prediction heads?
2. Do non-linear heads improve performance over linear heads for embeddings?
"""

import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
)

# Import from the plotting script
import sys
sys.path.append('.')
from importlib import import_module
plotting_module = import_module('8_make_results_plots')
plot_embedding_comparison_shared_heads = plotting_module.plot_embedding_comparison_shared_heads
plot_detailed_task_embedding_comparison = plotting_module.plot_detailed_task_embedding_comparison

def analyze_embedding_comparison(df_results: pd.DataFrame, score: str = 'auroc'):
    """Analyze CLMBR vs ClinicalBERT using shared prediction heads"""
    print(f"\n🔬 RESEARCH QUESTION 1: CLMBR vs ClinicalBERT Embedding Comparison ({score.upper()})")
    print("=" * 80)
    
    shared_heads = ['lr_lbfgs', 'knn']
    
    results = {}
    
    for task in sorted(df_results['labeling_function'].unique()):
        # CLMBR performance with shared heads
        clmbr_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['model'] == 'clmbr') &
            (df_results['head'].isin(shared_heads)) &
            (df_results['k'] == -1)
        ]
        
        # ClinicalBERT performance with shared heads
        cb_models = [m for m in df_results['model'].unique() if 'clinicalbert' in m]
        cb_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['model'].isin(cb_models)) &
            (df_results['head'].isin(shared_heads)) &
            (df_results['k'] == -1)
        ]
        
        if not clmbr_data.empty and not cb_data.empty:
            clmbr_perf = clmbr_data['value'].mean()
            cb_best_perf = cb_data.groupby(['model', 'head'])['value'].mean().max()
            cb_avg_perf = cb_data['value'].mean()
            
            results[task] = {
                'clmbr': clmbr_perf,
                'cb_best': cb_best_perf,
                'cb_avg': cb_avg_perf,
                'diff_best': cb_best_perf - clmbr_perf,
                'diff_avg': cb_avg_perf - clmbr_perf
            }
    
    # Print results table
    print(f"{'Task':<20} {'CLMBR':<8} {'CB Best':<8} {'CB Avg':<8} {'Diff Best':<10} {'Diff Avg':<10}")
    print("-" * 80)
    
    for task, data in results.items():
        task_name = LABELING_FUNCTION_2_PAPER_NAME.get(task, task)[:18]
        print(f"{task_name:<20} {data['clmbr']:<8.3f} {data['cb_best']:<8.3f} {data['cb_avg']:<8.3f} "
              f"{data['diff_best']:+<10.3f} {data['diff_avg']:+<10.3f}")
    
    # Summary statistics
    if results:
        clmbr_wins_best = sum(1 for d in results.values() if d['diff_best'] < 0)
        clmbr_wins_avg = sum(1 for d in results.values() if d['diff_avg'] < 0)
        
        print(f"\nSummary:")
        print(f"Tasks where CLMBR > Best ClinicalBERT: {clmbr_wins_best}/{len(results)} ({100*clmbr_wins_best/len(results):.1f}%)")
        print(f"Tasks where CLMBR > Avg ClinicalBERT: {clmbr_wins_avg}/{len(results)} ({100*clmbr_wins_avg/len(results):.1f}%)")
        print(f"Mean performance difference (CLMBR - CB Best): {np.mean([d['diff_best'] for d in results.values()]):+.3f}")
        print(f"Mean performance difference (CLMBR - CB Avg): {np.mean([d['diff_avg'] for d in results.values()]):+.3f}")
    
    return results

def analyze_head_types(df_results: pd.DataFrame, score: str = 'auroc'):
    """Analyze linear vs non-linear prediction heads"""
    print(f"\n🔬 RESEARCH QUESTION 2: Linear vs Non-Linear Prediction Heads ({score.upper()})")
    print("=" * 80)
    
    linear_heads = ['lr_lbfgs']
    nonlinear_heads = ['knn', 'rf', 'gbm']
    
    results = {}
    
    # Analyze each model type
    for model_type in ['clmbr', 'count']:
        model_results = {}
        
        for task in sorted(df_results['labeling_function'].unique()):
            task_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['score'] == score) &
                (df_results['model'] == model_type) &
                (df_results['k'] == -1)
            ]
            
            if not task_data.empty:
                linear_perf = task_data[task_data['head'].isin(linear_heads)]['value'].mean()
                nonlinear_perf = task_data[task_data['head'].isin(nonlinear_heads)]['value'].mean()
                
                if not np.isnan(linear_perf) and not np.isnan(nonlinear_perf):
                    model_results[task] = {
                        'linear': linear_perf,
                        'nonlinear': nonlinear_perf,
                        'improvement': nonlinear_perf - linear_perf
                    }
        
        results[model_type] = model_results
    
    # Analyze ClinicalBERT models
    cb_models = [m for m in df_results['model'].unique() if 'clinicalbert' in m]
    cb_results = {}
    
    for task in sorted(df_results['labeling_function'].unique()):
        cb_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['score'] == score) &
            (df_results['model'].isin(cb_models)) &
            (df_results['k'] == -1)
        ]
        
        if not cb_data.empty:
            linear_perf = cb_data[cb_data['head'].isin(linear_heads)]['value'].mean()
            nonlinear_perf = cb_data[cb_data['head'].isin(nonlinear_heads)]['value'].mean()
            
            if not np.isnan(linear_perf) and not np.isnan(nonlinear_perf):
                cb_results[task] = {
                    'linear': linear_perf,
                    'nonlinear': nonlinear_perf,
                    'improvement': nonlinear_perf - linear_perf
                }
    
    results['clinicalbert'] = cb_results
    
    # Print results
    print(f"{'Task':<20} {'Model':<12} {'Linear':<8} {'NonLinear':<8} {'Improvement':<12}")
    print("-" * 80)
    
    for model_type, model_results in results.items():
        for task, data in model_results.items():
            task_name = LABELING_FUNCTION_2_PAPER_NAME.get(task, task)[:18]
            print(f"{task_name:<20} {model_type.upper():<12} {data['linear']:<8.3f} "
                  f"{data['nonlinear']:<8.3f} {data['improvement']:+<12.3f}")
    
    # Summary statistics
    print(f"\nSummary by Model Type:")
    for model_type, model_results in results.items():
        if model_results:
            improvements = [d['improvement'] for d in model_results.values()]
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            mean_improvement = np.mean(improvements)
            
            print(f"{model_type.upper()}:")
            print(f"  Mean improvement: {mean_improvement:+.3f}")
            print(f"  Tasks with improvement: {positive_improvements}/{len(improvements)} "
                  f"({100*positive_improvements/len(improvements):.1f}%)")
    
    return results

def create_detailed_comparison_table(df_results: pd.DataFrame, output_dir: str):
    """Create detailed CSV tables for the research questions"""
    
    # Research Question 1: Embedding Comparison
    print("\n📊 Creating detailed comparison tables...")
    
    embedding_results = []
    shared_heads = ['lr_lbfgs', 'knn']
    
    for score in ['auroc', 'auprc']:
        for task in sorted(df_results['labeling_function'].unique()):
            # CLMBR data
            clmbr_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['score'] == score) &
                (df_results['model'] == 'clmbr') &
                (df_results['head'].isin(shared_heads)) &
                (df_results['k'] == -1)
            ]
            
            # ClinicalBERT data
            cb_models = [m for m in df_results['model'].unique() if 'clinicalbert' in m]
            cb_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['score'] == score) &
                (df_results['model'].isin(cb_models)) &
                (df_results['head'].isin(shared_heads)) &
                (df_results['k'] == -1)
            ]
            
            if not clmbr_data.empty and not cb_data.empty:
                clmbr_perf = clmbr_data['value'].mean()
                clmbr_std = clmbr_data['value'].std()
                
                # Best CB model
                cb_best = cb_data.groupby(['model', 'head'])['value'].mean()
                best_combo = cb_best.idxmax()
                best_perf = cb_best.max()
                best_std = cb_data[
                    (cb_data['model'] == best_combo[0]) & 
                    (cb_data['head'] == best_combo[1])
                ]['value'].std()
                
                # Average CB performance
                cb_avg_perf = cb_data['value'].mean()
                cb_avg_std = cb_data['value'].std()
                
                embedding_results.append({
                    'task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
                    'score': score.upper(),
                    'clmbr_mean': clmbr_perf,
                    'clmbr_std': clmbr_std,
                    'cb_best_model': f"{best_combo[0]}+{best_combo[1]}",
                    'cb_best_mean': best_perf,
                    'cb_best_std': best_std,
                    'cb_avg_mean': cb_avg_perf,
                    'cb_avg_std': cb_avg_std,
                    'clmbr_vs_best': clmbr_perf - best_perf,
                    'clmbr_vs_avg': clmbr_perf - cb_avg_perf
                })
    
    # Save embedding comparison table
    embedding_df = pd.DataFrame(embedding_results)
    embedding_df.to_csv(os.path.join(output_dir, 'embedding_comparison_detailed.csv'), index=False)
    
    # Research Question 2: Head Type Analysis
    head_results = []
    linear_heads = ['lr_lbfgs']
    nonlinear_heads = ['knn', 'rf', 'gbm']
    
    for score in ['auroc', 'auprc']:
        for model_type in ['clmbr', 'count']:
            for task in sorted(df_results['labeling_function'].unique()):
                task_data = df_results[
                    (df_results['labeling_function'] == task) &
                    (df_results['score'] == score) &
                    (df_results['model'] == model_type) &
                    (df_results['k'] == -1)
                ]
                
                if not task_data.empty:
                    linear_data = task_data[task_data['head'].isin(linear_heads)]
                    nonlinear_data = task_data[task_data['head'].isin(nonlinear_heads)]
                    
                    if not linear_data.empty and not nonlinear_data.empty:
                        head_results.append({
                            'task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
                            'score': score.upper(),
                            'model': model_type.upper(),
                            'linear_mean': linear_data['value'].mean(),
                            'linear_std': linear_data['value'].std(),
                            'nonlinear_mean': nonlinear_data['value'].mean(),
                            'nonlinear_std': nonlinear_data['value'].std(),
                            'improvement': nonlinear_data['value'].mean() - linear_data['value'].mean()
                        })
        
        # Add ClinicalBERT analysis
        cb_models = [m for m in df_results['model'].unique() if 'clinicalbert' in m]
        for task in sorted(df_results['labeling_function'].unique()):
            cb_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['score'] == score) &
                (df_results['model'].isin(cb_models)) &
                (df_results['k'] == -1)
            ]
            
            if not cb_data.empty:
                linear_data = cb_data[cb_data['head'].isin(linear_heads)]
                nonlinear_data = cb_data[cb_data['head'].isin(nonlinear_heads)]
                
                if not linear_data.empty and not nonlinear_data.empty:
                    head_results.append({
                        'task': LABELING_FUNCTION_2_PAPER_NAME.get(task, task),
                        'score': score.upper(),
                        'model': 'CLINICALBERT',
                        'linear_mean': linear_data['value'].mean(),
                        'linear_std': linear_data['value'].std(),
                        'nonlinear_mean': nonlinear_data['value'].mean(),
                        'nonlinear_std': nonlinear_data['value'].std(),
                        'improvement': nonlinear_data['value'].mean() - linear_data['value'].mean()
                    })
    
    # Save head type analysis table
    head_df = pd.DataFrame(head_results)
    head_df.to_csv(os.path.join(output_dir, 'head_type_analysis_detailed.csv'), index=False)
    
    return embedding_df, head_df

def main():
    parser = argparse.ArgumentParser(description="Research Questions Analysis")
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
            analyze_embedding_comparison(df_results, score)
            analyze_head_types(df_results, score)
    
    # Create detailed comparison tables
    embedding_df, head_df = create_detailed_comparison_table(df_results, args.path_to_output_dir)
    
    # Generate the updated embedding comparison plots
    print("\n📊 Generating embedding comparison plots...")
    for score in ['auroc', 'auprc']:
        if score in df_results['score'].unique():
            print(f"  - Creating embedding_comparison_shared_heads_{score}.png")
            plot_embedding_comparison_shared_heads(df_results, score, args.path_to_output_dir)
            print(f"  - Creating detailed_task_embedding_comparison_{score}.png")
            plot_detailed_task_embedding_comparison(df_results, score, args.path_to_output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {args.path_to_output_dir}")
    print(f"📊 Files created:")
    print(f"  - embedding_comparison_detailed.csv")
    print(f"  - head_type_analysis_detailed.csv")
    print(f"  - embedding_comparison_shared_heads_auroc.png")
    print(f"  - embedding_comparison_shared_heads_auprc.png")
    print(f"  - detailed_task_embedding_comparison_auroc.png")
    print(f"  - detailed_task_embedding_comparison_auprc.png")

if __name__ == "__main__":
    main() 