#!/usr/bin/env python3
"""
Statistical Analysis for Dimensionality Reduction

This script performs statistical tests to answer:
- Does dimensionality reduction significantly affect kNN performance?
- Which dimensionality reduction methods perform best?
- What are the optimal dimensions for each method?
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import argparse
from loguru import logger

def load_baseline_knn_results(results_dir: str, tasks: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Load baseline kNN results (full 768D) from main results.
    
    Returns:
        Dict[task][model] = {'auroc': score, 'auprc': score}
    """
    baseline_results = {}
    
    for task in tasks:
        csv_path = os.path.join(results_dir, f"{task}/all_results.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"No results file found for {task} at {csv_path}")
            continue
            
        try:
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
                auroc_row = clmbr_data[clmbr_data['score'] == 'auroc']
                auprc_row = clmbr_data[clmbr_data['score'] == 'auprc']
                if not auroc_row.empty and not auprc_row.empty:
                    auroc = auroc_row['value'].iloc[0]
                    auprc = auprc_row['value'].iloc[0]
                    task_results['clmbr'] = {'auroc': auroc, 'auprc': auprc}
            
            # Get ClinicalBERT type3 clinicalbert_pool kNN baseline
            cb_data = knn_baseline[knn_baseline['model'] == 'clinicalbert_type3_clinicalbert_pool']
            if not cb_data.empty:
                auroc_row = cb_data[cb_data['score'] == 'auroc']
                auprc_row = cb_data[cb_data['score'] == 'auprc']
                if not auroc_row.empty and not auprc_row.empty:
                    auroc = auroc_row['value'].iloc[0]
                    auprc = auprc_row['value'].iloc[0]
                    task_results['clinicalbert_type3_clinicalbert_pool'] = {'auroc': auroc, 'auprc': auprc}
            
            if task_results:
                baseline_results[task] = task_results
                
        except Exception as e:
            logger.error(f"Error loading baseline results for {task}: {e}")
            continue
    
    return baseline_results

def test_dimensionality_effect(df_dim: pd.DataFrame, baseline_results: Dict) -> Dict:
    """
    Test if dimensionality reduction significantly affects performance.
    Uses paired t-tests comparing best reduced dimension vs full dimension baseline.
    """
    print("\n🔬 RESEARCH QUESTION: Does dimensionality reduction significantly affect kNN performance?")
    print("=" * 80)
    
    results = {}
    
    for metric in ['auroc', 'auprc']:
        print(f"\n--- {metric.upper()} Analysis ---")
        
        metric_results = {}
        
        for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
            model_label = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
            print(f"\n{model_label}:")
            
            # Get dimensionality reduction results for this model
            model_data = df_dim[
                (df_dim['model'] == model) & 
                (df_dim['metric'] == metric)
            ]
            
            if model_data.empty:
                continue
            
            # Find best performance for each task (across all methods and dimensions)
            task_best_scores = []
            task_baseline_scores = []
            task_improvements = []
            
            for task in model_data['task'].unique():
                task_data = model_data[model_data['task'] == task]
                
                # Get best reduced dimension score
                best_score = task_data['score'].max()
                best_row = task_data[task_data['score'] == best_score].iloc[0]
                
                # Get baseline score
                if (task in baseline_results and 
                    model in baseline_results[task] and 
                    metric in baseline_results[task][model]):
                    baseline_score = baseline_results[task][model][metric]
                    
                    task_best_scores.append(best_score)
                    task_baseline_scores.append(baseline_score)
                    task_improvements.append(best_score - baseline_score)
                    
                    print(f"  {task}: Best={best_score:.3f} ({best_row['method']} {best_row['dimension']}D) vs Baseline={baseline_score:.3f} "
                          f"(Δ={best_score-baseline_score:+.3f})")
            
            if len(task_improvements) > 1:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(task_best_scores, task_baseline_scores)
                
                # Effect size (Cohen's d for paired samples)
                mean_diff = np.mean(task_improvements)
                std_diff = np.std(task_improvements, ddof=1)
                cohens_d = mean_diff / std_diff if std_diff > 0 else 0
                
                # Summary statistics
                n_improvements = sum(1 for imp in task_improvements if imp > 0)
                
                print(f"\n  Statistical Test Results:")
                print(f"    Mean improvement: {mean_diff:+.3f}")
                print(f"    Tasks with improvement: {n_improvements}/{len(task_improvements)} ({100*n_improvements/len(task_improvements):.1f}%)")
                print(f"    Paired t-test: t={t_stat:.3f}, p={p_value:.3f}")
                print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
                
                # Interpretation
                if p_value < 0.05:
                    direction = "improves" if mean_diff > 0 else "reduces"
                    print(f"    ✓ Dimensionality reduction significantly {direction} performance")
                else:
                    print(f"    ✗ No significant effect of dimensionality reduction")
                
                metric_results[model] = {
                    'mean_improvement': mean_diff,
                    'n_improved': n_improvements,
                    'n_total': len(task_improvements),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'improvements': task_improvements
                }
        
        results[metric] = metric_results
    
    return results

def test_method_comparison(df_dim: pd.DataFrame) -> Dict:
    """
    Compare dimensionality reduction methods using Kruskal-Wallis and Mann-Whitney U tests.
    """
    print("\n🔬 RESEARCH QUESTION: Which dimensionality reduction method performs best?")
    print("=" * 80)
    
    results = {}
    
    for metric in ['auroc', 'auprc']:
        print(f"\n--- {metric.upper()} Method Comparison ---")
        
        metric_results = {}
        
        for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
            model_label = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
            print(f"\n{model_label}:")
            
            model_data = df_dim[
                (df_dim['model'] == model) & 
                (df_dim['metric'] == metric)
            ]
            
            if model_data.empty:
                continue
            
            # Get scores for each method
            methods = model_data['method'].unique()
            method_scores = {}
            
            for method in methods:
                method_data = model_data[model_data['method'] == method]
                method_scores[method] = method_data['score'].values
                print(f"  {method.upper()}: {len(method_data)} results, mean={np.mean(method_data['score']):.3f} ± {np.std(method_data['score']):.3f}")
            
            # Kruskal-Wallis test (non-parametric ANOVA)
            if len(methods) > 2:
                try:
                    kw_stat, kw_p = stats.kruskal(*[method_scores[m] for m in methods])
                    print(f"\n  Kruskal-Wallis test: H={kw_stat:.3f}, p={kw_p:.3f}")
                    
                    if kw_p < 0.05:
                        print(f"  ✓ Significant difference between methods")
                        
                        # Pairwise comparisons with Mann-Whitney U
                        print(f"\n  Pairwise comparisons (Mann-Whitney U):")
                        for i, method1 in enumerate(methods):
                            for j, method2 in enumerate(methods):
                                if i < j:
                                    u_stat, u_p = stats.mannwhitneyu(
                                        method_scores[method1], 
                                        method_scores[method2], 
                                        alternative='two-sided'
                                    )
                                    mean1 = np.mean(method_scores[method1])
                                    mean2 = np.mean(method_scores[method2])
                                    diff = mean2 - mean1
                                    
                                    significance = "**" if u_p < 0.01 else "*" if u_p < 0.05 else ""
                                    print(f"    {method1.upper()} vs {method2.upper()}: "
                                          f"Δ={diff:+.3f}, p={u_p:.3f} {significance}")
                    else:
                        print(f"  ✗ No significant difference between methods")
                        
                except Exception as e:
                    print(f"  Error in statistical test: {e}")
            
            metric_results[model] = {
                'methods': methods,
                'method_scores': method_scores,
                'method_means': {m: np.mean(scores) for m, scores in method_scores.items()}
            }
        
        results[metric] = metric_results
    
    return results

def analyze_optimal_dimensions(df_dim: pd.DataFrame) -> Dict:
    """
    Analyze optimal dimensions for each method.
    """
    print("\n🔬 RESEARCH QUESTION: What are the optimal dimensions for each method?")
    print("=" * 80)
    
    results = {}
    
    for metric in ['auroc', 'auprc']:
        print(f"\n--- {metric.upper()} Optimal Dimensions ---")
        
        metric_results = {}
        
        for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
            model_label = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
            print(f"\n{model_label}:")
            
            model_data = df_dim[
                (df_dim['model'] == model) & 
                (df_dim['metric'] == metric)
            ]
            
            if model_data.empty:
                continue
            
            method_optimal = {}
            
            for method in model_data['method'].unique():
                method_data = model_data[model_data['method'] == method]
                
                # Find optimal dimension by mean performance across tasks
                dim_performance = method_data.groupby('dimension')['score'].agg(['mean', 'std', 'count']).reset_index()
                optimal_dim = dim_performance.loc[dim_performance['mean'].idxmax(), 'dimension']
                optimal_score = dim_performance.loc[dim_performance['mean'].idxmax(), 'mean']
                
                method_optimal[method] = {
                    'optimal_dimension': optimal_dim,
                    'optimal_score': optimal_score,
                    'dimension_performance': dim_performance
                }
                
                print(f"  {method.upper()}: Optimal dimension = {optimal_dim}D (score = {optimal_score:.3f})")
                
                # Show performance curve summary
                print(f"    Dimension performance:")
                for _, row in dim_performance.iterrows():
                    print(f"      {int(row['dimension'])}D: {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count'])})")
            
            metric_results[model] = method_optimal
        
        results[metric] = metric_results
    
    return results

def create_summary_report(
    dimensionality_effects: Dict,
    method_comparisons: Dict,
    optimal_dimensions: Dict,
    output_dir: str
) -> None:
    """
    Create a summary report of all statistical analyses.
    """
    summary_path = os.path.join(output_dir, 'dimensionality_statistical_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("DIMENSIONALITY REDUCTION STATISTICAL ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("RESEARCH QUESTIONS:\n")
        f.write("1. Does dimensionality reduction significantly affect kNN performance?\n")
        f.write("2. Which dimensionality reduction method performs best?\n")
        f.write("3. What are the optimal dimensions for each method?\n\n")
        
        # Summary of dimensionality effects
        f.write("KEY FINDINGS:\n")
        f.write("-" * 30 + "\n\n")
        
        for metric in ['auroc', 'auprc']:
            f.write(f"{metric.upper()} Results:\n")
            
            if metric in dimensionality_effects:
                for model, results in dimensionality_effects[metric].items():
                    model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
                    improvement = results['mean_improvement']
                    p_value = results['p_value']
                    n_improved = results['n_improved']
                    n_total = results['n_total']
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    
                    f.write(f"  {model_name}: Mean improvement = {improvement:+.3f}, "
                           f"p = {p_value:.3f}{significance}, "
                           f"{n_improved}/{n_total} tasks improved\n")
            
            f.write("\n")
        
        # Optimal dimensions summary
        f.write("OPTIMAL DIMENSIONS:\n")
        f.write("-" * 30 + "\n")
        
        for metric in ['auroc', 'auprc']:
            f.write(f"\n{metric.upper()}:\n")
            
            if metric in optimal_dimensions:
                for model, methods in optimal_dimensions[metric].items():
                    model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
                    f.write(f"  {model_name}:\n")
                    
                    for method, results in methods.items():
                        optimal_dim = results['optimal_dimension']
                        optimal_score = results['optimal_score']
                        f.write(f"    {method.upper()}: {optimal_dim}D (score = {optimal_score:.3f})\n")
        
    print(f"\n📊 Summary report saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Statistical analysis for dimensionality reduction results')
    parser.add_argument('--dimensionality_csv', required=True, 
                       help='Path to dimensionality results CSV file')
    parser.add_argument('--results_dir', required=True,
                       help='Path to main results directory (for baseline kNN results)')
    parser.add_argument('--output_dir', default='.',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load dimensionality reduction results
    print(f"Loading dimensionality results from: {args.dimensionality_csv}")
    df_dim = pd.read_csv(args.dimensionality_csv)
    
    print(f"Data shape: {df_dim.shape}")
    print(f"Tasks: {sorted(df_dim['task'].unique())}")
    print(f"Models: {sorted(df_dim['model'].unique())}")
    print(f"Methods: {sorted(df_dim['method'].unique())}")
    print(f"Dimensions: {sorted(df_dim['dimension'].unique())}")
    
    # Load baseline kNN results for comparison
    tasks = df_dim['task'].unique()
    baseline_results = load_baseline_knn_results(args.results_dir, tasks)
    print(f"Loaded baseline results for {len(baseline_results)} tasks")
    
    # Run statistical analyses
    print("\n" + "="*80)
    print("STARTING STATISTICAL ANALYSIS")
    print("="*80)
    
    dimensionality_effects = test_dimensionality_effect(df_dim, baseline_results)
    method_comparisons = test_method_comparison(df_dim)
    optimal_dimensions = analyze_optimal_dimensions(df_dim)
    
    # Create summary report
    create_summary_report(
        dimensionality_effects,
        method_comparisons,
        optimal_dimensions,
        args.output_dir
    )
    
    print("\n✅ Statistical analysis completed!")

if __name__ == "__main__":
    main() 