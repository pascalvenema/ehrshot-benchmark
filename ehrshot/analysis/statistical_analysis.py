#!/usr/bin/env python3
"""
Improved Statistical Significance Analysis for EHRSHOT
Addresses few-shot learning focus and fair comparison issues
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, kruskal, mannwhitneyu, friedmanchisquare
from scipy import stats
from statsmodels.stats.multitest import multipletests
import argparse
import os
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Task name mapping for prettier output
LABELING_FUNCTION_2_PAPER_NAME = {
    'guo_los': 'Long LOS',
    'guo_readmission': '30-day Readmission', 
    'guo_icu': 'ICU Admission',
    'new_pancan': 'Pancreatic Cancer',
    'new_celiac': 'Celiac',
    'new_lupus': 'Lupus',
    'new_acutemi': 'Acute MI',
    'new_hypertension': 'Hypertension',
    'new_hyperlipidemia': 'Hyperlipidemia',
    'lab_thrombocytopenia': 'Thrombocytopenia',
    'lab_hyperkalemia': 'Hyperkalemia',
    'lab_hypoglycemia': 'Hypoglycemia',
    'lab_hyponatremia': 'Hyponatremia',
    'lab_anemia': 'Anemia',
    'chexpert': 'ChexPert'
}

def load_results_data(results_dir: str) -> pd.DataFrame:
    """Load and combine all results files"""
    result_files = glob.glob(os.path.join(results_dir, '*/all_results.csv'))
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    all_data = []
    for file in result_files:
        task = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        df['labeling_function'] = task
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def analyze_rq1_1_kshot_aware(df_results: pd.DataFrame, score: str = 'auroc', alpha: float = 0.05) -> Dict:
    """
    RQ1.1: ClinicalBERT text transformation comparison across ALL k-shot conditions
    """
    print(f"\n🧪 RQ1.1 K-SHOT AWARE ANALYSIS: ClinicalBERT Types ({score.upper()})")
    print("=" * 80)
    
    # Get ClinicalBERT pooling data across all k values
    cb_data = df_results[
        (df_results['score'] == score) & 
        (df_results['model'].str.contains('clinicalbert', na=False)) &
        (df_results['model'].str.endswith('_clinicalbert_pool')) &
        (df_results['head'] == 'lr_lbfgs')
    ].copy()
    
    cb_data['cb_type'] = cb_data['model'].str.extract(r'clinicalbert_(type\d)_')[0]
    cb_data = cb_data.dropna(subset=['cb_type'])
    
    k_values = sorted([k for k in cb_data['k'].unique() if k != -1]) + [-1]
    print(f"Analyzing across k-values: {k_values}")
    print(f"ClinicalBERT types: {sorted(cb_data['cb_type'].unique())}")
    
    results = {
        'by_k': {},
        'overall_across_k': {},
        'k_trend_analysis': {}
    }
    
    # 1. Analysis by each k value
    type_performance_by_k = {k: {} for k in k_values}
    
    for k_val in k_values:
        k_data = cb_data[cb_data['k'] == k_val]
        
        if len(k_data['cb_type'].unique()) < 2:
            continue
            
        print(f"\n📊 k = {k_val} Analysis:")
        
        # Collect scores for each type
        type_groups = []
        type_names = []
        for cb_type in ['type1', 'type2', 'type3']:
            type_scores = k_data[k_data['cb_type'] == cb_type]['value'].values
            if len(type_scores) > 0:
                type_groups.append(type_scores)
                type_names.append(cb_type)
                type_performance_by_k[k_val][cb_type] = np.mean(type_scores)
                print(f"  {cb_type}: mean={np.mean(type_scores):.3f} (n={len(type_scores)})")
                
        if len(type_groups) >= 2:
            # Kruskal-Wallis for this k value
            kw_stat, kw_p = kruskal(*type_groups)
            print(f"  Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.6f}")
            
            # Post-hoc pairwise comparisons
            pairwise_results = {}
            for i, type1 in enumerate(type_names):
                for j, type2 in enumerate(type_names[i+1:], i+1):
                    stat, p_val = mannwhitneyu(type_groups[i], type_groups[j], alternative='two-sided')
                    mean_diff = np.mean(type_groups[j]) - np.mean(type_groups[i])
                    pairwise_results[f"{type1}_vs_{type2}"] = {
                        'mean_diff': mean_diff,
                        'p_value': p_val,
                        'significant': p_val < alpha
                    }
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    print(f"    {type1} vs {type2}: diff={mean_diff:+.3f}, p={p_val:.3f} {sig_marker}")
            
            results['by_k'][k_val] = {
                'kruskal_wallis': {'statistic': kw_stat, 'p_value': kw_p, 'significant': kw_p < alpha},
                'pairwise': pairwise_results
            }
    
    # 2. Trend analysis: Does type advantage change with k?
    print(f"\n📈 Trend Analysis: How does type advantage change with k?")
    
    if len(type_performance_by_k) > 3:
        # Focus on Type3 vs Type1 advantage across k values
        k_vals_for_trend = []
        type3_advantage = []
        
        for k_val in k_values:
            if k_val != -1 and 'type1' in type_performance_by_k[k_val] and 'type3' in type_performance_by_k[k_val]:
                k_vals_for_trend.append(k_val)
                advantage = type_performance_by_k[k_val]['type3'] - type_performance_by_k[k_val]['type1']
                type3_advantage.append(advantage)
        
        if len(k_vals_for_trend) > 3:
            corr, corr_p = stats.pearsonr(k_vals_for_trend, type3_advantage)
            print(f"Type3 vs Type1 advantage correlation with k: r={corr:.3f}, p={corr_p:.3f}")
            
            if corr > 0:
                trend_desc = "Type3 advantage INCREASES with more training data"
            else:
                trend_desc = "Type3 advantage DECREASES with more training data"
            
            print(f"Interpretation: {trend_desc}")
            
            results['k_trend_analysis'] = {
                'correlation': corr,
                'correlation_p': corr_p,
                'interpretation': trend_desc,
                'k_values': k_vals_for_trend,
                'advantages': type3_advantage
            }
    
    # 3. Overall Friedman test across all k values
    print(f"\n🌍 Overall Analysis Across All K-Values (Friedman Test):")
    
    friedman_data = []
    for task in cb_data['labeling_function'].unique():
        task_data = cb_data[cb_data['labeling_function'] == task]
        
        for k_val in k_values:
            k_task_data = task_data[task_data['k'] == k_val]
            
            type1_scores = k_task_data[k_task_data['cb_type'] == 'type1']['value'].values
            type2_scores = k_task_data[k_task_data['cb_type'] == 'type2']['value'].values  
            type3_scores = k_task_data[k_task_data['cb_type'] == 'type3']['value'].values
            
            if len(type1_scores) > 0 and len(type2_scores) > 0 and len(type3_scores) > 0:
                friedman_data.append([
                    np.mean(type1_scores),
                    np.mean(type2_scores), 
                    np.mean(type3_scores)
                ])
    
    if len(friedman_data) > 5:
        friedman_data = np.array(friedman_data)
        friedman_stat, friedman_p = friedmanchisquare(
            friedman_data[:, 0],  # type1
            friedman_data[:, 1],  # type2  
            friedman_data[:, 2]   # type3
        )
        
        print(f"Friedman test (repeated measures): χ²={friedman_stat:.3f}, p={friedman_p:.6f}")
        print(f"Significant: {'Yes' if friedman_p < alpha else 'No'}")
        
        type_means = {
            'type1': np.mean(friedman_data[:, 0]),
            'type2': np.mean(friedman_data[:, 1]), 
            'type3': np.mean(friedman_data[:, 2])
        }
        print(f"Overall means: Type1={type_means['type1']:.3f}, Type2={type_means['type2']:.3f}, Type3={type_means['type3']:.3f}")
        
        results['overall_across_k']['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p,
            'significant': friedman_p < alpha,
            'type_means': type_means
        }
        
    return results

def analyze_rq1_2_kshot_aware(df_results: pd.DataFrame, score: str = 'auroc', alpha: float = 0.05) -> Dict:
    """
    RQ1.2: CLMBR vs ClinicalBERT comparison across ALL k-shot conditions
    """
    print(f"\n🧪 RQ1.2 K-SHOT AWARE ANALYSIS: CLMBR vs ClinicalBERT ({score.upper()})")
    print("=" * 80)
    
    comparison_data = df_results[
        (df_results['score'] == score) & 
        (df_results['head'] == 'lr_lbfgs') &
        (df_results['model'].isin(['clmbr', 'clinicalbert_type3_clinicalbert_pool']))
    ].copy()
    
    k_values = sorted([k for k in comparison_data['k'].unique() if k != -1]) + [-1]
    print(f"Analyzing across k-values: {k_values}")
    
    results = {
        'by_k': {},
        'k_trend_analysis': {},
        'summary': {}
    }
    
    clmbr_advantages = []
    k_vals_for_trend = []
    
    # 1. Analysis by each k value
    for k_val in k_values:
        k_data = comparison_data[comparison_data['k'] == k_val]
        
        clmbr_scores = []
        cb_scores = []
        
        for task in k_data['labeling_function'].unique():
            task_data = k_data[k_data['labeling_function'] == task]
            
            clmbr_task = task_data[task_data['model'] == 'clmbr']['value'].values
            cb_task = task_data[task_data['model'] == 'clinicalbert_type3_clinicalbert_pool']['value'].values
            
            if len(clmbr_task) > 0 and len(cb_task) > 0:
                clmbr_scores.append(np.mean(clmbr_task))
                cb_scores.append(np.mean(cb_task))
        
        if len(clmbr_scores) > 1:
            clmbr_scores = np.array(clmbr_scores)
            cb_scores = np.array(cb_scores)
            
            # Paired tests
            wilcoxon_stat, wilcoxon_p = wilcoxon(clmbr_scores, cb_scores, alternative='two-sided')
            mean_diff = np.mean(clmbr_scores - cb_scores)
            clmbr_wins = np.sum(clmbr_scores > cb_scores)
            
            sig_marker = "***" if wilcoxon_p < 0.001 else "**" if wilcoxon_p < 0.01 else "*" if wilcoxon_p < 0.05 else "ns"
            
            print(f"\nk = {k_val:>3}: Mean diff = {mean_diff:+.3f}, p = {wilcoxon_p:.4f} {sig_marker}, CLMBR wins = {clmbr_wins}/{len(clmbr_scores)}")
            
            results['by_k'][k_val] = {
                'mean_diff': mean_diff,
                'wilcoxon_p': wilcoxon_p,
                'clmbr_wins': int(clmbr_wins),
                'total_tasks': len(clmbr_scores),
                'significant': wilcoxon_p < alpha
            }
            
            # Store for trend analysis
            if k_val != -1:
                k_vals_for_trend.append(k_val)
                clmbr_advantages.append(mean_diff)
    
    # 2. Trend analysis: Does CLMBR advantage change with k?
    print(f"\n📈 Trend Analysis: How does CLMBR advantage change with k?")
    
    if len(k_vals_for_trend) > 3:
        corr, corr_p = stats.pearsonr(k_vals_for_trend, clmbr_advantages)
        print(f"Correlation between k and CLMBR advantage: r={corr:.3f}, p={corr_p:.3f}")
        
        if corr > 0:
            trend_desc = "CLMBR advantage INCREASES with more training data"
        else:
            trend_desc = "CLMBR advantage DECREASES with more training data"
        
        print(f"Interpretation: {trend_desc}")
        
        results['k_trend_analysis'] = {
            'correlation': corr,
            'correlation_p': corr_p,
            'interpretation': trend_desc,
            'k_values': k_vals_for_trend,
            'advantages': clmbr_advantages
        }
    
    # 3. Summary statistics
    all_significant = sum(1 for k_res in results['by_k'].values() if k_res['significant'])
    total_k_vals = len(results['by_k'])
    
    print(f"\n📊 Summary:")
    print(f"Significant advantages for CLMBR: {all_significant}/{total_k_vals} k-values")
    
    if results['by_k']:
        avg_advantage = np.mean([res['mean_diff'] for res in results['by_k'].values()])
        avg_win_rate = np.mean([res['clmbr_wins']/res['total_tasks'] for res in results['by_k'].values()])
        print(f"Average CLMBR advantage across all k: {avg_advantage:+.3f}")
        print(f"Average CLMBR win rate: {avg_win_rate:.1%}")
        
        results['summary'] = {
            'significant_k_count': all_significant,
            'total_k_count': total_k_vals,
            'avg_advantage': avg_advantage,
            'avg_win_rate': avg_win_rate
        }
    
    return results

def analyze_rq2_1_fair_comparison(df_results: pd.DataFrame, score: str = 'auroc', alpha: float = 0.05) -> Dict:
    """
    RQ2.1: Fair comparison between linear and BEST non-linear heads
    """
    print(f"\n🧪 RQ2.1 FAIR COMPARISON ANALYSIS: Linear vs Best Non-Linear ({score.upper()})")
    print("=" * 80)
    
    linear_heads = ['lr_lbfgs']
    nonlinear_heads = ['knn', 'rf', 'gbm']
    
    # Use k=-1 for this analysis (could be extended to k-shot)
    full_data = df_results[
        (df_results['score'] == score) & 
        (df_results['k'] == -1)
    ].copy()
    
    results = {
        'by_model': {},
        'summary': {}
    }
    
    model_types = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    
    for model_type in model_types:
        print(f"\n📊 {model_type.upper()} Analysis:")
        
        model_data = full_data[full_data['model'] == model_type]
        
        if model_data.empty:
            print("  No data found!")
            continue
        
        tasks = model_data['labeling_function'].unique()
        
        linear_task_scores = []
        best_nonlinear_task_scores = []
        task_improvements = []
        best_heads_used = []
        
        print(f"{'Task':<25} {'Linear':<8} {'Best NL':<8} {'Improvement':<8} {'Best Head'}")
        print("-" * 75)
        
        for task in tasks:
            task_data = model_data[model_data['labeling_function'] == task]
            
            # Linear performance
            linear_scores = task_data[task_data['head'].isin(linear_heads)]['value'].values
            
            # Find best non-linear performance
            best_nonlinear_score = -1
            best_nonlinear_head = None
            
            for head in nonlinear_heads:
                head_scores = task_data[task_data['head'] == head]['value'].values
                if len(head_scores) > 0:
                    head_mean = np.mean(head_scores)
                    if head_mean > best_nonlinear_score:
                        best_nonlinear_score = head_mean
                        best_nonlinear_head = head
            
            if len(linear_scores) > 0 and best_nonlinear_score > -1:
                linear_mean = np.mean(linear_scores)
                improvement = best_nonlinear_score - linear_mean
                
                linear_task_scores.append(linear_mean)
                best_nonlinear_task_scores.append(best_nonlinear_score)
                task_improvements.append(improvement)
                best_heads_used.append(best_nonlinear_head)
                
                task_name = LABELING_FUNCTION_2_PAPER_NAME.get(task, task)[:23]
                improvement_marker = "+" if improvement > 0 else ""
                print(f"{task_name:<25} {linear_mean:<8.3f} {best_nonlinear_score:<8.3f} "
                      f"{improvement_marker}{improvement:<7.3f} {best_nonlinear_head}")
        
        # Overall analysis
        if len(task_improvements) > 1:
            task_improvements = np.array(task_improvements)
            linear_scores_arr = np.array(linear_task_scores)
            nonlinear_scores_arr = np.array(best_nonlinear_task_scores)
            
            # Paired tests: best non-linear vs linear
            wilcoxon_stat, wilcoxon_p = wilcoxon(nonlinear_scores_arr, linear_scores_arr, alternative='greater')
            
            mean_improvement = np.mean(task_improvements)
            positive_improvements = np.sum(task_improvements > 0)
            
            # Effect size
            cohens_d = mean_improvement / np.std(task_improvements, ddof=1)
            
            # Confidence interval
            ci_lower = np.percentile(task_improvements, 2.5)
            ci_upper = np.percentile(task_improvements, 97.5)
            
            print(f"\nOverall Results for {model_type}:")
            print(f"  Mean improvement: {mean_improvement:+.3f}")
            print(f"  95% CI: [{ci_lower:+.3f}, {ci_upper:+.3f}]")
            print(f"  Tasks with improvement: {positive_improvements}/{len(task_improvements)} ({positive_improvements/len(task_improvements):.1%})")
            print(f"  Wilcoxon p-value: {wilcoxon_p:.4f}")
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
            print(f"  Significant: {'Yes' if wilcoxon_p < alpha else 'No'}")
            
            # Most commonly used best head
            from collections import Counter
            head_counts = Counter(best_heads_used)
            most_common_head = head_counts.most_common(1)[0]
            print(f"  Most effective non-linear head: {most_common_head[0]} (used in {most_common_head[1]}/{len(best_heads_used)} tasks)")
            
            results['by_model'][model_type] = {
                'mean_improvement': mean_improvement,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'positive_count': int(positive_improvements),
                'total_tasks': len(task_improvements),
                'wilcoxon_p': wilcoxon_p,
                'cohens_d': cohens_d,
                'significant': wilcoxon_p < alpha,
                'best_head_counts': dict(head_counts),
                'most_common_head': most_common_head[0]
            }
    
    return results

def save_improved_results_summary(rq1_1_results: Dict, rq1_2_results: Dict, rq2_1_results: Dict, 
                                 output_dir: str, score: str):
    """Save improved analysis summary"""
    
    summary_path = os.path.join(output_dir, f"improved_statistical_summary_{score}.txt")
    
    with open(summary_path, 'w') as f:
        f.write(f"EHRSHOT Improved K-Shot Aware Statistical Analysis ({score.upper()})\n")
        f.write("=" * 80 + "\n\n")
        
        # RQ1.1 Results
        f.write("1. RQ1.1: CLINICALBERT TYPE COMPARISON (K-SHOT AWARE)\n")
        f.write("-" * 55 + "\n")
        
        if rq1_1_results and 'overall_across_k' in rq1_1_results:
            if 'friedman' in rq1_1_results['overall_across_k']:
                friedman = rq1_1_results['overall_across_k']['friedman']
                f.write(f"Friedman test (across all k-values): χ²={friedman['statistic']:.3f}, p={friedman['p_value']:.6f}\n")
                f.write(f"Significant difference between types: {'Yes' if friedman['significant'] else 'No'}\n")
                
                if 'type_means' in friedman:
                    tm = friedman['type_means']
                    f.write(f"Overall means: Type1={tm['type1']:.3f}, Type2={tm['type2']:.3f}, Type3={tm['type3']:.3f}\n")
        
        if 'k_trend_analysis' in rq1_1_results:
            trend = rq1_1_results['k_trend_analysis']
            f.write(f"\nTrend Analysis:\n")
            f.write(f"  Correlation (k vs Type3 advantage): r={trend['correlation']:.3f}, p={trend['correlation_p']:.3f}\n")
            f.write(f"  Interpretation: {trend['interpretation']}\n")
        
        # RQ1.2 Results  
        f.write("\n\n2. RQ1.2: CLMBR vs CLINICALBERT (K-SHOT AWARE)\n")
        f.write("-" * 50 + "\n")
        
        if rq1_2_results and 'summary' in rq1_2_results:
            summary = rq1_2_results['summary']
            f.write(f"Significant CLMBR advantages: {summary['significant_k_count']}/{summary['total_k_count']} k-values\n")
            f.write(f"Average CLMBR advantage: {summary['avg_advantage']:+.3f}\n")
            f.write(f"Average CLMBR win rate: {summary['avg_win_rate']:.1%}\n")
        
        if 'k_trend_analysis' in rq1_2_results:
            trend = rq1_2_results['k_trend_analysis']
            f.write(f"\nTrend Analysis:\n")
            f.write(f"  Correlation (k vs CLMBR advantage): r={trend['correlation']:.3f}, p={trend['correlation_p']:.3f}\n")
            f.write(f"  Interpretation: {trend['interpretation']}\n")
        
        # RQ2.1 Results
        f.write("\n\n3. RQ2.1: LINEAR vs BEST NON-LINEAR HEADS (FAIR COMPARISON)\n")
        f.write("-" * 65 + "\n")
        
        if rq2_1_results and 'by_model' in rq2_1_results:
            for model_type, data in rq2_1_results['by_model'].items():
                f.write(f"\n{model_type.upper()}:\n")
                f.write(f"  Mean improvement: {data['mean_improvement']:+.3f}\n")
                f.write(f"  95% CI: [{data['ci_lower']:+.3f}, {data['ci_upper']:+.3f}]\n")
                f.write(f"  Tasks improved: {data['positive_count']}/{data['total_tasks']} ({data['positive_count']/data['total_tasks']:.1%})\n")
                f.write(f"  Wilcoxon p-value: {data['wilcoxon_p']:.4f}\n")
                f.write(f"  Effect size (Cohen's d): {data['cohens_d']:.3f}\n")
                f.write(f"  Most effective head: {data['most_common_head']}\n")
                f.write(f"  Significant: {'Yes' if data['significant'] else 'No'}\n")
    
    print(f"📄 Saved improved summary to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Improved K-Shot Aware Statistical Analysis for EHRSHOT")
    parser.add_argument("--path_to_results_dir", required=True, type=str,
                       help="Path to directory containing results")
    parser.add_argument("--path_to_output_dir", required=True, type=str,
                       help="Path to save analysis outputs")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level (default: 0.05)")
    parser.add_argument("--scores", nargs='+', default=['auroc', 'auprc'],
                       help="Scores to analyze (default: auroc auprc)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Load data
    print("📂 Loading results data...")
    df_results = load_results_data(args.path_to_results_dir)
    print(f"✅ Loaded {len(df_results)} rows from {len(df_results['labeling_function'].unique())} tasks")
    
    # Run improved analyses for each score
    for score in args.scores:
        if score not in df_results['score'].unique():
            print(f"Warning: {score} not found in results data")
            continue
        
        print(f"\n{'='*80}")
        print(f"IMPROVED K-SHOT AWARE STATISTICAL ANALYSIS: {score.upper()}")
        print(f"{'='*80}")
        
        # 1. RQ1.1: ClinicalBERT types analysis (k-shot aware)
        rq1_1_results = analyze_rq1_1_kshot_aware(df_results, score, args.alpha)
        
        # 2. RQ1.2: CLMBR vs ClinicalBERT analysis (k-shot aware)
        rq1_2_results = analyze_rq1_2_kshot_aware(df_results, score, args.alpha)
        
        # 3. RQ2.1: Linear vs Non-linear heads analysis (fair comparison)
        rq2_1_results = analyze_rq2_1_fair_comparison(df_results, score, args.alpha)
        
        # 4. Save improved summary
        save_improved_results_summary(rq1_1_results, rq1_2_results, rq2_1_results, 
                                     args.path_to_output_dir, score)
    
    print(f"\n✅ Improved k-shot aware statistical analysis complete!")
    print(f"📊 Results saved to: {args.path_to_output_dir}")

if __name__ == "__main__":
    main() 