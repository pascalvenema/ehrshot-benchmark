#!/usr/bin/env python3
"""
Analyze optimal dimensions from dimensionality analysis and create k=-1 comparison plot.

This script:
1. Analyzes optimal dimensions for CLMBR and ClinicalBERT from dimensionality analysis
2. Creates a comparison plot of k=-1 scores across three task categories for all prediction heads
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Task group definitions
TASK_GROUPS = {
    "Operational Outcomes": ['guo_los', 'guo_readmission', 'guo_icu'],
    "Lab Values": ['lab_thrombocytopenia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia'],
    "New Diagnoses": ['new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 'new_lupus', 'new_acutemi']
}

def analyze_optimal_dimensions(dimensionality_file: str) -> Dict:
    """Analyze optimal dimensions for CLMBR and ClinicalBERT from dimensionality analysis."""
    print("\n" + "="*80)
    print("🔬 ANALYZING OPTIMAL DIMENSIONS FOR CLMBR AND CLINICALBERT")
    print("="*80)
    
    # Try to load detailed results first for proper analysis
    dimensionality_dir = os.path.dirname(dimensionality_file)
    detailed_file = os.path.join(dimensionality_dir, "dimensionality_results.csv")
    
    if os.path.exists(detailed_file):
        print("📊 Using detailed results for accurate dimension analysis...")
        df_detailed = pd.read_csv(detailed_file)
        return analyze_optimal_dimensions_from_detailed(df_detailed)
    else:
        print("⚠️ Detailed results not found, falling back to summary analysis...")
        df_dim = pd.read_csv(dimensionality_file)
        return analyze_optimal_dimensions_from_summary(df_dim)

def analyze_optimal_dimensions_from_detailed(df_detailed: pd.DataFrame) -> Dict:
    """Analyze optimal dimensions using detailed results - average performance across ALL tasks."""
    optimal_dims = {}
    
    for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
        model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
        print(f"\n📊 {model_name} OPTIMAL DIMENSIONS (average across all tasks):")
        print("-" * 50)
        
        model_data = df_detailed[df_detailed['model'] == model]
        optimal_dims[model] = {}
        
        for metric in ['auroc', 'auprc']:
            print(f"\n  {metric.upper()}:")
            optimal_dims[model][metric] = {}
            
            for method in ['pca', 'umap']:
                method_data = model_data[
                    (model_data['method'] == method) & 
                    (model_data['metric'] == metric)
                ]
                
                if not method_data.empty:
                    # Find average performance at each dimension across ALL tasks
                    dim_performance = method_data.groupby('dimension')['score'].agg(['mean', 'count']).reset_index()
                    best_dim_row = dim_performance.loc[dim_performance['mean'].idxmax()]
                    best_dim = int(best_dim_row['dimension'])
                    best_score = best_dim_row['mean']
                    task_count = int(best_dim_row['count'])
                    
                    optimal_dims[model][metric][method] = {
                        'dimension': best_dim,
                        'avg_score': best_score,
                        'task_count': task_count
                    }
                    
                    print(f"    {method.upper()}: {best_dim}D (avg: {best_score:.3f}, {task_count} tasks)")
                    
                    # Show performance at other key dimensions for comparison
                    print(f"      Dimension performance:")
                    dim_perf_sorted = dim_performance.sort_values('mean', ascending=False)
                    for _, row in dim_perf_sorted.head(4).iterrows():
                        dim = int(row['dimension'])
                        mean_score = row['mean']
                        count = int(row['count'])
                        marker = " ⭐" if dim == best_dim else ""
                        print(f"        {dim}D: {mean_score:.3f} ({count} tasks){marker}")
    
    # Summary analysis
    print(f"\n📈 SUMMARY - BEST PERFORMING DIMENSIONS:")
    print("-" * 50)
    
    for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
        model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
        
        if model in optimal_dims:
            # Find best overall method/dimension for AUROC
            auroc_results = optimal_dims[model].get('auroc', {})
            if auroc_results:
                best_method = max(auroc_results.keys(), key=lambda x: auroc_results[x]['avg_score'])
                best_dim = auroc_results[best_method]['dimension']
                best_score = auroc_results[best_method]['avg_score']
                
                print(f"{model_name}: {best_method.upper()} at {best_dim}D (AUROC: {best_score:.3f})")
    
    return optimal_dims

def analyze_optimal_dimensions_from_summary(df_dim: pd.DataFrame) -> Dict:
    """Fallback analysis using summary data (less accurate)."""
    optimal_dims = {}
    
    for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
        model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
        print(f"\n📊 {model_name} OPTIMAL DIMENSIONS:")
        print("-" * 50)
        
        model_data = df_dim[df_dim['model'] == model]
        optimal_dims[model] = {}
        
        for metric in ['auroc', 'auprc']:
            print(f"\n  {metric.upper()}:")
            optimal_dims[model][metric] = {}
            
            for method in ['pca', 'umap']:
                method_data = model_data[
                    (model_data['method'] == method) & 
                    (model_data['metric'] == metric)
                ]
                
                if not method_data.empty:
                    # Use overall average performance (less accurate but best we can do)
                    overall_avg = method_data['best_score'].mean()
                    # Use most common optimal dimension as representative
                    most_common_dim = method_data['optimal_dimension'].mode().iloc[0] if len(method_data) > 0 else 25
                    
                    optimal_dims[model][metric][method] = {
                        'dimension': most_common_dim,
                        'avg_score': overall_avg,
                        'task_count': len(method_data)
                    }
                    
                    print(f"    {method.upper()}: ~{most_common_dim}D (overall avg: {overall_avg:.3f}, {len(method_data)} tasks)")
    
    return optimal_dims

def load_full_data_results(results_dir: str) -> pd.DataFrame:
    """Load k=-1 (full data) results from all tasks."""
    print(f"\n📂 LOADING K=-1 (FULL DATA) RESULTS FROM {results_dir}")
    print("-" * 50)
    
    all_results = []
    
    # Get all task directories
    task_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    for task in task_dirs:
        results_file = os.path.join(results_dir, task, 'all_results.csv')
        if os.path.exists(results_file):
            try:
                df = pd.read_csv(results_file)
                
                # Filter for k=-1 (full data) and first replicate only
                full_data = df[
                    (df['k'] == -1) & 
                    (df['replicate'] == 0) &
                    (df['score'].isin(['auroc', 'auprc']))
                ]
                
                if not full_data.empty:
                    all_results.append(full_data)
                    print(f"  ✓ Loaded {task}: {len(full_data)} records")
                else:
                    print(f"  ⚠️  No k=-1 data for {task}")
                    
            except Exception as e:
                print(f"  ❌ Error loading {task}: {e}")
        else:
            print(f"  ❌ Missing results file for {task}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\n✅ Combined data: {len(combined_df)} total records across {len(task_dirs)} tasks")
        return combined_df
    else:
        print("❌ No results loaded!")
        return pd.DataFrame()

def create_intuitive_comparison_plot(df_results: pd.DataFrame, 
                                   df_reduced_knn: pd.DataFrame,
                                   optimal_dims: Dict,
                                   output_path: str) -> None:
    """Create a clean, focused comparison plot highlighting kNN dimensionality reduction impact."""
    print(f"\n🎨 CREATING CLEAN DIMENSIONALITY REDUCTION COMPARISON PLOT")
    print("-" * 50)
    
    # Filter for the models and metrics we want
    target_models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    
    # Focus on key prediction heads: LR (baseline), kNN (full), and GBM (best tree method)
    target_heads = ['lr_lbfgs', 'knn', 'gbm']
    
    df_filtered = df_results[
        (df_results['model'].isin(target_models)) &
        (df_results['head'].isin(target_heads)) &
        (df_results['score'].isin(['auroc', 'auprc']))
    ].copy()
    
    if df_filtered.empty:
        print("❌ No filtered data available for plotting!")
        return
    
    # Add task category information
    task_to_category = {}
    for category, tasks in TASK_GROUPS.items():
        for task in tasks:
            task_to_category[task] = category
    
    df_filtered['task_category'] = df_filtered['labeling_function'].map(task_to_category)
    df_filtered = df_filtered.dropna(subset=['task_category'])
    
    # Process reduced kNN data if available
    reduced_knn_processed = []
    if not df_reduced_knn.empty:
        print("📊 Processing reduced kNN data...")
        
        for _, row in df_reduced_knn.iterrows():
            task = row['task']
            if task in task_to_category and row['metric'] in ['auroc', 'auprc']:
                reduced_knn_processed.append({
                    'labeling_function': task,
                    'task_category': task_to_category[task],
                    'model': row['model'],
                    'head': 'knn_reduced',
                    'score': row['metric'],
                    'value': row['score'],
                    'k': -1,
                    'replicate': 0
                })
        
        if reduced_knn_processed:
            df_reduced_formatted = pd.DataFrame(reduced_knn_processed)
            df_combined = pd.concat([df_filtered, df_reduced_formatted], ignore_index=True)
            print(f"✅ Added {len(reduced_knn_processed)} reduced kNN records")
        else:
            df_combined = df_filtered
            print("⚠️  No valid reduced kNN records to add")
    else:
        df_combined = df_filtered
        print("⚠️  No reduced kNN data available")
    
    # Calculate category means for cleaner visualization
    category_means = df_combined.groupby([
        'task_category', 'model', 'head', 'score'
    ])['value'].mean().reset_index()
    
    # Create separate plots for AUROC and AUPRC
    for metric in ['auroc', 'auprc']:
        # Create figure with side-by-side subplots for each model - reduced height
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        metric_data = category_means[category_means['score'] == metric]
        
        if metric_data.empty:
            continue
        
        categories = list(TASK_GROUPS.keys())
        
        # Define clean color scheme
        colors = {
            'lr_lbfgs': '#2C3E50',      # Dark blue-gray for baseline
            'knn': '#E74C3C',           # Red for kNN full
            'knn_reduced': '#27AE60',   # Green for kNN reduced (HIGHLIGHT)
            'gbm': '#F39C12'            # Orange for GBM (more distinct from LR)
        }
        
        # Plot for each model
        for ax, model, model_name in [(ax1, 'clmbr', 'CLMBR'), 
                                      (ax2, 'clinicalbert_type3_clinicalbert_pool', 'ClinicalBERT')]:
            
            model_data = metric_data[metric_data['model'] == model]
            
            if model_data.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}')
                continue
            
            # Prepare data for each category - reduced bar width and better spacing
            x_pos = np.arange(len(categories))
            bar_width = 0.15  # Reduced from 0.2 to prevent overlap
            
            # Get scores for each head across categories - kNN methods together on the right
            heads_to_plot = ['lr_lbfgs', 'gbm', 'knn', 'knn_reduced']
            head_labels = ['Logistic Regression', 'GBM', 'kNN (Full)', 'kNN (Reduced)']
            
            for i, (head, label) in enumerate(zip(heads_to_plot, head_labels)):
                scores = []
                for category in categories:
                    head_data = model_data[
                        (model_data['task_category'] == category) &
                        (model_data['head'] == head)
                    ]
                    score = head_data['value'].iloc[0] if len(head_data) > 0 else 0
                    scores.append(score)
                
                # Skip if no data for this head
                if all(s == 0 for s in scores):
                    continue
                
                # Standard styling for all bars
                alpha = 0.8
                edgecolor = 'white'
                linewidth = 1
                hatch = None
                
                # Plot bars - improved positioning to prevent overlap
                positions = x_pos + (i - 1.5) * bar_width  # Center the bars better
                bars = ax.bar(positions, scores, bar_width,
                            label=label,
                            color=colors[head],
                            alpha=alpha,
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                            hatch=hatch)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    if score > 0:
                        height = bar.get_height()
                        # Use black text for all bars for visibility
                        text_color = 'black'
                        fontweight = 'normal'
                        fontsize = 10
                        
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.2f}', ha='center', va='bottom',
                               fontsize=fontsize, fontweight=fontweight, color=text_color)
            
            # Customize subplot
            ax.set_xlabel('', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{metric.upper()}', fontsize=14, fontweight='bold')
            
            # Get optimal dimensions info for this model
            optimal_info = ""
            if model in optimal_dims and metric in optimal_dims[model]:
                metric_results = optimal_dims[model][metric]
                if metric_results:
                    # Use overall performance to select best method
                    best_method = max(metric_results.keys(), key=lambda x: metric_results[x]['avg_score'])
                    best_dim = metric_results[best_method]['dimension']
                    optimal_info = f"\n(kNN Reduced: {best_method.upper()}-{best_dim}D)"
            
            ax.set_title(f'{model_name}{optimal_info}', fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits
            if metric == 'auroc':
                ax.set_ylim(0.5, 0.90)
            else:
                ax.set_ylim(0, 0.90)
            
            # Improve aesthetics
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add individual legend to each subplot (more space efficient)
            ax.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # More space since no bottom legend needed
        
        # Save the plot
        metric_output = output_path.replace('.png', f'_{metric}.png')
        plt.savefig(metric_output, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ {metric.upper()} plot saved to: {metric_output}")

def load_reduced_knn_results(dimensionality_dir: str, optimal_dims: Dict) -> pd.DataFrame:
    """Load reduced kNN results from dimensionality analysis if available."""
    print(f"\n📂 LOADING REDUCED kNN RESULTS FROM {dimensionality_dir}")
    print("-" * 50)
    
    # Try to find reduced kNN results files
    reduced_results_file = os.path.join(dimensionality_dir, "dimensionality_results.csv")
    
    if not os.path.exists(reduced_results_file):
        print(f"❌ Reduced kNN results file not found: {reduced_results_file}")
        return pd.DataFrame()
    
    try:
        df_reduced = pd.read_csv(reduced_results_file)
        
        # Extract optimal dimensions from our analysis
        clmbr_optimal_dim = 25  # Default
        clinicalbert_optimal_dim = 200  # Default
        clmbr_optimal_method = 'pca'  # fallback
        clinicalbert_optimal_method = 'pca'  # fallback
        
        if 'clmbr' in optimal_dims and 'auroc' in optimal_dims['clmbr']:
            clmbr_results = optimal_dims['clmbr']['auroc']
            # Use the method with best overall performance
            if clmbr_results:
                best_method = max(clmbr_results.keys(), key=lambda x: clmbr_results[x]['avg_score'])
                if best_method in clmbr_results:
                    clmbr_optimal_dim = clmbr_results[best_method]['dimension']
                    clmbr_optimal_method = best_method
            else:
                clmbr_optimal_method = 'pca'  # fallback
        else:
            clmbr_optimal_method = 'pca'  # fallback
        
        if 'clinicalbert_type3_clinicalbert_pool' in optimal_dims and 'auroc' in optimal_dims['clinicalbert_type3_clinicalbert_pool']:
            cb_results = optimal_dims['clinicalbert_type3_clinicalbert_pool']['auroc']
            # Use the method with best overall performance
            if cb_results:
                best_method = max(cb_results.keys(), key=lambda x: cb_results[x]['avg_score'])
                if best_method in cb_results:
                    clinicalbert_optimal_dim = cb_results[best_method]['dimension']
                    clinicalbert_optimal_method = best_method
            else:
                clinicalbert_optimal_method = 'pca'  # fallback
        else:
            clinicalbert_optimal_method = 'pca'  # fallback
        
        # Filter for optimal dimensions using the actually best methods
        optimal_clmbr = df_reduced[
            (df_reduced['model'] == 'clmbr') &
            (df_reduced['method'] == clmbr_optimal_method) &
            (df_reduced['dimension'] == clmbr_optimal_dim)
        ]
        
        optimal_clinicalbert = df_reduced[
            (df_reduced['model'] == 'clinicalbert_type3_clinicalbert_pool') &
            (df_reduced['method'] == clinicalbert_optimal_method) &
            (df_reduced['dimension'] == clinicalbert_optimal_dim)
        ]
        
        reduced_knn = pd.concat([optimal_clmbr, optimal_clinicalbert], ignore_index=True)
        
        if not reduced_knn.empty:
            print(f"✅ Loaded {len(reduced_knn)} reduced kNN records")
            print(f"   CLMBR {clmbr_optimal_method.upper()}-{clmbr_optimal_dim}D: {len(optimal_clmbr)} records")
            print(f"   ClinicalBERT {clinicalbert_optimal_method.upper()}-{clinicalbert_optimal_dim}D: {len(optimal_clinicalbert)} records")
            return reduced_knn
        else:
            print("❌ No reduced kNN data found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error loading reduced kNN results: {e}")
        return pd.DataFrame()

def main():
    """Main analysis pipeline."""
    print("🚀 EHRSHOT DIMENSIONALITY ANALYSIS & K=-1 COMPARISON")
    print("="*80)
    
    # Paths
    dimensionality_file = "EHRSHOT_ASSETS/dimensionality_analysis/dimensionality_summary.csv"
    dimensionality_dir = "EHRSHOT_ASSETS/dimensionality_analysis"
    results_dir = "EHRSHOT_ASSETS/results"
    output_plot = "k_minus_1_intuitive_category_comparison.png"
    
    # Check if files exist
    if not os.path.exists(dimensionality_file):
        print(f"❌ Dimensionality file not found: {dimensionality_file}")
        return
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Step 1: Analyze optimal dimensions
    optimal_dims = analyze_optimal_dimensions(dimensionality_file)
    
    # Step 2: Load k=-1 results
    df_results = load_full_data_results(results_dir)
    
    if df_results.empty:
        print("❌ No results data loaded. Cannot create comparison plot.")
        return
    
    # Step 3: Load reduced kNN results (if available)
    df_reduced_knn = load_reduced_knn_results(dimensionality_dir, optimal_dims)
    
    # Step 4: Create intuitive comparison plots (separate for AUROC and AUPRC)
    create_intuitive_comparison_plot(df_results, df_reduced_knn, optimal_dims, output_plot)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Plots saved with separate AUROC and AUPRC figures")

if __name__ == "__main__":
    main() 