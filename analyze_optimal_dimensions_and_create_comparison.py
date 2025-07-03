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
    
    # Load dimensionality analysis results
    df_dim = pd.read_csv(dimensionality_file)
    
    # Group by model, method, and metric to find optimal dimensions
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
                    # Find the dimension that gives the highest average score across tasks
                    avg_scores = method_data.groupby('optimal_dimension')['best_score'].mean()
                    best_dim = avg_scores.idxmax()
                    best_score = avg_scores.max()
                    
                    optimal_dims[model][metric][method] = {
                        'dimension': best_dim,
                        'avg_score': best_score,
                        'task_count': len(method_data)
                    }
                    
                    print(f"    {method.upper()}: {best_dim}D (avg score: {best_score:.3f}, {len(method_data)} tasks)")
    
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
    """Create an intuitive comparison plot using category means."""
    print(f"\n🎨 CREATING INTUITIVE CATEGORY COMPARISON PLOT")
    print("-" * 50)
    
    # Filter for the models and metrics we want
    target_models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    target_heads = ['lr_lbfgs', 'knn', 'gbm', 'rf']  # All prediction heads
    
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
    
    # Mark all current kNN as 'full' (non-reduced)
    df_filtered['head_type'] = df_filtered['head'].apply(lambda x: 'knn_full' if x == 'knn' else x)
    
    # Process reduced kNN data if available
    if not df_reduced_knn.empty:
        print("📊 Processing reduced kNN data...")
        
        # Convert reduced kNN data to match our format
        reduced_knn_processed = []
        
        for _, row in df_reduced_knn.iterrows():
            # Map the task names and add category information
            task = row['task']
            if task in task_to_category and row['metric'] in ['auroc', 'auprc']:
                reduced_knn_processed.append({
                    'labeling_function': task,
                    'task_category': task_to_category[task],
                    'model': row['model'],
                    'head_type': 'knn_reduced',
                    'score': row['metric'],
                    'value': row['score'],
                    'k': -1,
                    'replicate': 0
                })
        
        if reduced_knn_processed:
            df_reduced_formatted = pd.DataFrame(reduced_knn_processed)
            # Combine with the main results
            df_combined = pd.concat([df_filtered, df_reduced_formatted], ignore_index=True)
            print(f"✅ Added {len(reduced_knn_processed)} reduced kNN records")
        else:
            df_combined = df_filtered
            print("⚠️  No valid reduced kNN records to add")
    else:
        df_combined = df_filtered
        print("⚠️  No reduced kNN data available")
    
    # Calculate category means
    category_means = df_combined.groupby([
        'task_category', 'model', 'head_type', 'score'
    ])['value'].mean().reset_index()
    
    # Colors and styling
    colors = {
        'clmbr': {
            'lr_lbfgs': '#1f77b4', 
            'knn_full': '#ff7f0e', 
            'knn_reduced': '#ffbb78',  # lighter orange for reduced kNN
            'gbm': '#2ca02c', 
            'rf': '#d62728'
        },
        'clinicalbert_type3_clinicalbert_pool': {
            'lr_lbfgs': '#9467bd', 
            'knn_full': '#8c564b', 
            'knn_reduced': '#c5b0d5',  # lighter purple for reduced kNN
            'gbm': '#e377c2', 
            'rf': '#7f7f7f'
        }
    }
    
    # Create separate plots for AUROC and AUPRC
    for metric in ['auroc', 'auprc']:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        metric_data = category_means[category_means['score'] == metric]
        
        if metric_data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.upper()}')
            continue
        
        # Prepare data for grouped bar chart
        categories = list(TASK_GROUPS.keys())
        n_categories = len(categories)
        target_heads_expanded = ['lr_lbfgs', 'knn_full', 'knn_reduced', 'gbm', 'rf']
        n_models = len(target_models)
        
        # Width of bars and positions
        bar_width = 0.08
        r1 = np.arange(n_categories)
        
        # Plot bars for each model-head combination
        bar_pos = 0
        for model in target_models:
            model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
            
            for head in target_heads_expanded:
                # Check if this combination exists in our data
                has_data = any(
                    (metric_data['model'] == model) & 
                    (metric_data['head_type'] == head)
                )
                
                if not has_data:
                    continue
                    
                head_display = head.replace('lr_lbfgs', 'LR').replace('_', ' ').upper()
                head_display = head_display.replace('KNN FULL', 'kNN (Full)')
                head_display = head_display.replace('KNN REDUCED', 'kNN (Reduced)')
                
                # Get scores for this model-head combination across categories
                scores = []
                for category in categories:
                    score_data = metric_data[
                        (metric_data['task_category'] == category) &
                        (metric_data['model'] == model) &
                        (metric_data['head_type'] == head)
                    ]
                    scores.append(score_data['value'].iloc[0] if len(score_data) > 0 else 0)
                
                # Create label
                label = f'{model_name} + {head_display}'
                
                # Plot bars
                positions = [x + bar_pos * bar_width for x in r1]
                bars = ax.bar(positions, scores, bar_width, 
                            label=label, 
                            color=colors[model][head],
                            alpha=0.8,
                            edgecolor='white',
                            linewidth=0.5)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    if score > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
                
                bar_pos += 1
        
        # Customize subplot
        ax.set_xlabel('Task Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
        
        # Add optimal dimensions info to title
        optimal_info = "OPTIMAL DIMS: "
        for model in ['clmbr', 'clinicalbert_type3_clinicalbert_pool']:
            model_name = 'CLMBR' if model == 'clmbr' else 'ClinicalBERT'
            if model in optimal_dims and metric in optimal_dims[model]:
                metric_results = optimal_dims[model][metric]
                if metric_results:
                    best_method = max(metric_results.keys(), key=lambda x: metric_results[x]['avg_score'])
                    best_dim = metric_results[best_method]['dimension']
                    optimal_info += f"{model_name}: {best_method.upper()}-{best_dim}D  "
        
        ax.set_title(f'{metric.upper()} Performance by Category (K=-1 Full Data)\n{optimal_info}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks([r + bar_width * (bar_pos - 1) / 2 for r in r1])
        ax.set_xticklabels(categories, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        if metric == 'auroc':
            ax.set_ylim(0.5, 1.0)
        else:
            ax.set_ylim(0, 1.0)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # Save individual metric plots
        metric_output = output_path.replace('.png', f'_{metric}.png')
        plt.savefig(metric_output, dpi=300, bbox_inches='tight')
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
        
        if 'clmbr' in optimal_dims and 'auroc' in optimal_dims['clmbr']:
            clmbr_results = optimal_dims['clmbr']['auroc']
            if 'pca' in clmbr_results:
                clmbr_optimal_dim = clmbr_results['pca']['dimension']
        
        if 'clinicalbert_type3_clinicalbert_pool' in optimal_dims and 'auroc' in optimal_dims['clinicalbert_type3_clinicalbert_pool']:
            cb_results = optimal_dims['clinicalbert_type3_clinicalbert_pool']['auroc']
            if 'umap' in cb_results:
                clinicalbert_optimal_dim = cb_results['umap']['dimension']
        
        # Filter for optimal dimensions
        optimal_clmbr = df_reduced[
            (df_reduced['model'] == 'clmbr') &
            (df_reduced['method'] == 'pca') &
            (df_reduced['dimension'] == clmbr_optimal_dim)
        ]
        
        optimal_clinicalbert = df_reduced[
            (df_reduced['model'] == 'clinicalbert_type3_clinicalbert_pool') &
            (df_reduced['method'] == 'umap') &
            (df_reduced['dimension'] == clinicalbert_optimal_dim)
        ]
        
        reduced_knn = pd.concat([optimal_clmbr, optimal_clinicalbert], ignore_index=True)
        
        if not reduced_knn.empty:
            print(f"✅ Loaded {len(reduced_knn)} reduced kNN records")
            print(f"   CLMBR PCA-{clmbr_optimal_dim}D: {len(optimal_clmbr)} records")
            print(f"   ClinicalBERT UMAP-{clinicalbert_optimal_dim}D: {len(optimal_clinicalbert)} records")
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