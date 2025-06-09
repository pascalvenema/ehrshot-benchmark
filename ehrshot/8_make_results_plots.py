import os
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    TASK_GROUP_2_PAPER_NAME,
    TASK_GROUP_2_LABELING_FUNCTION,
    HEAD_2_INFO,
    MODEL_2_INFO, 
    SHOT_STRATS,
    SCORE_MODEL_HEAD_2_COLOR,
    filter_df,
    type_tuple_list,
)
from plot import (
    plot_one_labeling_function,
    plot_one_task_group,
    plot_one_task_group_box_plot,
    _plot_unified_legend,
)

def plot_all_labeling_functions(df_results: pd.DataFrame, 
                                score: str, 
                                path_to_output_dir: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None,
                                is_x_scale_log: bool = True,
                                is_std_bars: bool = True):
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    labeling_functions: List[str] = df_results[df_results['score'] == score]['labeling_function'].unique().tolist()
    for idx, labeling_function in enumerate(labeling_functions):
        sub_tasks: List[str] = df_results[(df_results['score'] == score) & (df_results['labeling_function'] == labeling_function)]['sub_task'].unique().tolist()
        plot_one_labeling_function(df_results, 
                                    axes.flat[idx], 
                                    labeling_function, 
                                    sub_tasks, 
                                    score,
                                    model_heads=model_heads,
                                    is_x_scale_log=is_x_scale_log,
                                    is_std_bars=False if labeling_function == 'chexpert' else is_std_bars)

    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, ncol=2, fontsize=8)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.savefig(os.path.join(path_to_output_dir, f"tasks_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_groups(df_results: pd.DataFrame, 
                        score: str, 
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None,
                        is_x_scale_log: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in enumerate(task_groups):
        plot_one_task_group(df_results, 
                            axes.flat[idx], 
                            task_group, 
                            score,
                            model_heads=model_heads,
                            is_x_scale_log=is_x_scale_log)
    
    # Create a unified legend for the entire figure
    _plot_unified_legend(fig, axes, ncol=2, fontsize=8)

    # Plot aesthetics
    fig.suptitle(f'{score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_group_box_plots(df_results: pd.DataFrame,
                            score: str, 
                            path_to_output_dir: str,
                            model_heads: Optional[List[Tuple[str, str]]] = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())

    for idx, task_group in tqdm(enumerate(task_groups)):
        plot_one_task_group_box_plot(df_results, 
                                    axes.flat[idx], 
                                    task_group, 
                                    score,
                                    model_heads=model_heads)
    
    # Create a unified legend for the entire figure
    df_ = filter_df(df_results, score=score, model_heads=model_heads)
    legend_n_col: int = 2
    handles = [ 
        Patch(
            facecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head], 
            edgecolor=SCORE_MODEL_HEAD_2_COLOR[score][model][head], 
            label=f"{MODEL_2_INFO[model]['label']}+{HEAD_2_INFO[head]['label']}"
        ) 
        for (model, head) in df_[['model', 'head']].drop_duplicates().itertuples(index=False)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=legend_n_col, fontsize=12)
    
    # Plot aesthetics
    fig.suptitle(f'Few-shot v. Full data {score.upper()} by Task Group', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.25)
    plt.savefig(os.path.join(path_to_output_dir, f"taskgroups_boxplot_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_clinicalbert_comparison_by_type(df_results: pd.DataFrame, 
                                        score: str, 
                                        path_to_output_dir: str,
                                        is_x_scale_log: bool = True):
    """Plot comparison of ClinicalBERT Type 1, 2, and 3 models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    # Filter for only ClinicalBERT models
    clinicalbert_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        
        # Get data for this task group
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        df_group = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(clinicalbert_models))
        ]
        
        if df_group.empty:
            ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]} - No ClinicalBERT Data', fontsize=12)
            continue
            
        # Group by ClinicalBERT type
        type1_data = df_group[df_group['model'].str.contains('type1')]
        type2_data = df_group[df_group['model'].str.contains('type2')]  
        type3_data = df_group[df_group['model'].str.contains('type3')]
        
        # Plot mean performance for each type
        for cb_type, cb_data, color, label in [
            ('type1', type1_data, 'red', 'ClinicalBERT Type 1'),
            ('type2', type2_data, 'blue', 'ClinicalBERT Type 2'),
            ('type3', type3_data, 'green', 'ClinicalBERT Type 3')
        ]:
            if cb_data.empty:
                continue
                
            # Group by k-value and compute mean across all models/heads of this type
            grouped = cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            
            # Plot line with error bars
            ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                       color=color, label=label, marker='o', linewidth=2, markersize=6)
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=10)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        
        if is_x_scale_log:
            ax.set_xscale('log')
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
            ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle(f'ClinicalBERT Type Comparison - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_type_comparison_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_best_clinicalbert_models(df_results: pd.DataFrame, 
                                 score: str, 
                                 path_to_output_dir: str,
                                 top_n: int = 5):
    """Plot the best performing ClinicalBERT models across all tasks"""
    # Filter for ClinicalBERT models and full data (k=-1)
    clinicalbert_data = df_results[
        (df_results['model'].str.contains('clinicalbert')) &
        (df_results['score'] == score) &
        (df_results['k'] == -1)
    ]
    
    if clinicalbert_data.empty:
        print(f"No ClinicalBERT data found for {score}")
        return None
    
    # Calculate mean performance for each model+head combination
    model_performance = clinicalbert_data.groupby(['model', 'head'])['value'].agg(['mean', 'std']).reset_index()
    model_performance['model_head'] = model_performance['model'] + '+' + model_performance['head']
    
    # Sort by mean performance and take top N
    model_performance = model_performance.sort_values('mean', ascending=False).head(top_n)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(model_performance))
    bars = ax.barh(y_pos, model_performance['mean'], xerr=model_performance['std'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(model_performance))))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_performance['model_head'], fontsize=10)
    ax.set_xlabel(f'{score.upper()} Score', fontsize=12)
    ax.set_title(f'Top {top_n} ClinicalBERT Models (Full Data Performance)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, model_performance['mean'], model_performance['std'])):
        ax.text(bar.get_width() + std_val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{mean_val:.3f}±{std_val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_output_dir, f"best_clinicalbert_models_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_clinicalbert_vs_baseline_comparison(df_results: pd.DataFrame, 
                                           score: str, 
                                           path_to_output_dir: str):
    """Compare best ClinicalBERT models against baseline models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    # Define baseline and ClinicalBERT models
    baseline_models = ['clmbr', 'count']
    clinicalbert_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Get baseline data
        baseline_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(baseline_models))
        ]
        
        # Get ClinicalBERT data
        cb_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(clinicalbert_models))
        ]
        
        if baseline_data.empty and cb_data.empty:
            ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]} - No Data', fontsize=12)
            continue
        
        # Plot baseline models
        for model in baseline_models:
            model_data = baseline_data[baseline_data['model'] == model]
            if not model_data.empty:
                grouped = model_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                           label=f'{MODEL_2_INFO[model]["label"]}', linewidth=2, marker='s', markersize=6)
        
        # Plot best ClinicalBERT model (by full data performance)
        if not cb_data.empty:
            full_data_cb = cb_data[cb_data['k'] == -1]
            if not full_data_cb.empty:
                best_cb_combo = full_data_cb.groupby(['model', 'head'])['value'].mean().idxmax()
                best_cb_data = cb_data[
                    (cb_data['model'] == best_cb_combo[0]) & 
                    (cb_data['head'] == best_cb_combo[1])
                ]
                grouped = best_cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                
                cb_label = f"Best ClinicalBERT ({best_cb_combo[0].replace('clinicalbert_', '').replace('_', ' ').title()})"
                ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                           label=cb_label, linewidth=2, marker='o', markersize=6, color='purple')
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=10)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle(f'ClinicalBERT vs Baseline Models - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_vs_baseline_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_clinicalbert_pooling_comparison(df_results: pd.DataFrame, 
                                       score: str, 
                                       path_to_output_dir: str,
                                       cb_type: str = 'type3'):
    """Compare different pooling strategies for a specific ClinicalBERT type"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    pooling_strategies = ['max_pool', 'mean_pool', 'clinicalbert_pool']
    colors = ['red', 'blue', 'green']
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        for pool_strategy, color in zip(pooling_strategies, colors):
            model_name = f'clinicalbert_{cb_type}_{pool_strategy}'
            model_data = df_results[
                (df_results['score'] == score) & 
                (df_results['labeling_function'].isin(labeling_functions)) &
                (df_results['model'] == model_name)
            ]
            
            if not model_data.empty:
                grouped = model_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['k'], grouped['mean'], yerr=grouped['std'], 
                           color=color, label=pool_strategy.replace('_', ' ').title(), 
                           linewidth=2, marker='o', markersize=6)
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=10)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle(f'ClinicalBERT {cb_type.upper()} Pooling Strategies - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_{cb_type}_pooling_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_embedding_comparison_shared_heads(df_results: pd.DataFrame, 
                                         score: str, 
                                         path_to_output_dir: str):
    """Compare CLMBR vs ClinicalBERT embeddings using shared prediction heads"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    # Define shared heads that both CLMBR and ClinicalBERT use
    shared_heads = ['lr_lbfgs', 'knn']
    
    # Define colors and styles for each head
    head_styles = {
        'lr_lbfgs': {'color': 'blue', 'linestyle': '-', 'marker': 's'},
        'knn': {'color': 'red', 'linestyle': '--', 'marker': 'o'}
    }
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Plot CLMBR performance for each head separately
        for head in shared_heads:
            clmbr_data = df_results[
                (df_results['score'] == score) & 
                (df_results['labeling_function'].isin(labeling_functions)) &
                (df_results['model'] == 'clmbr') &
                (df_results['head'] == head)
            ]
            
            if not clmbr_data.empty:
                clmbr_grouped = clmbr_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                ax.errorbar(clmbr_grouped['k'], clmbr_grouped['mean'], yerr=clmbr_grouped['std'], 
                           color=head_styles[head]['color'], 
                           linestyle=head_styles[head]['linestyle'],
                           marker=head_styles[head]['marker'], 
                           label=f'CLMBR + {head.replace("_", " ").upper()}',
                           linewidth=2, markersize=6, alpha=0.8)
        
        # Plot best ClinicalBERT performance for each head separately
        cb_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
        
        for head in shared_heads:
            cb_data = df_results[
                (df_results['score'] == score) & 
                (df_results['labeling_function'].isin(labeling_functions)) &
                (df_results['model'].isin(cb_models)) &
                (df_results['head'] == head)
            ]
            
            if not cb_data.empty:
                # Find best ClinicalBERT model for this head at full data
                full_data_cb = cb_data[cb_data['k'] == -1]
                if not full_data_cb.empty:
                    best_cb_model = full_data_cb.groupby('model')['value'].mean().idxmax()
                    best_cb_data = cb_data[cb_data['model'] == best_cb_model]
                    cb_grouped = best_cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                    
                    # Use lighter version of same color and different marker
                    light_color = head_styles[head]['color'] if head_styles[head]['color'] != 'blue' else 'lightblue'
                    light_color = 'lightcoral' if head_styles[head]['color'] == 'red' else light_color
                    
                    ax.errorbar(cb_grouped['k'], cb_grouped['mean'], yerr=cb_grouped['std'], 
                               color=light_color,
                               linestyle=head_styles[head]['linestyle'],
                               marker='^' if head_styles[head]['marker'] == 's' else 'v',
                               label=f'Best ClinicalBERT + {head.replace("_", " ").upper()}',
                               linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=10)
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle(f'CLMBR vs ClinicalBERT Embeddings by Prediction Head - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"embedding_comparison_shared_heads_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_linear_vs_nonlinear_heads(df_results: pd.DataFrame, 
                                  score: str, 
                                  path_to_output_dir: str):
    """Compare linear vs non-linear prediction heads for embedding models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    
    # Define head categories
    linear_heads = ['lr_lbfgs']
    nonlinear_heads = ['knn', 'rf', 'gbm']
    
    # Models to analyze
    embedding_models = ['clmbr'] + [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    for idx, task_group in enumerate(task_groups):
        ax = axes.flat[idx]
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Compare linear vs non-linear for CLMBR
        clmbr_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'] == 'clmbr') &
            (df_results['k'] == -1)  # Full data only
        ]
        
        if not clmbr_data.empty:
            # Linear performance
            linear_clmbr = clmbr_data[clmbr_data['head'].isin(linear_heads)]['value'].mean()
            # Non-linear performance  
            nonlinear_clmbr = clmbr_data[clmbr_data['head'].isin(nonlinear_heads)]['value'].mean()
            
            ax.scatter([1], [linear_clmbr], color='blue', s=100, marker='s', label='CLMBR Linear', alpha=0.7)
            ax.scatter([2], [nonlinear_clmbr], color='blue', s=100, marker='^', label='CLMBR Non-linear', alpha=0.7)
        
        # Compare linear vs non-linear for ClinicalBERT (average across all CB models)
        cb_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
        cb_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(cb_models)) &
            (df_results['k'] == -1)  # Full data only
        ]
        
        if not cb_data.empty:
            # Linear performance
            linear_cb = cb_data[cb_data['head'].isin(linear_heads)]['value'].mean()
            # Non-linear performance
            nonlinear_cb = cb_data[cb_data['head'].isin(nonlinear_heads)]['value'].mean()
            
            ax.scatter([1.1], [linear_cb], color='red', s=100, marker='s', label='ClinicalBERT Linear', alpha=0.7)
            ax.scatter([2.1], [nonlinear_cb], color='red', s=100, marker='^', label='ClinicalBERT Non-linear', alpha=0.7)
        
        # Add count-based models for reference
        count_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'] == 'count') &
            (df_results['k'] == -1)  # Full data only
        ]
        
        if not count_data.empty:
            linear_count = count_data[count_data['head'].isin(linear_heads)]['value'].mean()
            nonlinear_count = count_data[count_data['head'].isin(nonlinear_heads)]['value'].mean()
            
            ax.scatter([0.9], [linear_count], color='green', s=100, marker='s', label='Count Linear', alpha=0.7)
            ax.scatter([1.9], [nonlinear_count], color='green', s=100, marker='^', label='Count Non-linear', alpha=0.7)
        
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Linear\n(LR)', 'Non-Linear\n(KNN, RF, GBM)'])
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if idx == 0:  # Only show legend for first subplot
            ax.legend(fontsize=8, loc='best')
    
    # Use the remaining subplots for summary analysis
    # Subplot 4: Overall improvement analysis
    ax = axes.flat[4]
    
    improvements = {'CLMBR': [], 'ClinicalBERT': [], 'Count': []}
    
    for task_group in task_groups:
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # CLMBR improvement
        clmbr_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'] == 'clmbr') & (df_results['k'] == -1)
        ]
        if not clmbr_data.empty:
            linear_perf = clmbr_data[clmbr_data['head'].isin(linear_heads)]['value'].mean()
            nonlinear_perf = clmbr_data[clmbr_data['head'].isin(nonlinear_heads)]['value'].mean()
            improvements['CLMBR'].append(nonlinear_perf - linear_perf)
        
        # ClinicalBERT improvement
        cb_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(cb_models)) & (df_results['k'] == -1)
        ]
        if not cb_data.empty:
            linear_perf = cb_data[cb_data['head'].isin(linear_heads)]['value'].mean()
            nonlinear_perf = cb_data[cb_data['head'].isin(nonlinear_heads)]['value'].mean()
            improvements['ClinicalBERT'].append(nonlinear_perf - linear_perf)
        
        # Count improvement
        count_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'] == 'count') & (df_results['k'] == -1)
        ]
        if not count_data.empty:
            linear_perf = count_data[count_data['head'].isin(linear_heads)]['value'].mean()
            nonlinear_perf = count_data[count_data['head'].isin(nonlinear_heads)]['value'].mean()
            improvements['Count'].append(nonlinear_perf - linear_perf)
    
    # Create box plot of improvements
    improvement_data = [improvements['CLMBR'], improvements['ClinicalBERT'], improvements['Count']]
    bp = ax.boxplot(improvement_data, labels=['CLMBR', 'ClinicalBERT', 'Count'], patch_artist=True)
    colors = ['blue', 'red', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel(f'Non-linear - Linear {score.upper()}', fontsize=10)
    ax.set_title('Head Type Performance Difference', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Subplot 5: Summary statistics
    ax = axes.flat[5]
    ax.axis('off')
    
    # Calculate summary statistics
    summary_text = f"Summary Statistics ({score.upper()}):\n\n"
    
    for model_type, imps in improvements.items():
        if imps:
            mean_imp = np.mean(imps)
            summary_text += f"{model_type}:\n"
            summary_text += f"  Mean improvement: {mean_imp:+.3f}\n"
            summary_text += f"  Tasks improved: {sum(1 for x in imps if x > 0)}/{len(imps)}\n\n"
    
    ax.text(0.1, 0.8, summary_text, fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    fig.suptitle(f'Linear vs Non-Linear Prediction Heads - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"linear_vs_nonlinear_heads_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_detailed_task_embedding_comparison(df_results: pd.DataFrame, 
                                          score: str, 
                                          path_to_output_dir: str):
    """Create detailed task-by-task comparison of CLMBR vs ClinicalBERT for shared heads"""
    
    # Get tasks with both CLMBR and ClinicalBERT data
    shared_heads = ['lr_lbfgs', 'knn']
    cb_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    valid_tasks = []
    for task in df_results['labeling_function'].unique():
        clmbr_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['model'] == 'clmbr') &
            (df_results['head'].isin(shared_heads)) &
            (df_results['score'] == score)
        ]
        cb_data = df_results[
            (df_results['labeling_function'] == task) &
            (df_results['model'].isin(cb_models)) &
            (df_results['head'].isin(shared_heads)) &
            (df_results['score'] == score)
        ]
        if not clmbr_data.empty and not cb_data.empty:
            valid_tasks.append(task)
    
    if len(valid_tasks) == 0:
        print(f"No valid tasks found for detailed comparison ({score})")
        return None
    
    # Create subplot grid based on number of tasks
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
        
        for head in shared_heads:
            # CLMBR data
            clmbr_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'] == 'clmbr') &
                (df_results['head'] == head) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)  # Full data only for clarity
            ]
            
            # Best ClinicalBERT data for this head
            cb_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'].isin(cb_models)) &
                (df_results['head'] == head) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)
            ]
            
            if not clmbr_data.empty and not cb_data.empty:
                clmbr_perf = clmbr_data['value'].mean()
                best_cb_perf = cb_data.groupby('model')['value'].mean().max()
                
                # Plot bars for this head
                x_pos = 0 if head == 'lr_lbfgs' else 1
                bar_width = 0.35
                
                ax.bar(x_pos - bar_width/2, clmbr_perf, bar_width, 
                      label=f'CLMBR' if head == 'lr_lbfgs' else '', 
                      color='blue', alpha=0.7)
                ax.bar(x_pos + bar_width/2, best_cb_perf, bar_width, 
                      label=f'Best ClinicalBERT' if head == 'lr_lbfgs' else '', 
                      color='red', alpha=0.7)
                
                # Add value labels on bars
                ax.text(x_pos - bar_width/2, clmbr_perf + 0.01, f'{clmbr_perf:.3f}', 
                       ha='center', va='bottom', fontsize=8)
                ax.text(x_pos + bar_width/2, best_cb_perf + 0.01, f'{best_cb_perf:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['LR', 'KNN'])
        ax.set_ylabel(f'{score.upper()}', fontsize=10)
        ax.set_title(LABELING_FUNCTION_2_PAPER_NAME.get(task, task), fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_tasks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'CLMBR vs Best ClinicalBERT by Task and Prediction Head - {score.upper()}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(path_to_output_dir, f"detailed_task_embedding_comparison_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def merge_html_tables(path_to_output_dir: str):
    # Merge together all HTML tables for easy copying
    html_contents = []
    for file in os.listdir(path_to_output_dir):
        if file.endswith('_pretty.html'):
            name = file.replace('_pretty.html', '')
            name = LABELING_FUNCTION_2_PAPER_NAME[name] if name in LABELING_FUNCTION_2_PAPER_NAME else (TASK_GROUP_2_PAPER_NAME[name] if name in TASK_GROUP_2_PAPER_NAME else name)
            table_html = open(os.path.join(path_to_output_dir, file), 'r').read()
            table_html = table_html.replace('\n', '')
            html_contents.append(f'<h5>{name}</h5>\n' + table_html)
    with open(os.path.join(path_to_output_dir, 'merged.html'), 'w') as fd:
        for c in html_contents:
            fd.write(c + '\n\n')

def parse_args():
    parser = argparse.ArgumentParser(description="Make plots of results")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to directory containing saved labels and featurizers")
    parser.add_argument("--path_to_results_dir", required=True, type=str, help="Path to directory containing results from 7_eval.py")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory to save figures")
    parser.add_argument("--shot_strat", required=True, type=str, choices=SHOT_STRATS.keys(), help="What type of k-shot evaluation we are interested in.")
    parser.add_argument("--model_heads", type=type_tuple_list, default=[], help="Specific (model, head) combinations to plot. Format it as a Python list of tuples of strings, e.g. [('clmbr', 'lr'), ('count', 'gbm')]")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_RESULTS_DIR: str = args.path_to_results_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    SHOT_STRAT: str = args.shot_strat
    MODEL_HEADS: Optional[List[Tuple[str, str]]] = args.model_heads if len(args.model_heads) > 0 else None
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    
    # Load all results from CSVs
    dfs: List[pd.DataFrame] = []
    for idx, labeling_function in tqdm(enumerate(LABELING_FUNCTION_2_PAPER_NAME.keys())):
        path_to_csv = os.path.join(PATH_TO_RESULTS_DIR, f"{labeling_function}/{SHOT_STRAT}_results.csv")
        if not os.path.exists(path_to_csv): 
            print("Skipping ", labeling_function)
            continue
        dfs.append(pd.read_csv(path_to_csv))
    df_results: pd.DataFrame = pd.concat(dfs, ignore_index=True)

    
    ####################################
    ####################################
    #
    # Tables
    #
    ####################################
    ####################################
    
    df_means = df_results.groupby([
        'labeling_function',
        'sub_task',
        'model',
        'head',
        'score',
        'k',
    ]).agg({
        'value' : 'mean',
        'k' : 'first',
        'labeling_function' : 'first',
        'sub_task' : 'first',
        'model' : 'first',
        'head' : 'first',
        'score' : 'first',
    }).reset_index(drop = True)
    df_stds = df_results.groupby([
        'labeling_function',
        'sub_task',
        'model',
        'head',
        'score',
        'k',
    ]).agg({
        'value' : 'std',
        'k' : 'first',
        'labeling_function' : 'first',
        'sub_task' : 'first',
        'model' : 'first',
        'head' : 'first',
        'score' : 'first',
    }).reset_index(drop = True).fillna(0)
    
    # Table for each (labeling function, score)
    #   Rows = model + head
    #   Columns = k
    #   Cells = mean ± std of score
    for score in df_means['score'].unique():
        path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'individual_tasks', score)
        for sub_task in df_means['sub_task'].unique():
            os.makedirs(path_to_output_dir_, exist_ok=True)
            df_ = filter_df(df_means, sub_tasks=[sub_task], score=score, model_heads=MODEL_HEADS).sort_values(by=['model', 'head', 'k'])
            df_ = df_.rename(columns = {'value' : 'mean' })
            df_std_ = df_stds[(df_stds['sub_task'] == sub_task) & (df_stds['score'] == score)].sort_values(by=['model', 'head', 'k'])
            df_['std'] = df_std_['value']
            # Save raw df
            df_.to_csv(os.path.join(path_to_output_dir_, f'{sub_task}_raw.csv'), index=False)
            # Save pretty df
            df_ = df_.drop(columns = ['score', 'sub_task', 'labeling_function'])
            df_['value'] = df_['mean'].round(3).astype(str) + ' ± ' + df_['std'].round(3).astype(str)
            df_ = df_.drop(columns=['mean', 'std'])
            df_ = df_.pivot(index=['model', 'head'], columns='k', values='value').reset_index()
            df_.columns = [ str(x) for x in df_.columns ]
            df_ = df_.rename(columns={'-1' : 'All'})
            df_.to_csv(os.path.join(path_to_output_dir_, f'{sub_task}_pretty.csv'), index=False)
            # Create HTML Table with multicolumn header
            df_['model'] = df_['model'] + ' - ' + df_['head']
            df_ = df_.drop(columns=['head'])
            df_.columns = pd.MultiIndex.from_tuples([
                ('Model', ''),
                ('All', ''),
            ] + [ ('K', x) for x in df_.columns[2:] ])
            df_.to_html(os.path.join(path_to_output_dir_, f'{sub_task}_pretty.html'), classes=['leaderboard_table'], index=False)
        # Merge together all HTML tables for easy copying
        merge_html_tables(path_to_output_dir_)

    # Table for each (task group, score)
    #   Rows = model + head
    #   Columns = k
    #   Cells = mean ± std of score
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    for score in df_means['score'].unique():
        path_to_output_dir_: str = os.path.join(PATH_TO_OUTPUT_DIR, 'task_groups', score)
        for task_group in task_groups:
            os.makedirs(path_to_output_dir_, exist_ok=True)
            df_ = filter_df(df_means, task_group=task_group, score=score, model_heads=MODEL_HEADS)
            # Do another round of averaging over all subtasks:
            df_ = df_.groupby([
                'model',
                'head',
                'k',
            ]).agg({
                'value' : 'mean',
                'k' : 'first',
                'labeling_function' : 'first',
                'sub_task' : 'first',
                'model' : 'first',
                'head' : 'first',
                'score' : 'first'
            }).reset_index(drop = True)
            df_ = df_.rename(columns = {'value' : 'mean' })
            # Save raw df
            df_.to_csv(os.path.join(path_to_output_dir_, f'{task_group}_raw.csv'), index=False)
            # Save pretty df
            df_ = df_.drop(columns = ['score', 'sub_task', 'labeling_function'])
            df_['value'] = df_['mean'].round(3).astype(str)
            df_ = df_.drop(columns=['mean', ])
            df_ = df_.pivot(index=['model', 'head'], columns='k', values='value').reset_index()
            df_.columns = [ str(x) for x in df_.columns ]
            df_ = df_.rename(columns={'-1' : 'All'})
            df_.to_csv(os.path.join(path_to_output_dir_, f'{task_group}_pretty.csv'), index=False)
            # Create HTML Table with multicolumn header
            df_['model'] = df_['model'] + ' - ' + df_['head']
            df_ = df_.drop(columns=['head'])
            df_.columns = pd.MultiIndex.from_tuples([
                ('Model', ''),
                ('All', ''),
            ] + [ ('K', x) for x in df_.columns[2:] ])
            df_.to_html(os.path.join(path_to_output_dir_, f'{task_group}_pretty.html'), classes=['leaderboard_table'], index=False)
        # Merge together all HTML tables for easy copying
        merge_html_tables(path_to_output_dir_)
    
    ####################################
    ####################################
    #
    # Plots
    #
    ####################################
    ####################################

    print("Plotting Models: ", MODEL_HEADS)

    # Plotting individual AUROC/AUPRC plot for each labeling function
    for score in tqdm(df_results['score'].unique(), desc='plot_all_labeling_functions()'):
        if score == 'brier': continue
        plot_all_labeling_functions(df_results, score, PATH_TO_OUTPUT_DIR, 
                                    model_heads=MODEL_HEADS, is_x_scale_log=True, is_std_bars=True)

    # Plotting aggregated auroc and auprc plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_groups()'):
        if score == 'brier': continue
        plot_all_task_groups(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR, 
                             model_heads=MODEL_HEADS, is_x_scale_log=True)

    # plotting aggregated auroc and auprc box plots by task groups
    for score in tqdm(df_results['score'].unique(), desc='plot_all_task_group_box_plots()'):
        if score == 'brier': continue
        plot_all_task_group_box_plots(df_results, score, path_to_output_dir=PATH_TO_OUTPUT_DIR,
                                      model_heads=MODEL_HEADS)

    ####################################
    ####################################
    #
    # ClinicalBERT-specific Plots
    #
    ####################################
    ####################################
    
    print("Creating ClinicalBERT-specific visualizations...")
    
    # Create directory for ClinicalBERT plots
    clinicalbert_dir = os.path.join(PATH_TO_OUTPUT_DIR, 'clinicalbert')
    os.makedirs(clinicalbert_dir, exist_ok=True)
    
    # Plot ClinicalBERT type comparisons
    for score in tqdm(df_results['score'].unique(), desc='plot_clinicalbert_type_comparison()'):
        if score == 'brier': continue
        plot_clinicalbert_comparison_by_type(df_results, score, clinicalbert_dir)
    
    # Plot best ClinicalBERT models
    for score in tqdm(df_results['score'].unique(), desc='plot_best_clinicalbert_models()'):
        if score == 'brier': continue
        plot_best_clinicalbert_models(df_results, score, clinicalbert_dir, top_n=10)
    
    # Plot ClinicalBERT vs baseline comparison
    for score in tqdm(df_results['score'].unique(), desc='plot_clinicalbert_vs_baseline()'):
        if score == 'brier': continue
        plot_clinicalbert_vs_baseline_comparison(df_results, score, clinicalbert_dir)
    
    # Plot pooling strategy comparisons for each ClinicalBERT type
    for cb_type in ['type1', 'type2', 'type3']:
        for score in tqdm(df_results['score'].unique(), desc=f'plot_clinicalbert_{cb_type}_pooling()'):
            if score == 'brier': continue
            plot_clinicalbert_pooling_comparison(df_results, score, clinicalbert_dir, cb_type)
    
    # Plot embedding comparison using shared heads
    for score in tqdm(df_results['score'].unique(), desc='plot_embedding_comparison_shared_heads()'):
        if score == 'brier': continue
        plot_embedding_comparison_shared_heads(df_results, score, clinicalbert_dir)
    
    # Plot linear vs non-linear prediction heads
    for score in tqdm(df_results['score'].unique(), desc='plot_linear_vs_nonlinear_heads()'):
        if score == 'brier': continue
        plot_linear_vs_nonlinear_heads(df_results, score, clinicalbert_dir)
    
    # Plot detailed task embedding comparison
    for score in tqdm(df_results['score'].unique(), desc='plot_detailed_task_embedding_comparison()'):
        if score == 'brier': continue
        plot_detailed_task_embedding_comparison(df_results, score, clinicalbert_dir)
    
    print(f"ClinicalBERT plots saved to {clinicalbert_dir}")