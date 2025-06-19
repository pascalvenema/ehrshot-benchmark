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
    """Plot comparison of ClinicalBERT Type 1, 2, and 3 models with all tasks aggregated"""
    # Create a single figure instead of subplots
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter for only ClinicalBERT models and aggregate ALL tasks
    clinicalbert_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
    
    # Get data for ALL tasks (aggregated across all task groups)
    all_labeling_functions = []
    for task_group_functions in TASK_GROUP_2_LABELING_FUNCTION.values():
        all_labeling_functions.extend(task_group_functions)
    
    df_all_tasks = df_results[
        (df_results['score'] == score) & 
        (df_results['labeling_function'].isin(all_labeling_functions)) &
        (df_results['model'].isin(clinicalbert_models)) &
        (df_results['head'] == 'lr_lbfgs')  # Use only lr_lbfgs head
    ]
    
    if df_all_tasks.empty:
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_type_comparison_{score}.png"), dpi=300)
        plt.close('all')
        return fig
    
    # Handle "All" data point like in plot.py
    ks = sorted(df_all_tasks['k'].unique().tolist())
    x_tick_labels = [str(k) for k in ks]
    if -1 in ks:
        ks.remove(-1)
        full_data_k = 2 * max(ks) if ks else 256  # fallback if no few-shot data
        ks.append(full_data_k)
        x_tick_labels = [str(k) if k != full_data_k else 'All' for k in ks]
        # Update the data to use fake k value for plotting
        df_all_tasks = df_all_tasks.copy()
        df_all_tasks.loc[df_all_tasks['k'] == -1, 'k'] = full_data_k
        
    # Group by ClinicalBERT type - USE ONLY clinicalbert_pool strategy with lr_lbfgs head
    type1_data = df_all_tasks[df_all_tasks['model'] == 'clinicalbert_type1_clinicalbert_pool']
    type2_data = df_all_tasks[df_all_tasks['model'] == 'clinicalbert_type2_clinicalbert_pool']  
    type3_data = df_all_tasks[df_all_tasks['model'] == 'clinicalbert_type3_clinicalbert_pool']
    
    # Updated colors as requested: type1=blue, type2=green, type3=red
    # Use confidence interval lines instead of overlapping shaded areas
    for cb_type, cb_data, color, label in [
        ('type1', type1_data, '#1f77b4', 'ClinicalBERT Type 1 (Custom Pooling)+LR'),
        ('type2', type2_data, '#2ca02c', 'ClinicalBERT Type 2 (Custom Pooling)+LR'),
        ('type3', type3_data, '#d62728', 'ClinicalBERT Type 3 (Custom Pooling)+LR')
    ]:
        if cb_data.empty:
            continue
            
        # Group by k-value and compute mean/std across tasks for clinicalbert_pool + lr_lbfgs
        grouped = cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
        
        # Plot main line with thicker linewidth
        ax.plot(grouped['k'], grouped['mean'], 
               color=color, label=label, linewidth=3, marker='o', markersize=8)
        
        # Add thin confidence interval lines instead of overlapping shaded areas
        ax.plot(grouped['k'], grouped['mean'] + grouped['std'], 
               color=color, linewidth=1, alpha=0.6, linestyle='--')
        ax.plot(grouped['k'], grouped['mean'] - grouped['std'], 
               color=color, linewidth=1, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('# of Train Examples per Class', fontsize=14)
    ax.set_ylabel(f'Mean {score.upper()} Score Over All Tasks', fontsize=14)
    # Remove title as requested
    
    if is_x_scale_log:
        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.set_xticklabels(x_tick_labels)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')
    
    # Improve overall aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_type_comparison_{score}.png"), dpi=300, bbox_inches='tight')
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
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, classic red (matching type comparison)
    markers = ['o', 's', '^']
    
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
                           color=color, label=pool_strategy.replace('_', ' ').title() + '+LR', 
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

def plot_clinicalbert_pooling_comparison_mean_all_tasks(df_results: pd.DataFrame, 
                                                       score: str, 
                                                       path_to_output_dir: str,
                                                       cb_type: str = 'type3'):
    """Compare different pooling strategies for a specific ClinicalBERT type - Mean across ALL tasks"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get ALL labeling functions (aggregate across all task groups)
    all_labeling_functions = []
    for task_group_functions in TASK_GROUP_2_LABELING_FUNCTION.values():
        all_labeling_functions.extend(task_group_functions)
    
    pooling_strategies = ['max_pool', 'mean_pool', 'clinicalbert_pool']
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, classic red (matching type comparison)
    markers = ['o', 's', '^']
    
    # First get all data to determine k values for "All" handling
    all_model_data = df_results[
        (df_results['score'] == score) & 
        (df_results['labeling_function'].isin(all_labeling_functions)) &
        (df_results['head'] == 'lr_lbfgs')
    ]
    
    # Handle "All" data point like in plot.py
    ks = sorted(all_model_data['k'].unique().tolist())
    x_tick_labels = [str(k) for k in ks]
    if -1 in ks:
        ks.remove(-1)
        full_data_k = 2 * max(ks) if ks else 256  # fallback if no few-shot data
        ks.append(full_data_k)
        x_tick_labels = [str(k) if k != full_data_k else 'All' for k in ks]
    
    for pool_strategy, color, marker in zip(pooling_strategies, colors, markers):
        model_name = f'clinicalbert_{cb_type}_{pool_strategy}'
        model_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(all_labeling_functions)) &
            (df_results['model'] == model_name) &
            (df_results['head'] == 'lr_lbfgs')  # Only use LR prediction head
        ]
        
        if not model_data.empty:
            # Update k=-1 to fake k value for plotting if needed
            if -1 in model_data['k'].values:
                model_data = model_data.copy()
                model_data.loc[model_data['k'] == -1, 'k'] = full_data_k
            
            # Group by k-value and compute mean/std across ALL tasks
            grouped = model_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            
            # Create label with special handling for clinicalbert_pool
            if pool_strategy == 'clinicalbert_pool':
                label = 'ClinicalBERT Pool (Custom Pooling)+LR'
            else:
                label = pool_strategy.replace('_', ' ').title() + '+LR'
            
            # Plot main line
            ax.plot(grouped['k'], grouped['mean'], 
                   color=color, 
                   label=label, 
                   linewidth=3, 
                   marker=marker, 
                   markersize=8)
            
            # Add confidence interval lines instead of error bars for cleaner look
            ax.plot(grouped['k'], grouped['mean'] + grouped['std'], 
                   color=color, linewidth=1, alpha=0.6, linestyle='--')
            ax.plot(grouped['k'], grouped['mean'] - grouped['std'], 
                   color=color, linewidth=1, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('# of Train Examples per Class', fontsize=14)
    ax.set_ylabel(f'Mean {score.upper()} Score Over All Tasks', fontsize=14)
    # Title removed as requested
    
    ax.set_xscale('log')
    ax.set_xticks(ks)
    ax.set_xticklabels(x_tick_labels)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')
    
    # Improve overall aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_output_dir, f"clinicalbert_{cb_type}_pooling_mean_all_tasks_{score}.png"), dpi=300, bbox_inches='tight')
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
    """Create a single consolidated plot comparing CLMBR vs ClinicalBERT across all tasks"""
    
    # Get tasks with both CLMBR and ClinicalBERT data
    shared_heads = ['lr_lbfgs', 'knn']
    cb_model = 'clinicalbert_type3_clinicalbert_pool'  # Use type3 clinicalbert_pool as requested
    
    valid_tasks = []
    task_categories = []
    
    # Build task list with categories
    for task_group, tasks in TASK_GROUP_2_LABELING_FUNCTION.items():
        for task in tasks:
            clmbr_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'] == 'clmbr') &
                (df_results['head'].isin(shared_heads)) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)  # Full data only
            ]
            cb_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'] == cb_model) &
                (df_results['head'].isin(shared_heads)) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)
            ]
            if not clmbr_data.empty and not cb_data.empty:
                valid_tasks.append(task)
                task_categories.append(task_group)
    
    if len(valid_tasks) == 0:
        print(f"No valid tasks found for detailed comparison ({score})")
        return None
    
    # Create a single large plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Prepare data for plotting
    x_positions = np.arange(len(valid_tasks))
    bar_width = 0.35
    
    clmbr_best_scores = []
    cb_best_scores = []
    clmbr_best_heads = []
    cb_best_heads = []
    
    # For each task, find the best performing head for each model
    for task in valid_tasks:
        # CLMBR performance by head
        clmbr_performances = {}
        for head in shared_heads:
            clmbr_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'] == 'clmbr') &
                (df_results['head'] == head) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)
            ]
            if not clmbr_data.empty:
                clmbr_performances[head] = clmbr_data['value'].mean()
        
        # ClinicalBERT performance by head
        cb_performances = {}
        for head in shared_heads:
            cb_data = df_results[
                (df_results['labeling_function'] == task) &
                (df_results['model'] == cb_model) &
                (df_results['head'] == head) &
                (df_results['score'] == score) &
                (df_results['k'] == -1)
            ]
            if not cb_data.empty:
                cb_performances[head] = cb_data['value'].mean()
        
        # Find best performing heads
        if clmbr_performances:
            best_clmbr_head = max(clmbr_performances, key=clmbr_performances.get)
            clmbr_best_scores.append(clmbr_performances[best_clmbr_head])
            clmbr_best_heads.append(best_clmbr_head)
        else:
            clmbr_best_scores.append(0)
            clmbr_best_heads.append('N/A')
            
        if cb_performances:
            best_cb_head = max(cb_performances, key=cb_performances.get)
            cb_best_scores.append(cb_performances[best_cb_head])
            cb_best_heads.append(best_cb_head)
        else:
            cb_best_scores.append(0)
            cb_best_heads.append('N/A')
    
    # Create bars
    bars1 = ax.bar(x_positions - bar_width/2, clmbr_best_scores, bar_width, 
                   label='CLMBR (Best Head)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_positions + bar_width/2, cb_best_scores, bar_width, 
                   label='ClinicalBERT Type3 (Best Head)', color='lightcoral', alpha=0.8)
    
    # Add value labels on bars with best head information
    for i, (bar1, bar2, clmbr_head, cb_head) in enumerate(zip(bars1, bars2, clmbr_best_heads, cb_best_heads)):
        # CLMBR bars
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                f'{height1:.3f}\n({clmbr_head.replace("lr_lbfgs", "LR").replace("knn", "KNN")})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # ClinicalBERT bars  
        height2 = bar2.get_height()
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
                f'{height2:.3f}\n({cb_head.replace("lr_lbfgs", "LR").replace("knn", "KNN")})',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Customize x-axis with task names and categories
    task_labels = [LABELING_FUNCTION_2_PAPER_NAME.get(task, task) for task in valid_tasks]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(task_labels, rotation=45, ha='right')
    
    # Add category colors as background shading
    category_colors = {'operational_outcomes': 'lightblue', 'lab_values': 'lightgreen', 'new_diagnoses': 'lightyellow'}
    current_category = None
    start_idx = 0
    
    for i, category in enumerate(task_categories + [None]):  # Add None to trigger final category
        if category != current_category:
            if current_category is not None:
                # Shade the previous category
                ax.axvspan(start_idx - 0.5, i - 0.5, alpha=0.2, color=category_colors.get(current_category, 'lightgray'))
            current_category = category
            start_idx = i
    
    # Add category labels at the top
    category_positions = {}
    for i, category in enumerate(task_categories):
        if category not in category_positions:
            category_positions[category] = []
        category_positions[category].append(i)
    
    y_top = ax.get_ylim()[1]
    for category, positions in category_positions.items():
        center_pos = (min(positions) + max(positions)) / 2
        category_name = TASK_GROUP_2_PAPER_NAME.get(category, category).replace(' ', '\n')
        ax.text(center_pos, y_top * 1.15, category_name, ha='center', va='center', 
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=category_colors.get(category, 'lightgray'), alpha=0.7))
    
    # Formatting
    ax.set_ylabel(f'{score.upper()}', fontsize=14, fontweight='bold')
    ax.set_title(f'CLMBR vs ClinicalBERT Type3 Performance by Task\n(Best Prediction Head Shown for Each Model)', 
                 fontsize=16, fontweight='bold', pad=40)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=12)
    
    # Improve aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(clmbr_best_scores), max(cb_best_scores)) * 1.25)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(os.path.join(path_to_output_dir, f"detailed_task_embedding_comparison_{score}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig

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
    cb_style = {'color': 'lightcoral', 'linestyle': '-', 'marker': '^', 'label': 'Best ClinicalBERT + LR'}
    
    for idx, task_group in enumerate(task_groups):
        ax = axes[idx] if n_groups > 1 else axes
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Plot CLMBR performance with LR head
        clmbr_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'] == 'clmbr') &
            (df_results['head'] == lr_head)
        ]
        
        if not clmbr_data.empty:
            clmbr_grouped = clmbr_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            ax.errorbar(clmbr_grouped['k'], clmbr_grouped['mean'], yerr=clmbr_grouped['std'], 
                       color=clmbr_style['color'], 
                       linestyle=clmbr_style['linestyle'],
                       marker=clmbr_style['marker'], 
                       label=clmbr_style['label'],
                       linewidth=2.5, markersize=8, alpha=0.9)
        
        # Plot best ClinicalBERT performance with LR head
        cb_models = [model for model in df_results['model'].unique() if 'clinicalbert' in model]
        
        cb_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions)) &
            (df_results['model'].isin(cb_models)) &
            (df_results['head'] == lr_head)
        ]
        
        if not cb_data.empty:
            # Find best ClinicalBERT model for this head at full data
            full_data_cb = cb_data[cb_data['k'] == -1]
            if not full_data_cb.empty:
                best_cb_model = full_data_cb.groupby('model')['value'].mean().idxmax()
                best_cb_data = cb_data[cb_data['model'] == best_cb_model]
                cb_grouped = best_cb_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
                
                ax.errorbar(cb_grouped['k'], cb_grouped['mean'], yerr=cb_grouped['std'], 
                           color=cb_style['color'],
                           linestyle=cb_style['linestyle'],
                           marker=cb_style['marker'],
                           label=cb_style['label'],
                           linewidth=2.5, markersize=8, alpha=0.9)
        
        ax.set_xlabel('K (Number of Training Examples)', fontsize=12)
        ax.set_ylabel(f'{score.upper()}', fontsize=12)
        ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128'])
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
    
    fig.suptitle(f'CLMBR vs ClinicalBERT Embeddings - Logistic Regression Head - {score.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(path_to_output_dir, f"embedding_comparison_lr_only_{score}.png"), 
                dpi=300, bbox_inches='tight')
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
    
    # Plot pooling strategy comparisons for each ClinicalBERT type - Mean across ALL tasks
    for cb_type in ['type1', 'type2', 'type3']:
        for score in tqdm(df_results['score'].unique(), desc=f'plot_clinicalbert_{cb_type}_pooling_mean_all_tasks()'):
            if score == 'brier': continue
            plot_clinicalbert_pooling_comparison_mean_all_tasks(df_results, score, clinicalbert_dir, cb_type)
    
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
    
    # Plot embedding comparison using only LR prediction head
    for score in tqdm(df_results['score'].unique(), desc='plot_embedding_comparison_lr_only()'):
        if score == 'brier': continue
        plot_embedding_comparison_lr_only(df_results, score, clinicalbert_dir)
    
    print(f"ClinicalBERT plots saved to {clinicalbert_dir}")