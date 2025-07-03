#!/usr/bin/env python3
"""
EHRSHOT Essential Plots Generator
Generates only the specific plots needed by the user.
"""

import os
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

def _plot_unified_legend(fig, axes, ncol=None, fontsize=14):
    """Create a unified legend for the entire figure."""
    labels = []
    label2handle = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                label2handle[l] = h
    legend_n_col: int = len([ x for x in labels if '(Full)' not in x ]) if ncol is None else ncol
    fig.legend([ label2handle[l] for l in labels ], labels, loc='lower center', ncol=legend_n_col, fontsize=fontsize)

def plot_one_labeling_function(df: pd.DataFrame,
                                ax: plt.Axes,
                                labeling_function: str,
                                sub_tasks: List[str],
                                score: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None,
                                is_x_scale_log: bool = True,
                                is_std_bars: bool = True):
    """Line plot of each model+head's results for a single labeling function as a function of `k`."""
    # Limit to specific labeling_function, subtask, score, (model, head) combos
    df = filter_df(df, score=score, labeling_function=labeling_function, sub_tasks=sub_tasks, model_heads=model_heads)
    
    if df.shape[0] == 0:
        print(f"Skipping {labeling_function} because no results for {model_heads}")
        return

    if labeling_function == 'new_celiac':
        # Only 62 train examples, so cutoff plot at `k = 64`
        df = df[df['k'] <= 64]

    # Get all `k` shots tested
    ks: List[int] = sorted(df['k'].unique().tolist())
    
    # Create a fake `k` for the full data which is 2x the max `k` in the few-shot data
    x_tick_labels: List[str] = ks
    if -1 in ks:
        ks.remove(-1)
        full_data_k: int = 2 * max(ks)
        ks.append(full_data_k)
        df.loc[df['k'] == -1, 'k'] = full_data_k
        x_tick_labels = [ str(k) if k != full_data_k else 'All' for k in ks ]
    
    df_means = df.groupby([
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
    df_stds = df.groupby([
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
    }).reset_index(drop = True)

    models: List[str] = df['model'].unique().tolist()
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            # Skip unsupported scores
            if score not in SCORE_MODEL_HEAD_2_COLOR:
                continue
            if model not in SCORE_MODEL_HEAD_2_COLOR[score]:
                continue
            if head not in SCORE_MODEL_HEAD_2_COLOR[score][model]:
                continue
                
            model_name: str = MODEL_2_INFO[model]['label']
            head_name: str = HEAD_2_INFO[head]['label']

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')
            df_stds_ = df_stds[(df_stds['model'] == model) & (df_stds['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual subtasks
            for subtask in df_means_['sub_task'].unique():
                df_m_ = df_means_[df_means_['sub_task'] == subtask]
                df_s_ = df_stds_[df_stds_['sub_task'] == subtask]
                ax.plot(df_m_['k'], df_m_['value'], color=color, linestyle='-', linewidth=2, alpha=0.25)
                if is_std_bars:
                    ax.plot(df_m_['k'], df_m_['value'] - df_s_['value'], color=color, alpha=0.1)
                    ax.plot(df_m_['k'], df_m_['value'] + df_s_['value'], color=color, alpha=0.1)
                    ax.fill_between(df_m_['k'], df_m_['value'] - df_s_['value'], df_m_['value'] + df_s_['value'],color=color, alpha=0.2)

            # Plot average line across all subtasks
            df_ = df_means_.groupby(['k']).agg({ 'value' : 'mean', 'k': 'first', }).reset_index(drop = True)
            ax.plot(df_['k'], df_['value'], color=color, label=f'{model_name}+{head_name}', linestyle='-', marker='o', linewidth=3, markersize=7)

    # Plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(LABELING_FUNCTION_2_PAPER_NAME[labeling_function], size=14)
    ax.set_ylabel(score.upper(), fontsize=10)
    ax.set_xlabel("# of Train Examples per Class", fontsize=10)
    ax.set_xticks(ks, ks)
    ax.set_xticklabels(x_tick_labels)

def plot_one_task_group(df: pd.DataFrame, 
                        ax: plt.Axes, 
                        task_group: str, 
                        score: str, 
                        model_heads: Optional[List[Tuple[str, str]]] = None, 
                        is_x_scale_log: bool = True):    
    """Aggregated line plot of each model+head's results for all labeling functions within a task group."""
    # Limit to specific task_group, score, (model, head) combos
    df = filter_df(df, score=score, task_group=task_group, model_heads=model_heads)

    if df.shape[0] == 0:
        print(f"Skipping {task_group} because no results for {model_heads}")
        return

    # Get all `k` shots tested
    ks: List[int] = sorted(df['k'].unique().tolist())
    
    # Create a fake `k` for the full data which is 2x the max `k` in the few-shot data
    x_tick_labels: List[str] = ks
    if -1 in ks:
        ks.remove(-1)
        full_data_k: int = 2 * max(ks)
        ks.append(full_data_k)
        df.loc[df['k'] == -1, 'k'] = full_data_k
        x_tick_labels = [ str(k) if k != full_data_k else 'All' for k in ks ]

    df_means = df.groupby([
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

    models: List[str] = df['model'].unique().tolist()
    for m_idx, model in enumerate(models):
        heads: List[str] = df[df['model'] == model]['head'].unique().tolist()
        for h_idx, head in enumerate(heads):
            # Skip unsupported scores
            if score not in SCORE_MODEL_HEAD_2_COLOR:
                continue
            if model not in SCORE_MODEL_HEAD_2_COLOR[score]:
                continue
            if head not in SCORE_MODEL_HEAD_2_COLOR[score][model]:
                continue
                
            model_name: str = MODEL_2_INFO[model]['label']
            head_name: str = HEAD_2_INFO[head]['label']

            df_means_ = df_means[(df_means['model'] == model) & (df_means['head'] == head)].sort_values(by='k')

            # Color
            color: str = SCORE_MODEL_HEAD_2_COLOR[score][model][head]

            # Plot individual labeling functions (faded lines)
            for labeling_function in df_means_['labeling_function'].unique():
                df_lf_ = df_means_[df_means_['labeling_function'] == labeling_function]
                df_lf_mean = df_lf_.groupby(['k']).agg({ 'value' : 'mean', 'k': 'first', }).reset_index(drop = True)
                ax.plot(df_lf_mean['k'], df_lf_mean['value'], color=color, linestyle='-', linewidth=1, alpha=0.4)

            # Plot average line across all labeling functions (dark line)
            df_ = df_means_.groupby(['k']).agg({ 'value' : 'mean', 'k': 'first', }).reset_index(drop = True)
            ax.plot(df_['k'], df_['value'], color=color, label=f'{model_name}+{head_name}', linestyle='-', marker='o', linewidth=3, markersize=7)

    # Plot aesthetics
    if is_x_scale_log:
        ax.set_xscale("log")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(TASK_GROUP_2_PAPER_NAME[task_group], size=14)
    ax.set_ylabel(score.upper(), fontsize=10)
    ax.set_xlabel("# of Train Examples per Class", fontsize=10)
    ax.set_xticks(ks, ks)
    ax.set_xticklabels(x_tick_labels)

def plot_one_task_group_box_plot(df: pd.DataFrame, 
                                ax: plt.Axes,
                                task_group: str, 
                                score: str,
                                model_heads: Optional[List[Tuple[str, str]]] = None):
    """Box plot showing distribution of scores for each model+head combination in this task group (at k=-1)"""
    # Limit to full data (k=-1) for a specific task_group, score, (model, head) combos
    df = filter_df(df, score=score, task_group=task_group, model_heads=model_heads, ks=[-1])

    if df.shape[0] == 0:
        print(f"Skipping {task_group} because no results for {model_heads}")
        return

    # Aggregate results at the labeling function level (mean across subtasks and replicates)
    df_agg = df.groupby([
        'labeling_function',
        'model',
        'head',
        'score',
    ]).agg({
        'value' : 'mean',
        'labeling_function' : 'first',
        'model' : 'first',
        'head' : 'first',
        'score' : 'first',
    }).reset_index(drop = True)

    # Prepare data for box plot
    model_head_combinations = df_agg[['model', 'head']].drop_duplicates()
    box_data = []
    labels = []
    colors = []

    for _, row in model_head_combinations.iterrows():
        model, head = row['model'], row['head']
        
        # Skip unsupported scores
        if score not in SCORE_MODEL_HEAD_2_COLOR:
            continue
        if model not in SCORE_MODEL_HEAD_2_COLOR[score]:
            continue
        if head not in SCORE_MODEL_HEAD_2_COLOR[score][model]:
            continue
            
        model_head_data = df_agg[(df_agg['model'] == model) & (df_agg['head'] == head)]
        box_data.append(model_head_data['value'].tolist())
        
        model_name = MODEL_2_INFO[model]['label']
        head_name = HEAD_2_INFO[head]['label']
        labels.append(f'{model_name}+{head_name}')
        
        color = SCORE_MODEL_HEAD_2_COLOR[score][model][head]
        colors.append(color)

    if not box_data:
        print(f"No valid model-head combinations for {task_group}")
        return

    # Create box plot
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot aesthetics
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title(f'{TASK_GROUP_2_PAPER_NAME[task_group]} (Full Data)', size=14)
    ax.set_ylabel(score.upper(), fontsize=10)
    ax.grid(True, alpha=0.3)

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
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(path_to_output_dir, f"all_labeling_functions_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_groups(df_results: pd.DataFrame, 
                        score: str, 
                        path_to_output_dir: str,
                        model_heads: Optional[List[Tuple[str, str]]] = None,
                        is_x_scale_log: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(os.path.join(path_to_output_dir, f"all_task_groups_{score}.png"), dpi=300)
    plt.close('all')
    return fig

def plot_all_task_group_box_plots(df_results: pd.DataFrame,
                            score: str, 
                            path_to_output_dir: str,
                            model_heads: Optional[List[Tuple[str, str]]] = None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    task_groups: List[str] = list(TASK_GROUP_2_LABELING_FUNCTION.keys())
    for idx, task_group in enumerate(task_groups):
        plot_one_task_group_box_plot(df_results, 
                                    axes.flat[idx], 
                                    task_group, 
                                    score,
                                    model_heads=model_heads)
    
    # Plot aesthetics
    fig.suptitle(f'{score.upper()} Distribution by Task Group (Full Data)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(path_to_output_dir, f"all_task_groups_box_{score}.png"), dpi=300)
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
    
    # Get ALL labeling functions (aggregate across all task groups)
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
        ('type1', type1_data, '#1f77b4', 'ClinicalBERT "codes" (type 1) + LR'),
        ('type2', type2_data, '#2ca02c', 'ClinicalBERT "descriptions" (type 2) + LR'),
        ('type3', type3_data, '#d62728', 'ClinicalBERT "human_narrative" (type 3) + LR')
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
    ax.set_ylabel(f'{score.upper()}', fontsize=14)
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
    
    for pool_idx, (pooling, color, marker) in enumerate(zip(pooling_strategies, colors, markers)):
        model_name = f'clinicalbert_{cb_type}_{pooling}'
        
        # Get data for this model across ALL tasks
        model_data = df_results[
        (df_results['score'] == score) & 
        (df_results['labeling_function'].isin(all_labeling_functions)) &
            (df_results['model'] == model_name) &
            (df_results['head'] == 'lr_lbfgs')  # Use only lr_lbfgs head for consistency
        ]
        
        if model_data.empty:
            continue
        
        # Handle "All" data point
        ks = sorted(model_data['k'].unique().tolist())
        x_tick_labels = [str(k) for k in ks]
        if -1 in ks:
            ks.remove(-1)
            full_data_k = 2 * max(ks) if ks else 256
            ks.append(full_data_k)
            x_tick_labels = [str(k) if k != full_data_k else 'All' for k in ks]
            model_data = model_data.copy()
            model_data.loc[model_data['k'] == -1, 'k'] = full_data_k
            
        # Group by k-value and compute mean/std across ALL labeling functions
        grouped = model_data.groupby('k')['value'].agg(['mean', 'std']).reset_index()
            
        # Create label based on pooling strategy
        pool_label = pooling.replace('_', ' ').title()
        if pooling == 'clinicalbert_pool':
            pool_label = 'Custom Pooling'
            
        # Plot main line
        ax.plot(grouped['k'], grouped['mean'], 
               color=color, label=f'{pool_label}', 
               linewidth=3, marker=marker, markersize=8)
        
        # Add confidence interval lines
        ax.plot(grouped['k'], grouped['mean'] + grouped['std'], 
               color=color, linewidth=1, alpha=0.6, linestyle='--')
        ax.plot(grouped['k'], grouped['mean'] - grouped['std'], 
               color=color, linewidth=1, alpha=0.6, linestyle='--')
    
    ax.set_xlabel('# of Train Examples per Class', fontsize=14)
    ax.set_ylabel(f'Mean {score.upper()} Score Over All Tasks', fontsize=14)
    ax.set_title(f'ClinicalBERT {cb_type.title()} Pooling Strategy Comparison', fontsize=16)
    
    ax.set_xscale('log')
    if 'ks' in locals():
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

def plot_detailed_task_embedding_comparison(df_results: pd.DataFrame, 
                                          score: str, 
                                          path_to_output_dir: str):
    """Create a single consolidated plot comparing CLMBR vs ClinicalBERT Type3 Custom Pooling across all tasks"""
    
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
        
        # ClinicalBERT Type3 Custom Pooling performance by head
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
                   label='CLMBR (Best Prediction Head)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x_positions + bar_width/2, cb_best_scores, bar_width, 
                   label='ClinicalBERT Type3 Custom Pooling (Best Prediction Head)', color='lightcoral', alpha=0.8)
    
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
    
    # Formatting with explicit labeling
    ax.set_ylabel(f'{score.upper()}', fontsize=14, fontweight='bold')
    # Title removed as requested
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', fontsize=11)
    
    # Improve aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(clmbr_best_scores), max(cb_best_scores)) * 1.25)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.15)
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
    cb_style = {'color': 'lightcoral', 'linestyle': '-', 'marker': '^', 'label': 'ClinicalBERT Type 3 (Custom Pooling) + LR'}
    
    for idx, task_group in enumerate(task_groups):
        ax = axes[idx] if n_groups > 1 else axes
        labeling_functions = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        
        # Get task-specific data for k-value handling
        task_data = df_results[
            (df_results['score'] == score) & 
            (df_results['labeling_function'].isin(labeling_functions))
        ]
        
        # Handle "All" data point like in other functions
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
        
        # Plot ClinicalBERT Type 3 (clinicalbert_pool) performance with LR head
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
    
    fig.suptitle(f'CLMBR vs ClinicalBERT Embeddings - Logistic Regression Head - {score.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(path_to_output_dir, f"embedding_comparison_lr_only_{score}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close('all')
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate essential EHRSHOT plots")
    parser.add_argument("--path_to_results_dir", required=True, type=str,
                       help="Path to directory containing results")
    parser.add_argument("--path_to_output_dir", required=True, type=str,
                       help="Path to directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.path_to_output_dir, exist_ok=True)
    
    # Load results
    print("📊 Loading results...")
    dfs = []
    for labeling_function in tqdm(LABELING_FUNCTION_2_PAPER_NAME.keys()):
        path_to_csv = os.path.join(args.path_to_results_dir, f"{labeling_function}/all_results.csv")
        if os.path.exists(path_to_csv):
            dfs.append(pd.read_csv(path_to_csv))
    
    if not dfs:
        print("❌ No results files found!")
        return
    
    df_results = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(df_results)} result rows")
    
    # Skip unsupported scores (like old 'brier' scores)
    supported_scores = ['auroc', 'auprc']
    df_results = df_results[df_results['score'].isin(supported_scores)]
    
    # Generate all essential plots
    print("\n🎨 Generating essential plots...")
    
    for score in supported_scores:
        if score not in df_results['score'].unique():
            print(f"⚠️  {score.upper()} not found in results")
            continue
            
        print(f"\n📈 Creating {score.upper()} plots...")
        
        plot_clinicalbert_comparison_by_type(df_results, score, args.path_to_output_dir)
        plot_clinicalbert_pooling_comparison_mean_all_tasks(df_results, score, args.path_to_output_dir, 'type3')
        plot_detailed_task_embedding_comparison(df_results, score, args.path_to_output_dir)
        plot_embedding_comparison_lr_only(df_results, score, args.path_to_output_dir)
    
    print("\n🎉 All essential plots generated successfully!")

if __name__ == "__main__":
    main()