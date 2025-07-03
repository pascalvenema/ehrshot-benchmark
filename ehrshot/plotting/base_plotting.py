import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from ..utils import (
        LABELING_FUNCTION_2_PAPER_NAME,
        HEAD_2_INFO,
        MODEL_2_INFO, 
        TASK_GROUP_2_PAPER_NAME,
        SCORE_MODEL_HEAD_2_COLOR,
        filter_df,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import (
        LABELING_FUNCTION_2_PAPER_NAME,
        HEAD_2_INFO,
        MODEL_2_INFO, 
        TASK_GROUP_2_PAPER_NAME,
        SCORE_MODEL_HEAD_2_COLOR,
        filter_df,
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
    """
        Graph: Line plot of each model+head's results for a single labeling function as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            lines = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
    """
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
    """
        Graph: Aggregated line plot of each model+head's results for all of the labeling functions within a task group, as a function of `k`.
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = # of train examples per class (e.g. 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
            dark lines = model+head's achieved mean score across replicates, averaged across all labeling functions in this task group
            faded lines = model+head's achieved mean score across replicates for each individual labeling function
    """

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
    """
        Graph: Box plot showing distribution of scores for each model+head combination in this task group (at k=-1)
    
            y-axis = model+head's achieved mean score across replicates (e.g. AUROC/AUPRC)
            x-axis = model+head combinations
            box plots = distribution of model+head's score across labeling functions in this task group
    """

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

def plot_column_per_patient(df: pd.DataFrame, 
                          column: str, 
                          ax: plt.Axes, 
                          bins: int = 50,
                          patient_id_col: str = 'patient_id',
                          log_scale: bool = False) -> str:
    """
    Plot distribution of a column aggregated per patient.
    
    Args:
        df: DataFrame containing patient data
        column: Column to analyze (will count occurrences per patient)
        ax: Matplotlib axes to plot on
        bins: Number of histogram bins
        patient_id_col: Name of patient ID column
        log_scale: Whether to use log scale for y-axis
        
    Returns:
        Title string for the plot
    """
    # Count events per patient
    events_per_patient = df.groupby(patient_id_col).size()
    
    # Create visually appealing histogram
    ax.hist(events_per_patient.values, bins=bins, alpha=0.8, edgecolor='white', 
            color='#2E86AB', linewidth=0.8)
    
    # Calculate statistics
    mean_events = events_per_patient.mean()
    median_events = events_per_patient.median()
    total_patients = len(events_per_patient)
    
    # Enhanced visual formatting
    ax.set_xlabel('Number of Events per Patient', fontsize=12, fontweight='medium')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='medium')
    if log_scale:
        ax.set_yscale('log')
    
    # Fix x-axis: completely control ticks to avoid duplication
    ax.set_xlim(0, 40000)  # Set both left and right limits explicitly
    
    # Use locator to completely control tick positions
    from matplotlib.ticker import FixedLocator
    tick_positions = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    ax.xaxis.set_major_locator(FixedLocator(tick_positions))
    ax.tick_params(axis='x', which='minor', bottom=False)  # Remove minor ticks
    
    # Enhanced styling
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Subtle grid for better readability
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Generate improved title that clarifies the clamping
    title = f"Distribution of Event Counts per Patient (N={total_patients:,}, x-axis clamped at 40,000 for clarity)"
    
    return title 