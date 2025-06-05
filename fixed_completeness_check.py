#!/usr/bin/env python3
"""
Fixed EHRSHOT Results Completeness Check
This script properly analyzes the completeness without wrong assumptions.
"""

import os
import pandas as pd
from collections import defaultdict

# Define core expected models based on actual analysis
CORE_EXPECTED_MODELS = {
    'count,gbm',
    'count,lr_lbfgs', 
    'count,rf',
    'clmbr,lr_lbfgs'
}

# Define expected clinicalbert models (type1, type2, type3)
EXPECTED_CLINICALBERT_MODELS = {
    # Type 1
    'clinicalbert_type1_clinicalbert_pool,knn',
    'clinicalbert_type1_clinicalbert_pool,lr_lbfgs', 
    'clinicalbert_type1_clinicalbert_pool,rf',
    'clinicalbert_type1_max_pool,knn',
    'clinicalbert_type1_max_pool,lr_lbfgs',
    'clinicalbert_type1_max_pool,rf',
    'clinicalbert_type1_mean_pool,knn',
    'clinicalbert_type1_mean_pool,lr_lbfgs',
    'clinicalbert_type1_mean_pool,rf',
    # Type 2
    'clinicalbert_type2_clinicalbert_pool,knn',
    'clinicalbert_type2_clinicalbert_pool,lr_lbfgs',
    'clinicalbert_type2_clinicalbert_pool,rf',
    'clinicalbert_type2_max_pool,knn',
    'clinicalbert_type2_max_pool,lr_lbfgs',
    'clinicalbert_type2_max_pool,rf',
    'clinicalbert_type2_mean_pool,knn',
    'clinicalbert_type2_mean_pool,lr_lbfgs',
    'clinicalbert_type2_mean_pool,rf',
    # Type 3
    'clinicalbert_type3_clinicalbert_pool,knn',
    'clinicalbert_type3_clinicalbert_pool,lr_lbfgs',
    'clinicalbert_type3_clinicalbert_pool,rf',
    'clinicalbert_type3_max_pool,knn',
    'clinicalbert_type3_max_pool,lr_lbfgs',
    'clinicalbert_type3_max_pool,rf',
    'clinicalbert_type3_mean_pool,knn',
    'clinicalbert_type3_mean_pool,lr_lbfgs',
    'clinicalbert_type3_mean_pool,rf'
}

# Expected metrics and other parameters
EXPECTED_METRICS = ['auroc', 'auprc', 'brier']
EXPECTED_K_VALUES = [-1, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128]
EXPECTED_REPLICATES = [0, 1, 2, 3, 4]

def get_model_columns(df):
    """Determine which columns contain model and head info based on CSV format."""
    if 'Unnamed: 0.1' in df.columns:
        # Format with extra unnamed columns (guo_/new_ tasks)
        return 'model', 'head'  # These will be at positions 5,6 but pandas handles by name
    else:
        # Standard format (lab_/chexpert tasks)  
        return 'model', 'head'  # These will be at positions 3,4 but pandas handles by name

def analyze_task_completeness(task_name, results_dir):
    """Analyze completeness of a single task."""
    
    results_file = os.path.join(results_dir, task_name, 'all_results.csv')
    
    if not os.path.exists(results_file):
        return {
            'status': 'MISSING',
            'details': f"No results file found",
            'models': [],
            'metrics': [],
            'k_values': [],
            'total_rows': 0
        }
    
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        return {
            'status': 'ERROR',
            'details': f"Error reading file: {e}",
            'models': [],
            'metrics': [],
            'k_values': [],
            'total_rows': 0
        }
    
    # Check if required columns exist
    if 'model' not in df.columns or 'head' not in df.columns:
        return {
            'status': 'ERROR', 
            'details': f"Missing model/head columns. Available: {list(df.columns)}",
            'models': [],
            'metrics': [],
            'k_values': [],
            'total_rows': 0
        }
    
    # Get available models, metrics, k-values
    model_combinations = set(df['model'] + ',' + df['head'])
    available_metrics = set(df['score'].unique()) if 'score' in df.columns else set()
    available_k_values = set(df['k'].unique()) if 'k' in df.columns else set()
    
    # Check for core expected models
    missing_core_models = CORE_EXPECTED_MODELS - model_combinations
    has_core_models = len(missing_core_models) == 0
    
    # Check for clinicalbert models
    available_clinicalbert = model_combinations & EXPECTED_CLINICALBERT_MODELS
    missing_clinicalbert = EXPECTED_CLINICALBERT_MODELS - model_combinations
    
    # Count by type
    type1_available = len([m for m in available_clinicalbert if 'type1' in m])
    type2_available = len([m for m in available_clinicalbert if 'type2' in m])
    type3_available = len([m for m in available_clinicalbert if 'type3' in m])
    
    type1_expected = len([m for m in EXPECTED_CLINICALBERT_MODELS if 'type1' in m])
    type2_expected = len([m for m in EXPECTED_CLINICALBERT_MODELS if 'type2' in m])
    type3_expected = len([m for m in EXPECTED_CLINICALBERT_MODELS if 'type3' in m])
    
    # Check for expected metrics and k-values
    missing_metrics = set(EXPECTED_METRICS) - available_metrics
    missing_k_values = set(EXPECTED_K_VALUES) - available_k_values
    
    # Determine status
    if has_core_models and len(missing_metrics) == 0 and len(missing_k_values) == 0:
        status = 'COMPLETE'
        cb_info = f"CB: {type1_available}/{type1_expected} type1, {type2_available}/{type2_expected} type2, {type3_available}/{type3_expected} type3"
        additional_count = len(model_combinations) - len(CORE_EXPECTED_MODELS) - len(available_clinicalbert)
        details = f"All core models present + {len(available_clinicalbert)} clinicalbert + {additional_count} other models. {cb_info}"
    elif has_core_models:
        status = 'MOSTLY_COMPLETE'
        issues = []
        if missing_metrics:
            issues.append(f"missing metrics: {missing_metrics}")
        if missing_k_values:
            issues.append(f"missing k-values: {missing_k_values}")
        details = f"Core models present, but {'; '.join(issues)}"
    else:
        status = 'INCOMPLETE'
        details = f"Missing core models: {missing_core_models}"
    
    return {
        'status': status,
        'details': details,
        'models': sorted(list(model_combinations)),
        'core_models_present': sorted(list(CORE_EXPECTED_MODELS & model_combinations)),
        'additional_models': sorted(list(model_combinations - CORE_EXPECTED_MODELS)),
        'missing_core_models': sorted(list(missing_core_models)),
        'clinicalbert_models': sorted(list(available_clinicalbert)),
        'clinicalbert_type1': sorted([m for m in available_clinicalbert if 'type1' in m]),
        'clinicalbert_type2': sorted([m for m in available_clinicalbert if 'type2' in m]),
        'clinicalbert_type3': sorted([m for m in available_clinicalbert if 'type3' in m]),
        'metrics': sorted(list(available_metrics)),
        'k_values': sorted(list(available_k_values)),
        'total_rows': len(df),
        'has_core_models': has_core_models
    }

def main():
    results_dir = 'EHRSHOT_ASSETS/results'
    
    print("🔍 EHRSHOT Results Completeness Check (Fixed)")
    print("=" * 55)
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Get all available tasks
    available_tasks = []
    for item in os.listdir(results_dir):
        if os.path.isdir(os.path.join(results_dir, item)) and not item.startswith('.'):
            available_tasks.append(item)
    
    print(f"📊 Found {len(available_tasks)} tasks to analyze")
    print(f"🎯 Core expected models: {sorted(CORE_EXPECTED_MODELS)}")
    print(f"🧠 ClinicalBERT models: {len(EXPECTED_CLINICALBERT_MODELS)} total (9 each for types 1, 2, 3)")
    print()
    
    # Analyze each task
    results = {}
    status_counts = defaultdict(int)
    
    for task in sorted(available_tasks):
        result = analyze_task_completeness(task, results_dir)
        results[task] = result
        status_counts[result['status']] += 1
        
        # Status icon
        status_icons = {
            'COMPLETE': '✅',
            'MOSTLY_COMPLETE': '🟡',
            'INCOMPLETE': '⚠️',
            'MISSING': '❌',
            'ERROR': '💥'
        }
        icon = status_icons.get(result['status'], '❓')
        
        # Count clinicalbert models by type
        cb1_count = len(result['clinicalbert_type1'])
        cb2_count = len(result['clinicalbert_type2'])
        cb3_count = len(result['clinicalbert_type3'])
        cb_summary = f"CB: {cb1_count}/9, {cb2_count}/9, {cb3_count}/9"
        
        print(f"{icon} {task:<25} {result['status']:<15} {result['total_rows']:>6} rows  {cb_summary}")
        
        # Show details for non-complete tasks
        if result['status'] != 'COMPLETE':
            print(f"   └─ {result['details']}")
        
        # Show model breakdown for complete tasks  
        if result['status'] == 'COMPLETE':
            core_count = len(result['core_models_present'])
            cb_total = len(result['clinicalbert_models'])
            other_count = len(result['additional_models']) - cb_total
            print(f"   └─ {core_count} core + {cb_total} clinicalbert + {other_count} other models")
        
        print()
    
    # Summary
    print("📈 SUMMARY")
    print("=" * 20)
    total_tasks = len(available_tasks)
    
    for status, count in sorted(status_counts.items()):
        icon = status_icons.get(status, '❓')
        percentage = (count / total_tasks) * 100
        print(f"{icon} {status:<15}: {count:>3} tasks ({percentage:>5.1f}%)")
    
    # Overall completion rate (complete + mostly complete)
    complete_count = status_counts['COMPLETE'] + status_counts['MOSTLY_COMPLETE']
    completion_rate = (complete_count / total_tasks) * 100
    print(f"🎯 Overall completion: {complete_count}/{total_tasks} tasks ({completion_rate:.1f}%)")
    
    # ClinicalBERT Analysis
    print(f"\n🧠 CLINICALBERT ANALYSIS")
    print("=" * 30)
    
    # Count how many tasks have each type
    type1_tasks = sum(1 for r in results.values() if len(r['clinicalbert_type1']) > 0)
    type2_tasks = sum(1 for r in results.values() if len(r['clinicalbert_type2']) > 0)
    type3_tasks = sum(1 for r in results.values() if len(r['clinicalbert_type3']) > 0)
    
    print(f"📊 Tasks with ClinicalBERT models:")
    print(f"   Type 1: {type1_tasks}/{total_tasks} tasks ({100*type1_tasks/total_tasks:.1f}%)")
    print(f"   Type 2: {type2_tasks}/{total_tasks} tasks ({100*type2_tasks/total_tasks:.1f}%)")
    print(f"   Type 3: {type3_tasks}/{total_tasks} tasks ({100*type3_tasks/total_tasks:.1f}%)")
    
    # Show example of complete clinicalbert coverage
    complete_cb_tasks = [task for task, result in results.items() 
                        if len(result['clinicalbert_type1']) == 9 and 
                           len(result['clinicalbert_type2']) == 9 and 
                           len(result['clinicalbert_type3']) == 9]
    
    if complete_cb_tasks:
        print(f"\n✅ Tasks with complete ClinicalBERT coverage: {len(complete_cb_tasks)}")
        print(f"   Examples: {', '.join(complete_cb_tasks[:3])}")
    
    # Show some examples
    complete_tasks = [task for task, result in results.items() if result['status'] == 'COMPLETE']
    if complete_tasks:
        print(f"\n🔬 EXAMPLE COMPLETE TASK: {complete_tasks[0]}")
        print("=" * 35)
        example = results[complete_tasks[0]]
        print(f"Core models: {example['core_models_present']}")
        print(f"ClinicalBERT Type 1: {len(example['clinicalbert_type1'])}/9 models")
        print(f"ClinicalBERT Type 2: {len(example['clinicalbert_type2'])}/9 models")  
        print(f"ClinicalBERT Type 3: {len(example['clinicalbert_type3'])}/9 models")
        other_models = set(example['additional_models']) - set(example['clinicalbert_models'])
        if other_models:
            print(f"Other models: {len(other_models)} additional")
        print(f"Metrics: {example['metrics']}")
        print(f"K-values: {len(example['k_values'])} values from {min(example['k_values'])} to {max(example['k_values'])}")

if __name__ == '__main__':
    main() 