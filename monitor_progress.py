#!/usr/bin/env python3
"""
Progress Monitor for EHRSHOT Dimensionality Reduction Analysis

This script monitors the checkpoint files and provides real-time progress updates.
"""

import os
import json
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta
import sys

def load_checkpoint(output_dir):
    """Load the latest checkpoint data."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_completed.json')
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def load_results(output_dir):
    """Load the current results CSV."""
    results_path = os.path.join(output_dir, 'dimensionality_results.csv')
    
    if not os.path.exists(results_path):
        return pd.DataFrame()
    
    try:
        return pd.read_csv(results_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return pd.DataFrame()

def estimate_completion_time(checkpoint_data, total_combinations):
    """Estimate when the analysis will complete."""
    completed = len(checkpoint_data.get('completed_combinations', []))
    
    if completed == 0:
        return "Unknown"
    
    last_update = datetime.fromisoformat(checkpoint_data['last_update'])
    now = datetime.now()
    
    # Estimate based on current progress
    progress_pct = completed / total_combinations
    elapsed_time = now - last_update
    
    if progress_pct > 0:
        total_estimated_time = elapsed_time / progress_pct
        remaining_time = total_estimated_time - elapsed_time
        
        if remaining_time.total_seconds() > 0:
            completion_time = now + remaining_time
            return completion_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return "Unknown"

def print_progress_summary(output_dir, watch=False):
    """Print a comprehensive progress summary."""
    
    # Total expected combinations
    tasks = ['guo_los', 'guo_readmission', 'guo_icu', 'lab_thrombocytopenia', 
             'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia',
             'new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 
             'new_lupus', 'new_acutemi']
    models = ['clmbr', 'clinicalbert_type3_clinicalbert_pool']
    methods = ['pca', 'umap']
    dimensions = [2, 5, 10, 25, 50, 100, 200, 400]
    
    while True:
        # Clear screen for watch mode
        if watch:
            os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔍 EHRSHOT Dimensionality Reduction Progress Monitor")
        print("=" * 60)
        print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load checkpoint data
        checkpoint_data = load_checkpoint(output_dir)
        results_df = load_results(output_dir)
        
        # Check if baselines are being skipped by looking at completed combinations
        has_baseline = False
        if checkpoint_data:
            has_baseline = any('_none_' in combo for combo in checkpoint_data.get('completed_combinations', []))
        
        if has_baseline:
            total_combinations = len(tasks) * len(models) * (len(methods) * len(dimensions) + 1)  # +1 for baseline
        else:
            total_combinations = len(tasks) * len(models) * len(methods) * len(dimensions)  # No baseline
        
        if checkpoint_data is None:
            print("❌ No checkpoint file found. Analysis may not have started yet.")
            if not watch:
                return
            time.sleep(30)
            continue
        
        # Basic progress info
        completed = len(checkpoint_data.get('completed_combinations', []))
        total_results = checkpoint_data.get('total_results', 0)
        progress_pct = (completed / total_combinations) * 100
        
        print(f"\n📊 Overall Progress:")
        print(f"   Combinations completed: {completed}/{total_combinations} ({progress_pct:.1f}%)")
        print(f"   Total result entries: {total_results}")
        
        if not has_baseline:
            print("   🚫 Baseline evaluation skipped (as configured)")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * completed // total_combinations)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"   Progress: |{bar}| {progress_pct:.1f}%")
        
        # Current status
        current_task = checkpoint_data.get('current_task', 'Unknown')
        current_progress = checkpoint_data.get('current_progress', {})
        last_update = checkpoint_data.get('last_update', 'Unknown')
        
        print(f"\n🎯 Current Status:")
        print(f"   Current task: {current_task}")
        if current_progress:
            if 'model' in current_progress:
                print(f"   Current model: {current_progress.get('model', 'Unknown')}")
            if 'method' in current_progress:
                print(f"   Current method: {current_progress.get('method', 'Unknown')}")
            if 'dimension' in current_progress:
                print(f"   Current dimension: {current_progress.get('dimension', 'Unknown')}")
            if 'status' in current_progress:
                print(f"   Status: {current_progress.get('status', 'Unknown')}")
        print(f"   Last update: {last_update}")
        
        # Time estimates
        completion_time = estimate_completion_time(checkpoint_data, total_combinations)
        print(f"\n⏰ Time Estimates:")
        print(f"   Estimated completion: {completion_time}")
        
        # Task breakdown
        if not results_df.empty:
            print(f"\n📋 Task Breakdown:")
            task_progress = {}
            for task in tasks:
                task_results = results_df[results_df['task'] == task]
                task_combinations = len(models) * (len(methods) * len(dimensions) + 1)
                task_completed = len(task_results) // 2  # Each combination has 2 metrics
                task_progress[task] = (task_completed, task_combinations)
                progress_pct = (task_completed / task_combinations) * 100 if task_combinations > 0 else 0
                print(f"   {task}: {task_completed}/{task_combinations} ({progress_pct:.1f}%)")
        
        # Recent results summary
        if not results_df.empty:
            print(f"\n📈 Recent Results Summary:")
            # Show best results so far
            for model in models:
                model_results = results_df[
                    (results_df['model'] == model) & 
                    (results_df['metric'] == 'auroc')
                ]
                if not model_results.empty:
                    best_score = model_results['score'].max()
                    best_result = model_results[model_results['score'] == best_score].iloc[0]
                    print(f"   {model} best AUROC: {best_score:.3f} ({best_result['method']} {best_result['dimension']}D)")
        
        # File sizes
        results_path = os.path.join(output_dir, 'dimensionality_results.csv')
        checkpoint_path = os.path.join(output_dir, 'checkpoint_completed.json')
        
        print(f"\n📁 Output Files:")
        if os.path.exists(results_path):
            size_mb = os.path.getsize(results_path) / (1024 * 1024)
            print(f"   Results CSV: {size_mb:.1f} MB")
        if os.path.exists(checkpoint_path):
            size_kb = os.path.getsize(checkpoint_path) / 1024
            print(f"   Checkpoint: {size_kb:.1f} KB")
        
        print("\n" + "=" * 60)
        
        if not watch:
            break
        
        print(f"🔄 Refreshing in 30 seconds... (Ctrl+C to exit)")
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
            break

def main():
    parser = argparse.ArgumentParser(description="Monitor EHRSHOT dimensionality reduction progress")
    parser.add_argument("--output_dir", default="../EHRSHOT_ASSETS/dimensionality_analysis",
                       help="Output directory containing checkpoint files")
    parser.add_argument("--watch", action="store_true",
                       help="Continuously monitor progress (refresh every 30s)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        print("   Make sure the analysis has started and the path is correct.")
        sys.exit(1)
    
    print_progress_summary(args.output_dir, args.watch)

if __name__ == "__main__":
    main() 