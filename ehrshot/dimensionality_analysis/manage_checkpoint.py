#!/usr/bin/env python3
"""
Checkpoint Management for EHRSHOT Dimensionality Reduction Analysis

This script allows you to manage the checkpoint file to:
- View what's been completed
- Clear specific tasks/models/methods for re-running
- Reset the entire checkpoint
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime

def load_checkpoint(output_dir):
    """Load the checkpoint file."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_completed.json')
    
    if not os.path.exists(checkpoint_path):
        return None, checkpoint_path
    
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f), checkpoint_path
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, checkpoint_path

def save_checkpoint(checkpoint_data, checkpoint_path):
    """Save the checkpoint file."""
    checkpoint_data['last_update'] = datetime.now().isoformat()
    
    # Atomic write
    temp_path = checkpoint_path + '.tmp'
    try:
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        os.rename(temp_path, checkpoint_path)
        print(f"✅ Checkpoint saved: {len(checkpoint_data['completed_combinations'])} combinations")
        return True
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def view_checkpoint(output_dir):
    """View the current checkpoint status."""
    checkpoint_data, _ = load_checkpoint(output_dir)
    
    if checkpoint_data is None:
        print("❌ No checkpoint file found.")
        return
    
    completed = checkpoint_data.get('completed_combinations', [])
    total_results = checkpoint_data.get('total_results', 0)
    last_update = checkpoint_data.get('last_update', 'Unknown')
    
    print(f"📊 Checkpoint Status:")
    print(f"   Completed combinations: {len(completed)}")
    print(f"   Total result entries: {total_results}")
    print(f"   Last update: {last_update}")
    
    # Break down by task
    tasks = {}
    models = {}
    methods = {}
    
    for combo in completed:
        parts = combo.split('_')
        if len(parts) >= 4:
            task = parts[0]
            model = parts[1]
            method = parts[2]
            
            tasks[task] = tasks.get(task, 0) + 1
            models[model] = models.get(model, 0) + 1
            methods[method] = methods.get(method, 0) + 1
    
    print(f"\n📋 Breakdown by Task:")
    for task, count in sorted(tasks.items()):
        print(f"   {task}: {count} combinations")
    
    print(f"\n🤖 Breakdown by Model:")
    for model, count in sorted(models.items()):
        print(f"   {model}: {count} combinations")
    
    print(f"\n🔧 Breakdown by Method:")
    for method, count in sorted(methods.items()):
        print(f"   {method}: {count} combinations")

def clear_task(output_dir, task_name):
    """Clear all combinations for a specific task."""
    checkpoint_data, checkpoint_path = load_checkpoint(output_dir)
    
    if checkpoint_data is None:
        print("❌ No checkpoint file found.")
        return
    
    completed = checkpoint_data.get('completed_combinations', [])
    original_count = len(completed)
    
    # Remove combinations that start with the task name
    new_completed = [combo for combo in completed if not combo.startswith(f"{task_name}_")]
    
    removed_count = original_count - len(new_completed)
    
    if removed_count == 0:
        print(f"ℹ️  No combinations found for task '{task_name}'")
        return
    
    checkpoint_data['completed_combinations'] = new_completed
    
    if save_checkpoint(checkpoint_data, checkpoint_path):
        print(f"✅ Removed {removed_count} combinations for task '{task_name}'")
        
        # Also remove from results CSV if exists
        results_path = os.path.join(output_dir, 'dimensionality_results.csv')
        if os.path.exists(results_path):
            try:
                df = pd.read_csv(results_path)
                original_rows = len(df)
                df = df[df['task'] != task_name]
                new_rows = len(df)
                
                if new_rows < original_rows:
                    df.to_csv(results_path, index=False)
                    print(f"✅ Removed {original_rows - new_rows} result rows for task '{task_name}'")
            except Exception as e:
                print(f"⚠️  Could not update results CSV: {e}")

def clear_model(output_dir, model_name):
    """Clear all combinations for a specific model."""
    checkpoint_data, checkpoint_path = load_checkpoint(output_dir)
    
    if checkpoint_data is None:
        print("❌ No checkpoint file found.")
        return
    
    completed = checkpoint_data.get('completed_combinations', [])
    original_count = len(completed)
    
    # Remove combinations that contain the model name
    new_completed = [combo for combo in completed if f"_{model_name}_" not in combo]
    
    removed_count = original_count - len(new_completed)
    
    if removed_count == 0:
        print(f"ℹ️  No combinations found for model '{model_name}'")
        return
    
    checkpoint_data['completed_combinations'] = new_completed
    
    if save_checkpoint(checkpoint_data, checkpoint_path):
        print(f"✅ Removed {removed_count} combinations for model '{model_name}'")

def clear_method(output_dir, method_name):
    """Clear all combinations for a specific method."""
    checkpoint_data, checkpoint_path = load_checkpoint(output_dir)
    
    if checkpoint_data is None:
        print("❌ No checkpoint file found.")
        return
    
    completed = checkpoint_data.get('completed_combinations', [])
    original_count = len(completed)
    
    # Remove combinations that contain the method name
    new_completed = [combo for combo in completed if f"_{method_name}_" not in combo]
    
    removed_count = original_count - len(new_completed)
    
    if removed_count == 0:
        print(f"ℹ️  No combinations found for method '{method_name}'")
        return
    
    checkpoint_data['completed_combinations'] = new_completed
    
    if save_checkpoint(checkpoint_data, checkpoint_path):
        print(f"✅ Removed {removed_count} combinations for method '{method_name}'")

def reset_checkpoint(output_dir):
    """Reset the entire checkpoint file."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_completed.json')
    
    if os.path.exists(checkpoint_path):
        backup_path = checkpoint_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(checkpoint_path, backup_path)
        print(f"✅ Checkpoint backed up to: {backup_path}")
    
    # Also backup results CSV
    results_path = os.path.join(output_dir, 'dimensionality_results.csv')
    if os.path.exists(results_path):
        backup_results_path = results_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(results_path, backup_results_path)
        print(f"✅ Results backed up to: {backup_results_path}")
    
    print("✅ Checkpoint and results reset. Analysis will start from beginning.")

def main():
    parser = argparse.ArgumentParser(description="Manage EHRSHOT dimensionality reduction checkpoint")
    parser.add_argument("--output_dir", default="../EHRSHOT_ASSETS/dimensionality_analysis",
                       help="Output directory containing checkpoint files")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View current checkpoint status')
    
    # Clear task command
    clear_task_parser = subparsers.add_parser('clear-task', help='Clear all combinations for a specific task')
    clear_task_parser.add_argument('task', help='Task name to clear (e.g., guo_los)')
    
    # Clear model command
    clear_model_parser = subparsers.add_parser('clear-model', help='Clear all combinations for a specific model')
    clear_model_parser.add_argument('model', help='Model name to clear (e.g., clmbr)')
    
    # Clear method command
    clear_method_parser = subparsers.add_parser('clear-method', help='Clear all combinations for a specific method')
    clear_method_parser.add_argument('method', help='Method name to clear (e.g., pca)')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset entire checkpoint (creates backup)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        return 1
    
    if args.command == 'view':
        view_checkpoint(args.output_dir)
    elif args.command == 'clear-task':
        clear_task(args.output_dir, args.task)
    elif args.command == 'clear-model':
        clear_model(args.output_dir, args.model)
    elif args.command == 'clear-method':
        clear_method(args.output_dir, args.method)
    elif args.command == 'reset':
        confirm = input("⚠️  This will reset the entire checkpoint. Continue? (y/N): ")
        if confirm.lower() == 'y':
            reset_checkpoint(args.output_dir)
        else:
            print("❌ Reset cancelled.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 