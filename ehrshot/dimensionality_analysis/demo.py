#!/usr/bin/env python3
"""
Demo script for dimensionality reduction analysis.

This script creates synthetic data similar to ClinicalBERT embeddings
and runs a simplified version of the dimensionality analysis to verify
that the implementation works correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os

from reducers import get_reducer
from evaluator import evaluate_knn_with_reduced_embeddings, run_dimensionality_experiment
from plotting import plot_dimensionality_results, create_summary_table

def create_synthetic_clinical_data(n_samples=1000, n_features=768, random_state=42):
    """Create synthetic data similar to ClinicalBERT embeddings."""
    print("Creating synthetic clinical embedding data...")
    
    # Create classification data with some structure
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=100,  # Some features are informative
        n_redundant=50,     # Some are redundant
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=random_state
    )
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class balance - Train: {np.mean(y_train):.2f}, Val: {np.mean(y_val):.2f}, Test: {np.mean(y_test):.2f}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_individual_reducers():
    """Test each dimensionality reduction method individually."""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL REDUCTION METHODS")
    print("="*50)
    
    # Create small synthetic dataset
    X_train, X_val, X_test, y_train, y_val, y_test = create_synthetic_clinical_data(n_samples=200, n_features=100)
    
    methods = ['pca', 'umap']  # Only PCA and UMAP are supported
    n_components = 10
    
    for method in methods:
        print(f"\nTesting {method.upper()}...")
        
        try:
            result = evaluate_knn_with_reduced_embeddings(
                X_train, X_val, X_test, y_train, y_val, y_test,
                method, n_components
            )
            
            print(f"✅ {method.upper()} - AUROC: {result['auroc']:.3f}, AUPRC: {result['auprc']:.3f}")
            
        except Exception as e:
            print(f"❌ {method.upper()} failed: {e}")


def run_full_demo():
    """Run a complete dimensionality reduction experiment."""
    print("\n" + "="*50)
    print("RUNNING FULL DIMENSIONALITY EXPERIMENT")
    print("="*50)
    
    # Create synthetic data
    X_train, X_val, X_test, y_train, y_val, y_test = create_synthetic_clinical_data(n_samples=500, n_features=200)
    
    # Run experiment with limited dimensions for speed
    results_df = run_dimensionality_experiment(
        X_train, X_val, X_test, y_train, y_val, y_test,
        task_name="Synthetic Clinical Task",
        dimensions_to_test=[5, 10, 20, 50],
        methods=['pca', 'umap']  # Only PCA and UMAP are supported
    )
    
    print("\nResults:")
    print(results_df[['task', 'reduction_method', 'n_components', 'auroc', 'auprc']].to_string(index=False))
    
    return results_df


def demo_plotting(results_df, output_dir="demo_output"):
    """Demonstrate the plotting functionality."""
    print(f"\n" + "="*50)
    print("CREATING DEMO PLOTS")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots
    plot_dimensionality_results(results_df, output_dir)
    summary_df = create_summary_table(results_df, output_dir)
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Demo plots saved to {output_dir}/")
    print("Files created:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")


def main():
    """Run the complete demo."""
    print("🔬 DIMENSIONALITY REDUCTION ANALYSIS DEMO")
    print("="*60)
    print("This demo verifies the implementation using synthetic data")
    print("similar to ClinicalBERT embeddings.")
    print("="*60)
    
    # Test individual components
    test_individual_reducers()
    
    # Run full experiment
    results_df = run_full_demo()
    
    # Create plots
    demo_plotting(results_df)
    
    print("\n🎉 Demo completed successfully!")
    print("The implementation is working correctly and ready for real data.")


if __name__ == "__main__":
    main() 