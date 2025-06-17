#!/usr/bin/env python3
"""
Create comprehensive visualizations for dimensionality reduction analysis.
This script generates publication-ready plots showing the relationship between
dimensionality, compression, and k-NN performance on clinical prediction tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def load_results(results_path: str) -> pd.DataFrame:
    """Load and prepare the dimensionality reduction results."""
    df = pd.read_csv(results_path)
    
    # Add compression ratio
    df['compression_ratio'] = 768 / df['n_components']
    df['compression_ratio'] = df['compression_ratio'].fillna(1.0)  # Original has 1x compression
    
    # Add size reduction percentage
    df['size_reduction_pct'] = (1 - 1/df['compression_ratio']) * 100
    
    # Clean method names
    df['method_clean'] = df['method'].replace({
        'original': 'Original 768D',
        'pca': 'PCA',
        'truncated_svd': 'Truncated SVD'
    })
    
    return df


def plot_performance_vs_dimensionality(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance (AUROC/AUPRC) vs dimensionality."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for optimal k=15 results
    df_k15 = df[df['k_neighbors'] == 15].copy()
    
    # AUROC plot
    for method in df_k15['method_clean'].unique():
        method_data = df_k15[df_k15['method_clean'] == method]
        ax1.plot(method_data['n_components'], method_data['auroc'], 
                'o-', label=method, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('AUROC')
    ax1.set_title('AUROC vs Dimensionality (k=15)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # AUPRC plot
    for method in df_k15['method_clean'].unique():
        method_data = df_k15[df_k15['method_clean'] == method]
        ax2.plot(method_data['n_components'], method_data['auprc'], 
                'o-', label=method, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('AUPRC')
    ax2.set_title('AUPRC vs Dimensionality (k=15)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_dimensionality.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_compression_tradeoff(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot compression ratio vs performance trade-off."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for optimal k=15 results
    df_k15 = df[df['k_neighbors'] == 15].copy()
    
    # AUROC vs Compression
    for method in df_k15['method_clean'].unique():
        method_data = df_k15[df_k15['method_clean'] == method]
        ax1.scatter(method_data['compression_ratio'], method_data['auroc'], 
                   label=method, s=80, alpha=0.7)
        
        # Add annotations for key points
        for _, row in method_data.iterrows():
            if row['n_components'] in [16, 32, 64, 768]:
                ax1.annotate(f"{int(row['n_components'])}D", 
                           (row['compression_ratio'], row['auroc']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Compression Ratio (768D / n_components)')
    ax1.set_ylabel('AUROC')
    ax1.set_title('Performance vs Compression Trade-off (AUROC)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # AUPRC vs Compression
    for method in df_k15['method_clean'].unique():
        method_data = df_k15[df_k15['method_clean'] == method]
        ax2.scatter(method_data['compression_ratio'], method_data['auprc'], 
                   label=method, s=80, alpha=0.7)
        
        # Add annotations for key points
        for _, row in method_data.iterrows():
            if row['n_components'] in [16, 32, 64, 768]:
                ax2.annotate(f"{int(row['n_components'])}D", 
                           (row['compression_ratio'], row['auprc']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Compression Ratio (768D / n_components)')
    ax2.set_ylabel('AUPRC')
    ax2.set_title('Performance vs Compression Trade-off (AUPRC)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'compression_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_k_optimization(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot k-NN parameter optimization across dimensions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Focus on PCA results for clarity
    df_pca = df[df['method'] == 'pca'].copy()
    
    # AUROC heatmap
    pivot_auroc = df_pca.pivot(index='k_neighbors', columns='n_components', values='auroc')
    sns.heatmap(pivot_auroc, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('AUROC by k-neighbors and Dimensionality (PCA)')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('k-neighbors')
    
    # AUPRC heatmap
    pivot_auprc = df_pca.pivot(index='k_neighbors', columns='n_components', values='auprc')
    sns.heatmap(pivot_auprc, annot=True, fmt='.3f', cmap='viridis', ax=ax2)
    ax2.set_title('AUPRC by k-neighbors and Dimensionality (PCA)')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('k-neighbors')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'k_optimization_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_method_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare PCA vs Truncated SVD performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for k=15 and exclude original
    df_methods = df[(df['k_neighbors'] == 15) & (df['method'] != 'original')].copy()
    
    # AUROC comparison
    df_pivot_auroc = df_methods.pivot(index='n_components', columns='method_clean', values='auroc')
    df_pivot_auroc.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('AUROC: PCA vs Truncated SVD')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('AUROC')
    ax1.legend(title='Method')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # AUPRC comparison
    df_pivot_auprc = df_methods.pivot(index='n_components', columns='method_clean', values='auprc')
    df_pivot_auprc.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('AUPRC: PCA vs Truncated SVD')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('AUPRC')
    ax2.legend(title='Method')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_degradation(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot performance degradation from original 768D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for k=15 results
    df_k15 = df[df['k_neighbors'] == 15].copy()
    
    # Get original performance
    original_auroc = df_k15[df_k15['method'] == 'original']['auroc'].iloc[0]
    original_auprc = df_k15[df_k15['method'] == 'original']['auprc'].iloc[0]
    
    # Calculate performance drop
    df_reduced = df_k15[df_k15['method'] != 'original'].copy()
    df_reduced['auroc_drop_pct'] = ((original_auroc - df_reduced['auroc']) / original_auroc) * 100
    df_reduced['auprc_drop_pct'] = ((original_auprc - df_reduced['auprc']) / original_auprc) * 100
    
    # AUROC degradation
    for method in df_reduced['method_clean'].unique():
        method_data = df_reduced[df_reduced['method_clean'] == method]
        ax1.plot(method_data['n_components'], method_data['auroc_drop_pct'], 
                'o-', label=method, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('AUROC Performance Drop (%)')
    ax1.set_title('AUROC Performance Degradation from 768D Original')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    
    # AUPRC degradation
    for method in df_reduced['method_clean'].unique():
        method_data = df_reduced[df_reduced['method_clean'] == method]
        ax2.plot(method_data['n_components'], method_data['auprc_drop_pct'], 
                'o-', label=method, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('AUPRC Performance Drop (%)')
    ax2.set_title('AUPRC Performance Degradation from 768D Original')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_degradation.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_curse_of_dimensionality_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualization specifically addressing curse of dimensionality."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter for k=15 PCA results (best performing)
    df_pca_k15 = df[(df['method'] == 'pca') & (df['k_neighbors'] == 15)].copy()
    df_original = df[(df['method'] == 'original') & (df['k_neighbors'] == 15)].copy()
    
    # Combine for analysis
    df_analysis = pd.concat([df_original, df_pca_k15])
    
    # 1. Performance vs Dimensionality (log scale)
    ax1.semilogx(df_analysis['n_components'], df_analysis['auroc'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Dimensions (log scale)')
    ax1.set_ylabel('AUROC')
    ax1.set_title('k-NN Performance vs Dimensionality\n(Evidence Against Curse of Dimensionality)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.94, 1.0)
    
    # Add annotations
    for _, row in df_analysis.iterrows():
        if row['n_components'] in [16, 32, 768]:
            ax1.annotate(f"{int(row['n_components'])}D\n{row['auroc']:.3f}", 
                        (row['n_components'], row['auroc']),
                        xytext=(0, 10), textcoords='offset points', 
                        ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. Relative Performance (% of original)
    original_auroc = df_original['auroc'].iloc[0]
    df_pca_k15['relative_auroc'] = (df_pca_k15['auroc'] / original_auroc) * 100
    
    ax2.semilogx(df_pca_k15['n_components'], df_pca_k15['relative_auroc'], 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.set_xlabel('Number of Dimensions (log scale)')
    ax2.set_ylabel('Performance (% of 768D original)')
    ax2.set_title('Relative Performance Retention')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(95, 101)
    
    # 3. Compression vs Performance
    ax3.scatter(df_pca_k15['compression_ratio'], df_pca_k15['auroc'], s=100, alpha=0.7, c='purple')
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('AUROC')
    ax3.set_title('Compression vs Performance Trade-off')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for _, row in df_pca_k15.iterrows():
        ax3.annotate(f"{int(row['n_components'])}D", 
                    (row['compression_ratio'], row['auroc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 4. Performance drop vs dimensions
    df_pca_k15['auroc_drop_pct'] = ((original_auroc - df_pca_k15['auroc']) / original_auroc) * 100
    
    ax4.semilogx(df_pca_k15['n_components'], df_pca_k15['auroc_drop_pct'], 'go-', linewidth=2, markersize=8)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='1% drop')
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% drop')
    ax4.set_xlabel('Number of Dimensions (log scale)')
    ax4.set_ylabel('Performance Drop (%)')
    ax4.set_title('Performance Degradation from Original')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'curse_of_dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Create a summary table of key results."""
    # Filter for k=15 results
    df_k15 = df[df['k_neighbors'] == 15].copy()
    
    # Get original performance
    original_row = df_k15[df_k15['method'] == 'original'].iloc[0]
    
    # Create summary for reduced dimensions
    summary_data = []
    for method in ['pca', 'truncated_svd']:
        method_data = df_k15[df_k15['method'] == method]
        for _, row in method_data.iterrows():
            summary_data.append({
                'Method': method.upper(),
                'Dimensions': int(row['n_components']),
                'AUROC': f"{row['auroc']:.4f}",
                'AUPRC': f"{row['auprc']:.4f}",
                'AUROC Drop (%)': f"{((original_row['auroc'] - row['auroc']) / original_row['auroc'] * 100):.2f}",
                'AUPRC Drop (%)': f"{((original_row['auprc'] - row['auprc']) / original_row['auprc'] * 100):.2f}",
                'Compression': f"{row['compression_ratio']:.1f}x",
                'Size Reduction (%)': f"{row['size_reduction_pct']:.1f}"
            })
    
    # Add original
    summary_data.insert(0, {
        'Method': 'ORIGINAL',
        'Dimensions': 768,
        'AUROC': f"{original_row['auroc']:.4f}",
        'AUPRC': f"{original_row['auprc']:.4f}",
        'AUROC Drop (%)': "0.00",
        'AUPRC Drop (%)': "0.00",
        'Compression': "1.0x",
        'Size Reduction (%)': "0.0"
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'dimensionality_summary_table.csv', index=False)
    
    print("📋 Summary Table:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Create dimensionality reduction visualizations')
    parser.add_argument('--results_path', type=str, 
                       default='../EHRSHOT_ASSETS/knn_analysis_final/knn_dimensionality_results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='../EHRSHOT_ASSETS/dimensionality_visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Creating dimensionality reduction visualizations...")
    
    # Load results
    df = load_results(args.results_path)
    print(f"📊 Loaded {len(df)} result records")
    
    # Create visualizations
    print("📈 Creating performance vs dimensionality plots...")
    plot_performance_vs_dimensionality(df, output_dir)
    
    print("📉 Creating compression trade-off analysis...")
    plot_compression_tradeoff(df, output_dir)
    
    print("🔧 Creating k-parameter optimization heatmaps...")
    plot_k_optimization(df, output_dir)
    
    print("⚖️ Creating method comparison plots...")
    plot_method_comparison(df, output_dir)
    
    print("📊 Creating performance degradation analysis...")
    plot_performance_degradation(df, output_dir)
    
    print("🔍 Creating curse of dimensionality analysis...")
    create_curse_of_dimensionality_analysis(df, output_dir)
    
    print("📋 Creating summary table...")
    create_summary_table(df, output_dir)
    
    print(f"✅ All visualizations saved to: {output_dir}")
    print("\n🎯 Key findings visualized:")
    print("  • Performance vs dimensionality trade-offs")
    print("  • Evidence against curse of dimensionality")
    print("  • PCA vs Truncated SVD comparison")
    print("  • Optimal k-parameter analysis")
    print("  • Compression efficiency analysis")


if __name__ == "__main__":
    main() 