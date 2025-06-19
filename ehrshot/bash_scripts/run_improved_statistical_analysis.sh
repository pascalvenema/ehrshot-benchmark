#!/bin/bash

# Improved K-Shot Aware Statistical Significance Analysis for EHRSHOT Benchmark
# This script runs comprehensive statistical tests across all k-shot conditions

set -e

# Default paths - modify these as needed
RESULTS_DIR="${1:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${2:-EHRSHOT_ASSETS/improved_statistical_analysis}"
ALPHA="${3:-0.05}"

echo "🔬 EHRSHOT Improved K-Shot Aware Statistical Analysis"
echo "====================================================="
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Significance level: $ALPHA"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run improved statistical significance analysis
echo "🧪 Running improved k-shot aware statistical analysis..."
python ehrshot/improved_statistical_analysis.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR" \
    --alpha "$ALPHA" \
    --scores auroc auprc

echo ""
echo "✅ Analysis complete!"
echo "📊 Check the output directory for detailed results:"
echo "   - improved_statistical_summary_auroc.txt"
echo "   - improved_statistical_summary_auprc.txt"
echo ""
echo "🔍 Key improvements in this analysis:"
echo "   1. RQ1.1: Analyzes ClinicalBERT types across ALL k-values (not just k=-1)"
echo "   2. RQ1.2: Compares CLMBR vs ClinicalBERT across ALL k-values with trend analysis"
echo "   3. RQ2.1: Fair comparison using BEST non-linear head per task (not average)"
echo "   4. Uses appropriate repeated measures tests for k-shot dependencies" 