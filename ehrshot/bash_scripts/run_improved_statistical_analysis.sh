#!/bin/bash

# Improved K-Shot Aware Statistical Significance Analysis for EHRSHOT Benchmark
# This script runs comprehensive statistical tests across all k-shot conditions

set -e

# Default paths - modify these as needed
RESULTS_DIR="${1:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${2:-EHRSHOT_ASSETS/improved_statistical_analysis}"
ALPHA="${3:-0.05}"

echo "ehrshot improved k-shot aware statistical analysis"
echo "results directory: $RESULTS_DIR"
echo "output directory: $OUTPUT_DIR"
echo "significance level: $ALPHA"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run improved statistical significance analysis
echo "running improved k-shot aware statistical analysis"
python ehrshot/analysis/statistical_analysis.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR" \
    --alpha "$ALPHA" \
    --scores auroc auprc

echo "analysis complete"
echo "check the output directory for detailed results" 