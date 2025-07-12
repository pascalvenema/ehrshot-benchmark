#!/bin/bash

# Statistical Analysis of Dimensionality Reduction Results for EHRSHOT
# This script runs statistical tests on the dimensionality reduction results

set -e

# Default paths - modify these as needed
DIMENSIONALITY_CSV="${1:-EHRSHOT_ASSETS/dimensionality_analysis/dimensionality_results.csv}"
RESULTS_DIR="${2:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${3:-EHRSHOT_ASSETS/dimensionality_statistical_analysis}"

echo "ehrshot dimensionality statistical analysis"
echo "dimensionality csv: $DIMENSIONALITY_CSV"
echo "results directory: $RESULTS_DIR"
echo "output directory: $OUTPUT_DIR"

# Check if files exist
if [ ! -f "$DIMENSIONALITY_CSV" ]; then
    echo "error: dimensionality results csv not found: $DIMENSIONALITY_CSV"
    echo "please run the dimensionality reduction analysis first"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "error: results directory not found: $RESULTS_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run statistical analysis
echo "running statistical analysis"
python ehrshot/analysis/dimensionality_statistical_analysis.py \
    --dimensionality_csv "$DIMENSIONALITY_CSV" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

echo "statistical analysis completed" 