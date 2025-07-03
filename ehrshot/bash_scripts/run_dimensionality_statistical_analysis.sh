#!/bin/bash

# Dimensionality Reduction Statistical Analysis
# This script runs statistical tests on the dimensionality reduction results

set -e

# Default paths
DIMENSIONALITY_CSV="${1:-EHRSHOT_ASSETS/dimensionality_analysis/dimensionality_results.csv}"
RESULTS_DIR="${2:-EHRSHOT_ASSETS}"
OUTPUT_DIR="${3:-EHRSHOT_ASSETS/dimensionality_analysis}"

echo "🔬 Running Dimensionality Reduction Statistical Analysis"
echo "=================================================="
echo "Dimensionality CSV: $DIMENSIONALITY_CSV"
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if dimensionality results exist
if [ ! -f "$DIMENSIONALITY_CSV" ]; then
    echo "❌ Error: Dimensionality results not found at $DIMENSIONALITY_CSV"
    echo "Please run the dimensionality reduction analysis first."
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found at $RESULTS_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the statistical analysis
echo "🚀 Running statistical analysis..."
python3 ehrshot/analysis/dimensionality_statistical_analysis.py \
    --dimensionality_csv "$DIMENSIONALITY_CSV" \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Analysis completed! Check the output in $OUTPUT_DIR"
echo "   - dimensionality_statistical_summary.txt: Summary report" 