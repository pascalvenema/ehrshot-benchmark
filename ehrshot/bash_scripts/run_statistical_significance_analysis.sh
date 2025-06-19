#!/bin/bash

# Statistical Significance Analysis for EHRSHOT Benchmark
# This script runs comprehensive statistical tests to answer research questions

set -e

# Default paths - modify these as needed
RESULTS_DIR="${1:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${2:-EHRSHOT_ASSETS/statistical_analysis}"
ALPHA="${3:-0.05}"

echo "🧪 EHRSHOT Statistical Significance Analysis"
echo "============================================"
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Significance level: $ALPHA"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run statistical significance analysis
echo "🔬 Running statistical significance analysis..."
python ehrshot/statistical_significance_analysis.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR" \
    --alpha "$ALPHA" \
    --scores auroc auprc

echo ""
echo "✅ Statistical analysis complete!"
echo "📊 Check results in: $OUTPUT_DIR"
echo ""
echo "Key findings will be printed above. For detailed results, see:"
echo "  - Text files with statistical summaries"
echo "  - Detailed console output above" 