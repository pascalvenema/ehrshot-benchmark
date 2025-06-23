#!/bin/bash

echo "📈 Generating CLMBR vs ClinicalBERT Comparison Plots"
echo "=================================================="

# Set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate CLMBR vs ClinicalBERT comparison plots
echo "📊 Generating CLMBR vs ClinicalBERT Type 3 comparison across all prediction heads..."

python3 ../plotting/heads_comparison.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ CLMBR vs ClinicalBERT comparison plots generated successfully!"
    echo "📂 Plots saved to: $OUTPUT_DIR"
else
    echo "❌ Comparison plot generation failed!"
    exit 1
fi 