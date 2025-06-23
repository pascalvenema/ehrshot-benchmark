#!/bin/bash

echo "🎨 Generating Essential Results Plots"
echo "====================================="

# Set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate essential plots only
echo "📊 Generating essential ClinicalBERT and embedding comparison plots..."

python3 ../plotting/results_plotting.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✅ Essential results plots generated successfully!"
    echo "📂 Plots saved to: $OUTPUT_DIR"
else
    echo "❌ Plot generation failed!"
    exit 1
fi