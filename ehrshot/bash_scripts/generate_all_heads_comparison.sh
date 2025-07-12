#!/bin/bash
echo "generating clmbr vs clinicalbert comparison plots"

# Set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate plots for all prediction heads
echo "generating clmbr vs clinicalbert type 3 comparison across all prediction heads"
python ehrshot/plotting/heads_comparison.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

echo "plots generated successfully" 