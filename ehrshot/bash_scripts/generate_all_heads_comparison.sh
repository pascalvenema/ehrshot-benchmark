#!/bin/bash

# Generate comparison plots for CLMBR vs ClinicalBERT Type 3 (clinicalbert_pool) across all prediction heads
# This script creates a 2×3 grid showing CLMBR (top row) vs ClinicalBERT Type 3 (bottom row) across task groups

# Default paths (can be overridden by command line arguments)
results_dir="../../EHRSHOT_ASSETS/results"
output_dir="../../EHRSHOT_ASSETS/figures"

# Create output directory
mkdir -p "$output_dir"

# Run the Python script
echo "🎨 Generating CLMBR vs ClinicalBERT Type 3 comparison plots with 2×3 layout..."
python ../generate_all_heads_comparison.py \
    --path_to_results_dir "$results_dir" \
    --path_to_output_dir "$output_dir"

echo "✅ All plots generated and saved to: $output_dir"
echo "📊 Check the following files:"
echo "   - clmbr_vs_clinicalbert_type3_by_model_auroc.png"
echo "   - clmbr_vs_clinicalbert_type3_by_model_auprc.png" 