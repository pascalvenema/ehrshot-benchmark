#!/bin/bash

# Generate essential ClinicalBERT and embedding comparison plots
# This script creates the core visualization plots for the EHRSHOT paper

set -e

# Set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "generating essential clinicalbert and embedding comparison plots"

# Generate results plots
python ehrshot/plotting/results_plotting.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

echo "plots generated successfully"