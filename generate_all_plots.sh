#!/bin/bash


# set default paths - can be overridden by environment variables
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directories
mkdir -p "$OUTPUT_DIR"

echo "generating essential plots only"

# 1. Results Plots (ClinicalBERT comparisons and embedding comparisons)
echo ""
echo "1. generating clinicalbert and embedding comparison plots"

python3 ehrshot/plotting/results_plotting.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "   clinicalbert and embedding plots completed"
else
    echo "   clinicalbert and embedding plots failed"
    exit 1
fi

# 2. Model Comparison Plots (CLMBR vs ClinicalBERT across heads)
echo ""
echo "2. generating clmbr vs clinicalbert comparison plots"

python3 ehrshot/plotting/heads_comparison.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "   model comparison plots completed"
else
    echo "   model comparison plots failed"
    exit 1
fi

# 3. Dimensionality Analysis Plots (if results exist)
echo ""
echo "3. generating dimensionality plots"

if [ -f "$RESULTS_DIR/../dimensionality_analysis/dimensionality_results.csv" ]; then
    echo "   dimensionality results found, generating plots"
    
    python3 ehrshot/plotting/dimensionality_plotting.py \
        --results_csv "$RESULTS_DIR/../dimensionality_analysis/dimensionality_results.csv" \
        --output_dir "$OUTPUT_DIR" \
        --baseline_dir "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        echo "   dimensionality plots completed"
    else
        echo "   dimensionality plots failed"
        exit 1
    fi
else
    echo "   dimensionality results not found"
    echo "   run dimensionality analysis first: bash ehrshot/bash_scripts/10_dimensionality_reduction.sh"
fi

echo ""
echo "essential plots generated successfully"
echo "output location: $OUTPUT_DIR" 