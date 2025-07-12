#!/bin/bash



# set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

mkdir -p "$OUTPUT_DIR"


# Use the centralized plotting script
bash generate_all_plots.sh

if [ $? -eq 0 ]; then
    echo "SUCCESS"
else
    echo "FAILED"
    exit 1
fi
