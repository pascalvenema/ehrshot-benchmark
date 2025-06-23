#!/bin/bash

echo "🎯 EHRSHOT - Essential Plots Only"
echo "=================================="
echo ""

# Set default paths
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

echo "📁 Creating output directory..."
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🎨 Generating essential plots..."
echo "==============================="

# Use the centralized plotting script
bash generate_all_plots.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS!"
    echo "=========="
    echo "✅ All essential plots generated successfully!"
    echo "📂 Check your plots in: $OUTPUT_DIR"
else
    echo ""
    echo "❌ FAILED!"
    echo "========="
    echo "Plot generation encountered errors."
    exit 1
fi
