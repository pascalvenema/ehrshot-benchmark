#!/bin/bash

echo "�� EHRSHOT - Generate Essential Plots Only"
echo "==========================================="
echo ""

# Set default paths - can be overridden by environment variables
RESULTS_DIR="${RESULTS_DIR:-EHRSHOT_ASSETS/results}"
OUTPUT_DIR="${OUTPUT_DIR:-EHRSHOT_ASSETS/figures}"

# Create output directories
echo "📁 Creating output directories..."
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🎯 Generating essential plots only..."
echo "====================================="

# 1. Results Plots (ClinicalBERT comparisons and embedding comparisons)
echo ""
echo "1️⃣  Generating ClinicalBERT and embedding comparison plots..."
echo "   📊 This includes:"
echo "      • ClinicalBERT type comparisons"
echo "      • ClinicalBERT pooling comparisons"
echo "      • Detailed task embedding comparisons"
echo "      • Embedding comparison (LR only)"

python3 ehrshot/plotting/results_plotting.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "   ✅ ClinicalBERT and embedding plots completed"
else
    echo "   ❌ ClinicalBERT and embedding plots failed"
    exit 1
fi

# 2. Model Comparison Plots (CLMBR vs ClinicalBERT across heads)
echo ""
echo "2️⃣  Generating CLMBR vs ClinicalBERT comparison plots..."
echo "   📈 CLMBR vs ClinicalBERT Type 3 comparison across all prediction heads"

python3 ehrshot/plotting/heads_comparison.py \
    --path_to_results_dir "$RESULTS_DIR" \
    --path_to_output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "   ✅ Model comparison plots completed"
else
    echo "   ❌ Model comparison plots failed"
    exit 1
fi

# 3. Dimensionality Analysis Plots (if results exist)
echo ""
echo "3️⃣  Generating dimensionality plots..."

if [ -f "$RESULTS_DIR/../dimensionality_analysis/dimensionality_results.csv" ]; then
    echo "   📊 Dimensionality results found, generating plots..."
    
    python3 ehrshot/plotting/dimensionality_plotting.py \
        --results_csv "$RESULTS_DIR/../dimensionality_analysis/dimensionality_results.csv" \
        --output_dir "$OUTPUT_DIR" \
        --baseline_dir "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Dimensionality plots completed"
    else
        echo "   ❌ Dimensionality plots failed"
        exit 1
    fi
else
    echo "   ⚠️  Dimensionality results not found at $RESULTS_DIR/../dimensionality_analysis/"
    echo "   📝 Note: Run dimensionality analysis first:"
    echo "      bash ehrshot/bash_scripts/10_dimensionality_reduction.sh"
fi

echo ""
echo "🎉 ESSENTIAL PLOTS GENERATED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "📂 Output location:"
echo "   📊 All plots: $OUTPUT_DIR"
echo ""
echo "📋 Generated plot files:"
echo "   • clean_auprc_vs_dimensions_separated.png"
echo "   • clean_auroc_vs_dimensions_separated.png"
echo "   • clinicalbert_type_comparison_auroc.png"
echo "   • clinicalbert_type_comparison_auprc.png"
echo "   • clinicalbert_type3_pooling_mean_all_tasks_auroc.png"
echo "   • clinicalbert_type3_pooling_mean_all_tasks_auprc.png"
echo "   • clmbr_vs_clinicalbert_type3_by_model_auroc.png"
echo "   • clmbr_vs_clinicalbert_type3_by_model_auprc.png"
echo "   • detailed_task_embedding_comparison_auroc.png"
echo "   • detailed_task_embedding_comparison_auprc.png"
echo "   • embedding_comparison_lr_only_auroc.png"
echo "   • embedding_comparison_lr_only_auprc.png"
echo ""
echo "🚀 Your essential visualizations are ready!" 