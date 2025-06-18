#!/bin/bash

# Dimensionality Reduction Analysis for EHRSHOT
# This script runs the dimensionality reduction analysis on both CLMBR and ClinicalBERT type3 embeddings

set -e

# Default paths - adjust these as needed
DEFAULT_DATABASE_PATH="../EHRSHOT_ASSETS/femr/extract"
DEFAULT_LABELS_PATH="../EHRSHOT_ASSETS/benchmark"
DEFAULT_FEATURES_PATH="../EHRSHOT_ASSETS/features"
DEFAULT_SPLITS_PATH="../EHRSHOT_ASSETS/splits/person_id_map.csv"
DEFAULT_OUTPUT_PATH="../EHRSHOT_ASSETS/dimensionality_analysis"

# Allow overriding via environment variables
DATABASE_PATH=${EHRSHOT_DATABASE_PATH:-$DEFAULT_DATABASE_PATH}
LABELS_PATH=${EHRSHOT_LABELS_PATH:-$DEFAULT_LABELS_PATH}
FEATURES_PATH=${EHRSHOT_FEATURES_PATH:-$DEFAULT_FEATURES_PATH}
SPLITS_PATH=${EHRSHOT_SPLITS_PATH:-$DEFAULT_SPLITS_PATH}
OUTPUT_PATH=${EHRSHOT_OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}

echo "🧬 Starting EHRSHOT Dimensionality Reduction Analysis"
echo "=================================================="
echo "Database: $DATABASE_PATH"
echo "Labels:   $LABELS_PATH"
echo "Features: $FEATURES_PATH"
echo "Splits:   $SPLITS_PATH"
echo "Output:   $OUTPUT_PATH"
echo ""

# Check if paths exist
if [ ! -d "$DATABASE_PATH" ]; then
    echo "❌ Error: Database path does not exist: $DATABASE_PATH"
    exit 1
fi

if [ ! -d "$LABELS_PATH" ]; then
    echo "❌ Error: Labels path does not exist: $LABELS_PATH"
    exit 1
fi

if [ ! -d "$FEATURES_PATH" ]; then
    echo "❌ Error: Features path does not exist: $FEATURES_PATH"
    exit 1
fi

if [ ! -f "$SPLITS_PATH" ]; then
    echo "❌ Error: Splits file does not exist: $SPLITS_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run the analysis
echo "🚀 Running dimensionality reduction analysis..."
python 10_dimensionality_reduction.py \
    --path_to_database "$DATABASE_PATH" \
    --path_to_labels_dir "$LABELS_PATH" \
    --path_to_features_dir "$FEATURES_PATH" \
    --path_to_split_csv "$SPLITS_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --tasks guo_los guo_readmission guo_icu lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi \
    --models clmbr clinicalbert_type3_clinicalbert_pool \
    --dimensions 2 5 10 25 50 100 200 400 \
    --methods pca umap tsne

echo ""
echo "✅ Dimensionality reduction analysis completed!"
echo "📊 Results saved to: $OUTPUT_PATH"
echo ""
echo "Generated files:"
echo "  - dimensionality_results.csv (raw results)"
echo "  - dimensionality_summary.csv (summary table)"
echo "  - dimensionality_analysis_comparison.png (separate model plots)"
echo "  - dimensionality_analysis_combined.png (combined comparison)"
echo "  - best_dimensions_heatmap.png (optimal dimensions heatmap)"
echo "  - dimensionality_analysis.log (execution log)" 