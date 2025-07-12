#!/bin/bash

set -e

# Default paths - adjust these as needed
DEFAULT_DATABASE_PATH="../EHRSHOT_ASSETS/femr/extract"
DEFAULT_LABELS_PATH="../EHRSHOT_ASSETS/benchmark"
DEFAULT_FEATURES_PATH="../EHRSHOT_ASSETS/features"
DEFAULT_SPLITS_PATH="../EHRSHOT_ASSETS/splits/person_id_map.csv"
DEFAULT_OUTPUT_PATH="../EHRSHOT_ASSETS/dimensionality_analysis"

# allow overriding via environment variables
DATABASE_PATH=${EHRSHOT_DATABASE_PATH:-$DEFAULT_DATABASE_PATH}
LABELS_PATH=${EHRSHOT_LABELS_PATH:-$DEFAULT_LABELS_PATH}
FEATURES_PATH=${EHRSHOT_FEATURES_PATH:-$DEFAULT_FEATURES_PATH}
SPLITS_PATH=${EHRSHOT_SPLITS_PATH:-$DEFAULT_SPLITS_PATH}
OUTPUT_PATH=${EHRSHOT_OUTPUT_PATH:-$DEFAULT_OUTPUT_PATH}

echo "ehrshot dimensionality reduction analysis"
echo "database: $DATABASE_PATH"
echo "labels: $LABELS_PATH"
echo "features: $FEATURES_PATH"
echo "splits: $SPLITS_PATH"
echo "output: $OUTPUT_PATH"

# check if paths exist
if [ ! -d "$DATABASE_PATH" ]; then
    echo "error: database path does not exist: $DATABASE_PATH"
    exit 1
fi

if [ ! -d "$LABELS_PATH" ]; then
    echo "error: labels path does not exist: $LABELS_PATH"
    exit 1
fi

if [ ! -d "$FEATURES_PATH" ]; then
    echo "error: features path does not exist: $FEATURES_PATH"
    exit 1
fi

if [ ! -f "$SPLITS_PATH" ]; then
    echo "error: splits file does not exist: $SPLITS_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run the analysis
echo "running dimensionality reduction analysis"
python analysis/dimensionality_reduction.py \
    --path_to_database "$DATABASE_PATH" \
    --path_to_labels_dir "$LABELS_PATH" \
    --path_to_features_dir "$FEATURES_PATH" \
    --path_to_split_csv "$SPLITS_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --tasks guo_los guo_readmission guo_icu lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi \
    --models clmbr clinicalbert_type3_clinicalbert_pool \
    --dimensions 2 5 10 25 50 100 200 400 \
    --methods pca umap \
    --skip_baseline

echo "dimensionality reduction analysis completed"
echo "results saved to: $OUTPUT_PATH" 