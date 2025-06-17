#!/bin/bash
#SBATCH --job-name=dimensionality_reduction
#SBATCH --output=logs/dimensionality_reduction_%A.out
#SBATCH --error=logs/dimensionality_reduction_%A.err
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10

echo "Starting Dimensionality Reduction Analysis for EHRSHOT Benchmark"
echo "=================================================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up paths
DATABASE_PATH="../../EHRSHOT_ASSETS/femr/extract"
LABELS_DIR="../../EHRSHOT_ASSETS/benchmark"
FEATURES_DIR="../../EHRSHOT_ASSETS/features"
SPLIT_CSV="../../EHRSHOT_ASSETS/splits/person_id_map.csv"
OUTPUT_DIR="../../EHRSHOT_ASSETS/dimensionality_reduction"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Configuration
MODEL_CONFIG="clinicalbert_type3_clinicalbert_pool"
PREDICTION_HEADS="knn,lr_lbfgs,rf"
REDUCTION_METHODS="pca,truncated_svd"
DIMENSIONS="32,64,128,256"
NUM_THREADS=10

# Tasks to test (focusing on a subset for initial analysis)
TASKS="guo_icu,guo_los,guo_readmission,lab_anemia,lab_hypoglycemia,new_hypertension"

echo "Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Prediction heads: $PREDICTION_HEADS"
echo "  Reduction methods: $REDUCTION_METHODS"
echo "  Dimensions: $DIMENSIONS"
echo "  Tasks: $TASKS"
echo "  Threads: $NUM_THREADS"
echo ""

# Run the dimensionality reduction analysis
echo "Running dimensionality reduction analysis..."
python3 ../10_dimensionality_reduction.py \
    --path_to_database "$DATABASE_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_features_dir "$FEATURES_DIR" \
    --path_to_split_csv "$SPLIT_CSV" \
    --path_to_output_dir "$OUTPUT_DIR" \
    --labeling_functions "$TASKS" \
    --reduction_methods "$REDUCTION_METHODS" \
    --dimensions "$DIMENSIONS" \
    --model_config "$MODEL_CONFIG" \
    --prediction_heads "$PREDICTION_HEADS" \
    --num_threads "$NUM_THREADS" \
    --create_visualizations

exit_code=$?

echo ""
echo "=================================================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Dimensionality Reduction Analysis completed successfully!"
    echo ""
    echo "📊 Results and visualizations saved to:"
    echo "   $OUTPUT_DIR"
    echo ""
    echo "📁 Generated outputs:"
    echo "   - Analysis reports: $OUTPUT_DIR/reports/"
    echo "   - Reduced features: $OUTPUT_DIR/reduced_features/"
    echo "   - Visualizations: $OUTPUT_DIR/visualizations/"
    echo "   - Main results: $OUTPUT_DIR/dimensionality_reduction_results.csv"
    echo ""
    echo "🔍 Key research insights:"
    echo "   - Check k-NN performance improvements with reduced dimensions"
    echo "   - Compare PCA vs Truncated SVD reduction methods"
    echo "   - Analyze optimal dimensionality for different tasks"
else
    echo "❌ Dimensionality Reduction Analysis failed with exit code: $exit_code"
    echo "Check the log files for detailed error information."
fi

echo ""
echo "Job completed at: $(date)"
echo "==================================================================" 