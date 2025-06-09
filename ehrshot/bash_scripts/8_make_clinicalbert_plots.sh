#!/bin/bash
#SBATCH --job-name=8_make_clinicalbert_figures
#SBATCH --output=logs/8_make_clinicalbert_figures_%A.out
#SBATCH --error=logs/8_make_clinicalbert_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Create output directory
mkdir -p ../../EHRSHOT_ASSETS/figures

# Generate ClinicalBERT visualizations with all available ClinicalBERT models
python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/results \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures \
    --model_heads "[]" \
    --shot_strat all

echo "ClinicalBERT plotting completed!"
echo "Check ../../EHRSHOT_ASSETS/figures/clinicalbert/ for ClinicalBERT-specific plots" 