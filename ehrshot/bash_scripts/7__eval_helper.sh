#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h100
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

module load 2024
module load Miniconda3/24.7.1-0

# Activate your environment
source activate EHRSHOT_ENV
# Check whether the GPU is available
srun python -uc "import torch; print('GPU available?', torch.cuda.is_available())"

python3 ../evaluation.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --labeling_function $6 \
    --shot_strat $7 \
    --num_threads $8