#!/bin/bash
#SBATCH --job-name=dim_reduction_all
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_dimensionality_%j.out
#SBATCH --error=slurm_dimensionality_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pascal.venema@student.uva.nl

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load environment
source ~/.bashrc
conda activate EHRSHOT_ENV

# Change to the correct directory
cd /home/pvenema/ehrshot-benchmark/ehrshot

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Available memory: $(free -h)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"

# Set threading for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the dimensionality reduction analysis
echo "Starting dimensionality reduction analysis..."
bash bash_scripts/10_dimensionality_reduction.sh

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Dimensionality reduction analysis completed successfully!"
    echo "Generated files:"
    ls -la ../EHRSHOT_ASSETS/dimensionality_analysis/
else
    echo "Dimensionality reduction analysis failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)" 