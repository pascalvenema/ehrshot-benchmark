#!/bin/bash
#SBATCH --job-name=dim_reduction_stable
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=150G
#SBATCH --time=16:00:00
#SBATCH --output=slurm_dimensionality_stable_%j.out
#SBATCH --error=slurm_dimensionality_stable_%j.err
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

# Conservative threading configuration for maximum stability
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# Conservative scikit-learn threading
export SKLEARN_N_JOBS=32

# Optimize memory allocation and CPU affinity
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_SCHEDULE=static

# Prevent threading conflicts
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE

# Memory optimizations
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=0

# Print threading configuration
echo "Stable Threading Configuration:"
echo "  CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "  Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS (conservative)"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS (conservative)"
echo "  SKLEARN_N_JOBS: $SKLEARN_N_JOBS (conservative)"
echo "  Node: $SLURM_NODELIST"

# Monitor resource usage
echo "Initial resource usage:"
free -h

# Run the dimensionality reduction analysis with stable configuration (skipping baseline)
echo "Starting STABLE dimensionality reduction analysis (skipping baseline)..."
time bash bash_scripts/10_dimensionality_reduction.sh

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo "Dimensionality reduction analysis completed successfully!"
    echo "Final resource usage:"
    free -h
    echo "Generated files:"
    ls -la ../EHRSHOT_ASSETS/dimensionality_analysis/
else
    echo "Dimensionality reduction analysis failed with exit code $?"
    exit 1
fi

echo "End Time: $(date)"
echo "Total execution time logged above." 