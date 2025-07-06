#!/bin/bash
#SBATCH --job-name=dim_reduction_maximum
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=280G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_dimensionality_max_%j.out
#SBATCH --error=slurm_dimensionality_max_%j.err
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

# Safe threading configuration - prevent OpenBLAS overload
# Many numerical libraries have limits around 64-128 threads
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

# Use more threads for scikit-learn since it handles parallelism better
export SKLEARN_N_JOBS=$SLURM_CPUS_PER_TASK

# Optimize memory allocation and CPU affinity
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_SCHEDULE=static

# Additional performance optimizations
export MALLOC_ARENA_MAX=4
export PYTHONHASHSEED=0

# Prevent threading overload that causes segfaults
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE

# Print threading configuration
echo "Safe Maximum Performance Threading Configuration:"
echo "  CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "  Memory allocated: ${SLURM_MEM_PER_NODE}MB"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS (capped for stability)"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS (capped for stability)"
echo "  SKLEARN_N_JOBS: $SKLEARN_N_JOBS (full allocation)"
echo "  Node: $SLURM_NODELIST"

# Monitor resource usage
echo "Initial resource usage:"
free -h
top -bn1 | head -15

# Run the dimensionality reduction analysis with stable parallelization (skipping baseline)
echo "Starting SAFE MAXIMUM PERFORMANCE dimensionality reduction analysis (skipping baseline)..."
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