#!/bin/bash

# NOTE: To run with slurm, pass the `--is_use_slurm` flag
# NOTE: Jobs will run sequentially (one after another) to avoid resource conflicts

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

path_to_database='../../EHRSHOT_ASSETS/femr/extract'
path_to_labels_dir='../../EHRSHOT_ASSETS/benchmark'
path_to_features_dir='../../EHRSHOT_ASSETS/features'
path_to_output_dir='../../EHRSHOT_ASSETS/results'
path_to_split_csv='../../EHRSHOT_ASSETS/splits/person_id_map.csv'

labeling_functions=(
    # "chexpert" # CheXpert first b/c slowest
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    # Labs take long time -- need more GB
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_anemia"
    "lab_hypoglycemia" # will OOM at 200G on `gpu` partition
)
shot_strats=("all")
num_threads=20

# Initialize variables for sequential job execution
previous_job_id=""

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        if [[ " $* " == *" --is_use_slurm "* ]]; then
            # Submit job with dependency on previous job (if any)
            if [ -z "$previous_job_id" ]; then
                # First job - no dependency
                job_output=$(sbatch 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads)
            else
                # Subsequent jobs - depend on previous job completion
                job_output=$(sbatch --dependency=afterok:$previous_job_id 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads)
            fi
            
            # Extract job ID from sbatch output (format: "Submitted batch job XXXXXX")
            previous_job_id=$(echo "$job_output" | grep -o '[0-9]\+$')
            echo "Submitted job $previous_job_id for $labeling_function (depends on: ${previous_job_id:-none})"
        else
            # Non-slurm execution - jobs already run sequentially by default
            echo "Running $labeling_function..."
            bash 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads
            echo "Completed $labeling_function"
        fi
    done
done

if [[ " $* " == *" --is_use_slurm "* ]]; then
    echo "All jobs submitted with sequential dependencies. Check job queue with 'squeue -u \$USER'"
fi
