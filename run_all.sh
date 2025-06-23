#!/bin/bash

cd ehrshot/bash_scripts

# Train baseline models and generate metrics.
# bash 7_eval.sh

# Generate plots
bash 8_make_results_plots.sh

# Generate cohort statistics
# bash 9_make_cohort_plots.sh
