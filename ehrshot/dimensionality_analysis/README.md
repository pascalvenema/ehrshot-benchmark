# Dimensionality Reduction Analysis for EHRSHOT

This module analyzes the impact of dimensionality reduction on ClinicalBERT type3 embeddings using kNN classifiers.

## Overview

The curse of dimensionality can negatively impact kNN performance on high-dimensional embeddings (768D ClinicalBERT). This analysis tests whether dimensionality reduction techniques can improve kNN performance on clinical prediction tasks.

## What the Code Does

### Simple Workflow:
1. **Load ClinicalBERT type3 embeddings** (768 dimensions) for selected clinical tasks
2. **Apply dimensionality reduction** using PCA, t-SNE, and UMAP to various target dimensions
3. **Train kNN classifiers** on the reduced embeddings with hyperparameter tuning
4. **Evaluate performance** using AUROC and AUPRC metrics
5. **Generate plots and summaries** comparing performance across methods and dimensions

### Files Structure:
```
ehrshot/
├── 10_dimensionality_reduction.py          # Main analysis script
├── bash_scripts/
│   └── 10_dimensionality_reduction.sh      # Bash script to run analysis
└── dimensionality_analysis/
    ├── __init__.py                         # Module initialization
    ├── reducers.py                         # PCA, t-SNE, UMAP implementations
    ├── evaluator.py                        # kNN evaluation with reduced embeddings
    ├── plotting.py                         # Visualization functions
    └── README.md                           # This file
```

## Key Features

### Dimensionality Reduction Methods:
- **PCA**: Linear dimensionality reduction, preserves global structure
- **t-SNE**: Non-linear, good for local structure preservation 
- **UMAP**: Non-linear, balances local and global structure

### Evaluation:
- **kNN Classifier**: Hyperparameter tuning for k, weights, and distance metrics
- **Metrics**: AUROC and AUPRC for binary classification
- **Baselines**: Compares reduced dimensions against full 768D embeddings

### Tasks Analyzed (Default):
- ICU Admission (operational outcome)
- 30-day Readmission (operational outcome)  
- Hypertension (new diagnosis)
- Hyperkalemia (lab value prediction)

## Usage

### Basic Usage:
```bash
cd ehrshot/bash_scripts
./10_dimensionality_reduction.sh
```

### Advanced Usage:
```bash
python ehrshot/10_dimensionality_reduction.py \
    --path_to_database path/to/femr/database \
    --path_to_labels_dir path/to/labels \
    --path_to_features_dir path/to/features \
    --path_to_output_dir path/to/output \
    --path_to_split_csv path/to/splits.csv \
    --dimensions 2 5 10 20 50 100 200 \
    --methods pca umap tsne \
    --tasks guo_icu guo_readmission \
    --k_shot -1
```

## Output Files

### Results:
- `dimensionality_results.csv`: Raw results for all experiments
- `dimensionality_summary.csv`: Summary table with best performance per method
- `dimensionality_analysis.log`: Detailed execution log

### Visualizations:
- `dimensionality_analysis_overview.png`: Main comparison plots
- `best_dimensions_heatmap.png`: Heatmap showing optimal dimensions
- `multi_task_comparison.png`: Task-by-task comparison (if multiple tasks)

## Key Research Questions Addressed

1. **Does dimensionality reduction improve kNN performance?**
   - Compares reduced vs. full 768D embeddings

2. **Which reduction method works best for clinical embeddings?**
   - Tests PCA (linear) vs. t-SNE/UMAP (non-linear)

3. **What are the optimal dimensions for each method?**
   - Tests range from 2D to 200D

4. **Are results consistent across different clinical tasks?**
   - Evaluates on diverse task types (operational, diagnostic, lab values)

## Implementation Details

### Why These Methods?
- **PCA**: Fast, interpretable, good baseline for linear reduction
- **t-SNE**: Popular for visualization, captures local neighborhoods well
- **UMAP**: Modern alternative to t-SNE, faster and preserves more global structure

### kNN Configuration:
- **Exactly matches main EHRSHOT pipeline**: k ∈ {1,3,5}, weights ∈ {uniform, distance}, metrics ∈ {euclidean, cosine}
- Uses train/validation split for hyperparameter selection
- Final evaluation on held-out test set

### Considerations:
- **t-SNE Limitation**: Cannot transform new data, requires refitting for test set
- **Sample Size**: Adjusts hyperparameters based on training set size for few-shot scenarios
- **Scaling**: Applies MaxAbsScaler to reduced features before kNN training

## Expected Outcomes

The analysis will show:
1. Whether curse of dimensionality affects kNN on ClinicalBERT embeddings
2. If certain clinical tasks benefit more from dimensionality reduction
3. Optimal dimensionality ranges for each reduction method
4. Trade-offs between computational efficiency and performance

This provides evidence for **RQ2.2** in your methodology: "How does the dimensionality of patient embeddings affect the relative performance of different prediction heads?" 