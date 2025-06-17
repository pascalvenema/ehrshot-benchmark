# Dimensionality Reduction Analysis for EHRSHOT

This module implements dimensionality reduction techniques to analyze the impact of reduced embeddings on clinical prediction performance, with a particular focus on k-NN models and the curse of dimensionality.

## Overview

The dimensionality reduction analysis addresses **Research Question 2.2**: "How does the dimensionality of the patient embeddings affect the relative performance of different prediction heads?"

This is particularly relevant for k-NN models, which can suffer from the curse of dimensionality when working with high-dimensional embeddings like ClinicalBERT's 768-dimensional representations.

## Module Structure

```
ehrshot/dimensionality_reduction/
├── __init__.py                 # Module initialization
├── reducers.py                 # Dimensionality reduction techniques
├── utils.py                    # Utility functions for loading, saving, plotting
├── evaluator.py                # Main evaluation framework
└── README.md                   # This file
```

## Reduction Methods Implemented

1. **PCA (Principal Component Analysis)**: Linear dimensionality reduction preserving maximum variance
2. **Truncated SVD**: Similar to PCA but works better with sparse matrices
3. **Factor Analysis**: Probabilistic model assuming latent factors
4. **Variance Threshold**: Removes features with low variance
5. **Univariate Feature Selection**: Selects features based on statistical tests

## Scripts

### Main Analysis Script

**`ehrshot/10_dimensionality_reduction.py`** - Comprehensive analysis pipeline

```bash
python3 10_dimensionality_reduction.py \
    --path_to_database '../EHRSHOT_ASSETS/femr/extract' \
    --path_to_labels_dir '../EHRSHOT_ASSETS/benchmark' \
    --path_to_features_dir '../EHRSHOT_ASSETS/features' \
    --path_to_split_csv '../EHRSHOT_ASSETS/splits/person_id_map.csv' \
    --path_to_output_dir '../EHRSHOT_ASSETS/dimensionality_reduction' \
    --labeling_functions 'guo_icu,guo_los,lab_anemia' \
    --reduction_methods 'pca,truncated_svd' \
    --dimensions '32,64,128,256' \
    --prediction_heads 'knn,lr_lbfgs,rf' \
    --num_threads 10 \
    --create_visualizations
```

### Quick k-NN Analysis

**`ehrshot/11_knn_dimensionality_analysis.py`** - Focused k-NN analysis

```bash
python3 11_knn_dimensionality_analysis.py \
    --path_to_features '../EHRSHOT_ASSETS/features/clinicalbert_type3_clinicalbert_pool_features.pkl' \
    --path_to_labels '../EHRSHOT_ASSETS/benchmark/guo_icu_labels.csv' \
    --path_to_output_dir '../EHRSHOT_ASSETS/knn_analysis' \
    --dimensions '16,32,64,128,256,384' \
    --reduction_methods 'pca,truncated_svd'
```

### Bash Script

**`ehrshot/bash_scripts/10_dimensionality_reduction.sh`** - SLURM-compatible execution

```bash
cd ehrshot/bash_scripts
sbatch 10_dimensionality_reduction.sh
# Or run directly:
bash 10_dimensionality_reduction.sh
```

## Key Research Questions Addressed

### RQ2.2: Dimensionality Impact on Prediction Heads

- **k-NN Performance**: How does k-NN performance change with reduced dimensions?
- **Optimal Dimensionality**: What is the optimal number of dimensions for k-NN?
- **Method Comparison**: Which reduction method works best for k-NN?
- **Curse of Dimensionality**: Do we see evidence of improved performance with dimension reduction?

## Expected Outputs

### Directory Structure
```
EHRSHOT_ASSETS/dimensionality_reduction/
├── analysis/
│   └── explained_variance_analysis.png
├── visualizations/
│   ├── embeddings_2d_pca.png
│   └── embeddings_2d_tsne.png
├── reduced_features/
│   ├── clinicalbert_type3_clinicalbert_pool_pca_32d_features.pkl
│   ├── clinicalbert_type3_clinicalbert_pool_pca_64d_features.pkl
│   └── ...
├── reports/
│   ├── dimensionality_reduction_summary.csv
│   ├── best_dimensionality_combinations.csv
│   ├── dimensionality_comparison_auroc.png
│   └── dimensionality_comparison_auprc.png
└── dimensionality_reduction_results.csv
```

### Key Files

1. **`dimensionality_reduction_results.csv`**: Complete results for all experiments
2. **`reports/dimensionality_reduction_summary.csv`**: Aggregated performance statistics
3. **`reports/best_dimensionality_combinations.csv`**: Best performing method/dimension combinations per task
4. **Visualization plots**: Performance comparisons and 2D embeddings

## Usage Examples

### Basic Usage

```python
from ehrshot.dimensionality_reduction import DimensionalityReductionEvaluator

# Initialize evaluator
evaluator = DimensionalityReductionEvaluator(
    path_to_database='../EHRSHOT_ASSETS/femr/extract',
    path_to_labels_dir='../EHRSHOT_ASSETS/benchmark',
    path_to_features_dir='../EHRSHOT_ASSETS/features',
    path_to_split_csv='../EHRSHOT_ASSETS/splits/person_id_map.csv',
    path_to_output_dir='../EHRSHOT_ASSETS/dimensionality_reduction'
)

# Run experiments
results_df = evaluator.run_full_experiment(
    labeling_functions=['guo_icu', 'lab_anemia'],
    k_shots=[-1]  # Full data
)
```

### Custom Reduction

```python
from ehrshot.dimensionality_reduction.reducers import create_reducer
from ehrshot.dimensionality_reduction.utils import load_features

# Load features
features, patient_ids, times = load_features('path_to_features.pkl')

# Apply PCA reduction
reducer = create_reducer('pca', n_components=128)
reduced_features = reducer.fit_transform(features)

print(f"Reduced from {features.shape[1]}D to {reduced_features.shape[1]}D")
print(f"Explained variance: {reducer.get_explained_variance_ratio()[:5]}")
```

## Performance Expectations

### Computational Requirements
- **Memory**: 64GB recommended for full dataset
- **Time**: 2-12 hours depending on configuration
- **CPU**: Benefits from multi-threading (10+ cores recommended)

### Expected Improvements for k-NN
Based on literature on curse of dimensionality:
- k-NN performance often improves with dimension reduction
- Optimal dimensions typically in 32-256 range for clinical data
- PCA and Truncated SVD usually perform similarly
- Performance improvements most notable for few-shot scenarios

## Configuration Options

### Reduction Methods
- `pca`: Principal Component Analysis (recommended)
- `truncated_svd`: Truncated SVD
- `factor_analysis`: Factor Analysis
- `variance_threshold`: Variance-based feature selection
- `univariate_selection`: Statistical feature selection

### Dimensions to Test
Recommended ranges:
- **Small**: 16, 32, 64 (for very constrained scenarios)
- **Medium**: 128, 256 (good balance)
- **Large**: 384, 512 (approaching original size)

### Prediction Heads
Focus on k-NN for curse of dimensionality analysis:
- `knn`: k-Nearest Neighbors (primary focus)
- `lr_lbfgs`: Logistic Regression (baseline)
- `rf`: Random Forest (comparison)

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `max_samples` or use fewer dimensions
2. **Import Error**: Ensure all dependencies are installed
3. **File Not Found**: Check paths to EHRSHOT_ASSETS directory
4. **Performance Issues**: Use fewer tasks or dimensions for initial testing

### Debug Mode

Add logging for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Main EHRSHOT Pipeline

This module integrates seamlessly with the existing EHRSHOT evaluation pipeline:

1. Uses the same feature loading utilities
2. Compatible with existing evaluation metrics
3. Follows the same train/val/test split methodology
4. Outputs results in the same CSV format as other experiments

## Citation

If you use this dimensionality reduction analysis in your research, please cite the original EHRSHOT paper and mention the dimensionality reduction extension:

```bibtex
@article{wornow2023ehrshot,
  title={EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models},
  author={Wornow, Michael and others},
  journal={arXiv preprint arXiv:2307.02028},
  year={2023}
}
``` 