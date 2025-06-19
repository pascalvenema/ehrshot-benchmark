# Statistical Significance Analysis for EHRSHOT Benchmark

## Overview

This document explains how to perform proper statistical significance testing for your EHRSHOT benchmark results, specifically addressing:

1. **ClinicalBERT Type Comparison**: Are there significant differences between Type1, Type2, and Type3 embeddings?
2. **RQ1**: How does CLMBR compare to ClinicalBERT when using shared prediction heads?
3. **RQ2**: Do non-linear prediction heads significantly outperform linear heads?

## Why Standard t-tests Won't Work

You're absolutely right that simple t-tests aren't appropriate here. Here's why:

### 1. **Multiple Testing Problem**
- Comparing 3 ClinicalBERT types across ~14 tasks = many comparisons
- Type I error inflation without correction
- Need multiple testing correction (e.g., Holm-Bonferroni)

### 2. **Non-Independence of Observations**
- Same models tested on different tasks
- Tasks may have different difficulty levels
- Need paired tests when comparing on same tasks

### 3. **Distribution Assumptions**
- Performance scores may not be normally distributed
- Small sample sizes for some comparisons
- Non-parametric tests are more robust

### 4. **Experimental Design**
- Multiple replicates per condition
- Mixed-effects structure (models × tasks × replicates)
- Need to account for experimental structure

## Recommended Statistical Approaches

### For ClinicalBERT Type Comparisons

#### 1. **Overall Comparison (Kruskal-Wallis Test)**
```python
# Test if ANY difference exists between types
kw_stat, p_value = kruskal(type1_scores, type2_scores, type3_scores)
```
- **Use when**: Testing overall difference between 3+ groups
- **Assumptions**: Independent observations, ordinal/continuous outcome
- **Null hypothesis**: All groups have same distribution

#### 2. **Pairwise Comparisons (Mann-Whitney U)**
```python
# Compare each pair of types
stat, p_value = mannwhitneyu(type1_scores, type3_scores, alternative='two-sided')
```
- **Use when**: Comparing two groups
- **Advantages**: No normality assumption, robust to outliers
- **Post-hoc**: Apply after significant Kruskal-Wallis test

#### 3. **Multiple Testing Correction**
```python
from statsmodels.stats.multitest import multipletests
rejected, p_corrected, _, _ = multipletests(p_values, method='holm')
```
- **Methods**: Holm (recommended), Bonferroni (conservative), FDR (liberal)
- **Why needed**: Control family-wise error rate across multiple tests

### For RQ1: CLMBR vs ClinicalBERT

#### **Paired Analysis (Wilcoxon Signed-Rank Test)**
```python
# Compare performance on same tasks
wilcoxon_stat, p_value = wilcoxon(clmbr_task_scores, cb_task_scores, 
                                 alternative='two-sided')
```
- **Use when**: Comparing same models on same tasks
- **Advantages**: Controls for task difficulty, more powerful than unpaired tests
- **Effect size**: Cohen's d for magnitude of difference

### For RQ2: Linear vs Non-Linear Heads

#### **One-Sample Tests on Improvements**
```python
# Test if mean improvement > 0
improvements = nonlinear_scores - linear_scores
wilcoxon_stat, p_value = wilcoxon(improvements, alternative='greater')
```
- **Rationale**: Test if non-linear consistently improves over linear
- **Directional**: One-sided test (we expect non-linear > linear)

## Effect Size Interpretation

### Cohen's d Guidelines
- **|d| < 0.2**: Negligible effect
- **0.2 ≤ |d| < 0.5**: Small effect  
- **0.5 ≤ |d| < 0.8**: Medium effect
- **|d| ≥ 0.8**: Large effect

### Bootstrap Confidence Intervals
```python
# 95% CI for mean difference
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)
```

## Usage

### Quick Start
```bash
# Run complete statistical analysis
./ehrshot/bash_scripts/run_statistical_significance_analysis.sh

# Or with custom parameters
python ehrshot/statistical_significance_analysis.py \
    --path_to_results_dir EHRSHOT_ASSETS/results \
    --path_to_output_dir statistical_results \
    --alpha 0.05 \
    --scores auroc auprc
```

### Expected Output

#### 1. **Console Output**
- Task-by-task comparison tables
- Overall test statistics
- Effect sizes and confidence intervals
- Multiple testing correction results

#### 2. **Summary Files**
- `statistical_summary_auroc.txt`: Text summary of all results
- `statistical_summary_auprc.txt`: Same for AUPRC

## Interpreting Results

### For ClinicalBERT Types
Look for:
- **Kruskal-Wallis p-value < 0.05**: Significant overall difference
- **Post-hoc comparisons**: Which specific pairs differ
- **Effect sizes**: Magnitude of differences (Cohen's d)
- **Multiple testing**: Still significant after correction?

### For RQ1 (CLMBR vs ClinicalBERT)
Look for:
- **Paired test p-value**: Is difference statistically significant?
- **Mean difference**: CLMBR - ClinicalBERT (positive = CLMBR better)
- **95% CI**: Does it include 0? (if not, significant difference)
- **Win rate**: On how many tasks does CLMBR win?

### For RQ2 (Linear vs Non-Linear)
Look for:
- **One-sample test p-value**: Is improvement significant?
- **Proportion improved**: How many tasks show improvement?
- **Effect size**: How large is the improvement?

## Sample Interpretation

```
🧪 STATISTICAL SIGNIFICANCE ANALYSIS: ClinicalBERT Types (AUROC)
===============================================================================

Kruskal-Wallis test:
  H-statistic: 15.432
  p-value: 0.000443
  Significant: ***

Post-hoc pairwise comparisons (overall):
  type1 vs type3: diff=-0.045 [-0.067, -0.023], p=0.001 **
  type2 vs type3: diff=-0.032 [-0.051, -0.013], p=0.015 *

Type3 vs Type1 effect size: Cohen's d = 0.623 (medium)
```

**Interpretation**: 
- Type3 significantly outperforms Type1 and Type2
- Medium-sized effect (d=0.623)
- Difference is robust (significant after multiple testing correction)

## Advanced Considerations

### When You Have More Data
- **Mixed-effects models**: Account for task and model random effects
- **Bayesian analysis**: Posterior probabilities instead of p-values
- **Permutation tests**: Exact p-values without distributional assumptions

### Power Analysis
```python
# Estimate sample size needed for desired power
from statsmodels.stats.power import ttest_power
power = ttest_power(effect_size=0.5, nobs=n_tasks, alpha=0.05)
```

### Robustness Checks
- **Bootstrap resampling**: Non-parametric confidence intervals
- **Cross-validation**: Out-of-sample performance validation
- **Sensitivity analysis**: Results stable across different α levels?

## Common Pitfalls to Avoid

1. **Multiple testing without correction** → Inflated Type I error
2. **Using unpaired tests on paired data** → Loss of power
3. **Ignoring effect sizes** → Statistical vs practical significance
4. **Cherry-picking comparisons** → p-hacking
5. **Assuming normality without checking** → Invalid test assumptions

## Further Reading

- **Multiple Testing**: Hochberg & Tamhane (1987)
- **Non-parametric Tests**: Hollander & Wolfe (2013)
- **Effect Sizes**: Cohen (1988), "Statistical Power Analysis"
- **Medical Statistics**: Altman (1991), "Practical Statistics for Medical Research" 