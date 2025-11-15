# RAG System Comparison Feature

## Overview

This feature enhances the RAG system to compare retrieval results between negative and positive mental health posts, providing statistical analysis of differences in DSM-5 criteria matching.

## New Files Added

### Core Implementation

- `src/models/comparison_rag_pipeline.py` - Enhanced RAG pipeline with comparison capabilities
- `comparison_main.py` - Main script for running comparisons
- `test_stats_only.py` - Standalone test for statistical functions
- `test_comparison.py` - Full functionality test (requires ML environment)

### Data Requirements

- `Data/mental_health_posts_negative_generated.csv` - Negative posts (generated dataset)
- `Data/translated_post_positive.csv` - Positive posts (new requirement)

## Features

### 1. Dual Post Type Support

- Handles both negative and positive posts with different CSV formats
- Automatically detects column names (`translated_post` vs `positive_post`)
- Preprocessing optimized for both data types

### 2. Comprehensive Metrics

- **Average Similarity Score**: Semantic similarity to DSM-5 criteria
- **Average SpanBERT Score**: Contextual matching confidence
- **Combined Score**: Average of similarity and SpanBERT scores
- **95% Confidence Intervals**: Statistical bounds for each metric

### 3. Statistical Analysis

- **Independent t-test**: For normally distributed data with similar variances
- **Mann-Whitney U test**: For non-normal or unequal variance data
- **P-value calculation**: Statistical significance testing (α = 0.05)
- **Custom implementation**: No dependency on scipy for compatibility

### 4. Results Output

- Comprehensive statistics summary
- Detailed JSON results with all metadata
- Timestamp and sample size tracking
- Interpretation guidance

## Usage

### Basic Comparison

```bash
python comparison_main.py --num_negative 100 --num_positive 100
```

### Advanced Options

```bash
python comparison_main.py \
  --num_negative 1000 \
  --num_positive 800 \
  --similarity_threshold 0.7 \
  --spanbert_threshold 0.5 \
  --top_k 10 \
  --save_index \
  --output_file results/my_comparison.json
```

### Parameters

- `--num_negative`: Number of negative posts to analyze (default: 1000)
- `--num_positive`: Number of positive posts to analyze (default: 1000)
- `--similarity_threshold`: FAISS similarity threshold (default: 0.7)
- `--spanbert_threshold`: SpanBERT confidence threshold (default: 0.5)
- `--top_k`: Number of criteria to retrieve per post (default: 10)
- `--save_index`: Save FAISS index for reuse
- `--load_index`: Load existing FAISS index
- `--output_file`: Custom output file path

## Output Format

### Console Output

```
================================================================================
RAG RETRIEVAL COMPARISON: NEGATIVE vs POSITIVE POSTS
================================================================================
Sample Sizes: 1000 negative posts, 800 positive posts
--------------------------------------------------------------------------------

COMBINED SCORE ANALYSIS:
  Negative Posts:
    Average Score: 0.3596
    95% CI: [0.3488, 0.3705]
  Positive Posts:
    Average Score: 0.1932
    95% CI: [0.1850, 0.2015]
  Difference (Positive - Negative): -0.1664
  Statistical Test: Independent t-test
  Test Statistic: 23.8326
  P-value: 0.000000
  Significant at α=0.05: YES
  → Positive posts have significantly lower scores than negative posts
```

### JSON Results Structure

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "negative_posts_count": 1000,
  "positive_posts_count": 800,
  "statistics": {
    "combined_score": {
      "negative_avg_score": 0.3596,
      "positive_avg_score": 0.1932,
      "negative_lower_bound": 0.3488,
      "negative_upper_bound": 0.3705,
      "positive_lower_bound": 0.1850,
      "positive_upper_bound": 0.2015,
      "p_value": 0.000001,
      "statistic": 23.8326,
      "significant": true,
      "test_type": "Independent t-test"
    }
  },
  "negative_results": [...],
  "positive_results": [...]
}
```

## Statistical Methods

### Test Selection

1. **Normality Assessment**: Based on sample size and variance comparison
2. **t-test**: Used when n≥5 for both samples and variance ratio <4
3. **Mann-Whitney U**: Used for small samples or unequal variances
4. **Conservative approach**: Ensures reliable results across different data distributions

### Confidence Intervals

- 95% confidence intervals using normal approximation
- Standard error calculation with proper degrees of freedom
- Margin of error: 1.96 × standard error

## Expected Results

Based on mental health literature, we expect:

- **Negative posts**: Higher scores (more criteria matches)
- **Positive posts**: Lower scores (fewer criteria matches)
- **Statistical significance**: Likely significant difference (p < 0.05)

This aligns with the hypothesis that negative mental health posts contain more content matching DSM-5 diagnostic criteria.

## Testing

### Statistical Functions Test

```bash
python test_stats_only.py
```

Tests the custom statistical implementations without ML dependencies.

### Full System Test (requires working ML environment)

```bash
python test_comparison.py
```

Tests the complete comparison pipeline.

## Troubleshooting

### Environment Issues

If you encounter library compatibility issues:

1. The statistical functions are tested independently
2. Core functionality is implemented without heavy dependencies
3. Custom statistical methods avoid scipy compatibility issues

### Data Format Issues

- Ensure `mental_health_posts_negative_generated.csv` has `post` column
- Ensure `translated_post_positive.csv` has `positive_post` column
- Both files should be UTF-8 encoded

### Memory Issues

- Reduce sample sizes using `--num_negative` and `--num_positive`
- Use smaller `--top_k` values
- Consider batch processing for large datasets

## Implementation Details

### Key Classes

- `ComparisonRAGPipeline`: Main comparison orchestrator
- `ComparisonStatistics`: Statistics container with all metrics
- `PostEvaluationResult`: Individual post evaluation result

### Statistical Functions

- `calculate_t_test()`: Custom t-test implementation
- `mann_whitney_u_test()`: Custom Mann-Whitney U implementation
- `normal_cdf()`: Standard normal CDF for p-value calculation

### Performance Optimizations

- Batch processing for embeddings
- Periodic cache clearing
- Efficient memory management for large datasets
