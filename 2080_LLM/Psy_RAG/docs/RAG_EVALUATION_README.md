# RAG Evaluation System

This module provides comprehensive evaluation capabilities for the RAG (Retrieval-Augmented Generation) system, comparing retrieval results against groundtruth data to assess accuracy and quality.

## Overview

The RAG evaluation system calculates precision, recall, F1-score, and accuracy metrics for each post by comparing retrieved criteria against manually annotated groundtruth data. It provides both macro and micro averages, detailed per-post results, and visualization capabilities.

## Features

- **Comprehensive Metrics**: Precision, Recall, F1-score, Accuracy
- **Both Aggregation Types**: Macro (average across posts) and Micro (aggregate across all predictions)
- **Detailed Analysis**: Per-post breakdown with confusion matrix components
- **Visualization**: Distribution plots, confusion matrices, and detailed reports
- **Flexible Evaluation**: Supports different similarity and confidence thresholds

## Files

- `src/models/rag_evaluator.py`: Core evaluation module with metrics calculation
- `evaluation_main.py`: Main script for running evaluations
- `test_evaluation.py`: Test script for verifying the evaluation system

## Usage

### Basic Evaluation

```bash
# Evaluate RAG retrieval accuracy with default settings
python evaluation_main.py --num_posts 100

# Evaluate with custom thresholds
python evaluation_main.py --num_posts 100 --similarity_threshold 0.6 --spanbert_threshold 0.4

# Generate detailed report with visualizations
python evaluation_main.py --num_posts 100 --generate_report
```

### Advanced Options

```bash
# Use custom data files
python evaluation_main.py \
    --posts "Data/mental_health_posts_negative_generated.csv" \
    --groundtruth "Data/Groundtruth/criteria_evaluation.csv" \
    --num_posts 500

# Save and reuse index for faster evaluation
python evaluation_main.py --num_posts 100 --save_index
python evaluation_main.py --num_posts 100 --load_index "Data/indices/evaluation_faiss_index"

# Specify output directory
python evaluation_main.py --num_posts 100 --output_dir "results/my_evaluation"
```

## Input Data Format

### Posts Data

- CSV file with post text and IDs
- Default: `Data/mental_health_posts_negative_generated.csv`

### Groundtruth Data

- CSV file with binary labels (0/1) for each criteria per post
- Columns format: `"Diagnosis - Criteria ID"` (e.g., "Major Depressive Disorder - A")
- Default: `Data/Groundtruth/criteria_evaluation.csv`

## Output

### Evaluation Results

The system generates several output files:

1. **evaluation_results.json**: Complete evaluation metrics and per-post results
2. **rag_results.json**: RAG pipeline results for reference
3. **evaluation_report.txt**: Human-readable summary report
4. **metrics_distribution.png**: Distribution plots of all metrics
5. **confusion_matrix.png**: Aggregated confusion matrix visualization

### Metrics Interpretation

- **Precision**: How many retrieved criteria were actually relevant (TP / (TP + FP))
- **Recall**: How many relevant criteria were successfully retrieved (TP / (TP + FN))
- **F1 Score**: Harmonic mean of precision and recall (2 _ P _ R / (P + R))
- **Accuracy**: Overall correctness ((TP + TN) / (TP + TN + FP + FN))

### Aggregation Types

- **Macro averages**: Equal weight to each post (average across posts)
- **Micro averages**: Weight by total number of predictions (aggregate across all criteria)

## Example Output

```
================================================================================
RAG RETRIEVAL EVALUATION SUMMARY
================================================================================
Total Posts Evaluated: 100

MACRO AVERAGES (average across posts):
  Precision: 0.7842
  Recall:    0.6943
  F1 Score:  0.7363
  Accuracy:  0.9126

MICRO AVERAGES (aggregate across all predictions):
  Precision: 0.7654
  Recall:    0.7012
  F1 Score:  0.7319

PERFORMANCE BREAKDOWN:
  High F1 (≥0.8):       45 posts (45.0%)
  Medium F1 (0.5-0.8):   35 posts (35.0%)
  Low F1 (<0.5):         20 posts (20.0%)
```

## Configuration

### Model Parameters

- `--similarity_threshold`: FAISS search similarity threshold (default: 0.7)
- `--spanbert_threshold`: SpanBERT confidence threshold (default: 0.5)
- `--top_k`: Number of top criteria to retrieve (default: 10)

### Performance Options

- `--save_index`: Save built FAISS index for reuse
- `--load_index`: Load pre-built FAISS index
- `--num_posts`: Limit number of posts to evaluate

## Testing

Run the test script to verify the evaluation system:

```bash
python test_evaluation.py
```

## Troubleshooting

### Common Issues

1. **Post ID Mismatch**: Ensure post IDs in RAG results match groundtruth data indexing
2. **Criteria Mapping**: Verify criteria naming consistency between RAG output and groundtruth
3. **Memory Issues**: Reduce `--num_posts` for large-scale evaluations
4. **FAISS Warnings**: Normal for small datasets; doesn't affect functionality

### Performance Tips

1. **Use Index Caching**: Save index with `--save_index` for repeated evaluations
2. **Batch Processing**: System automatically handles batching for large datasets
3. **GPU Acceleration**: Ensure CUDA is available for faster processing

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Evaluation metrics
- matplotlib: Visualization
- seaborn: Statistical plots
- torch: Deep learning framework
- faiss: Vector similarity search

## Integration

The evaluation system integrates seamlessly with the existing RAG pipeline:

```python
from src.models.rag_pipeline import RAGPipeline
from src.models.rag_evaluator import RAGEvaluator

# Initialize pipeline and evaluator
pipeline = RAGPipeline(...)
evaluator = RAGEvaluator(groundtruth_path)

# Run evaluation
rag_results = pipeline.evaluate_posts(num_posts=100)
evaluation_summary = evaluator.evaluate_results(rag_results)

# Save results
evaluator.save_evaluation_results(evaluation_summary, output_path)
```
