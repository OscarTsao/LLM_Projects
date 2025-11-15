"""
RAG Evaluation module for comparing retrieval results with groundtruth data
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from .rag_pipeline import RAGResult, CriteriaMatch

logger = logging.getLogger(__name__)


@dataclass
class PostEvaluationResult:
    """Evaluation result for a single post"""
    post_id: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    retrieved_criteria: List[str]
    groundtruth_criteria: List[str]
    post_text: str


@dataclass
class EvaluationSummary:
    """Summary of evaluation results across all posts"""
    total_posts: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    average_accuracy: float
    post_results: List[PostEvaluationResult]


class RAGEvaluator:
    """Evaluator for RAG retrieval quality using groundtruth data"""

    def __init__(self, groundtruth_path: Path):
        """
        Initialize the evaluator

        Args:
            groundtruth_path: Path to groundtruth CSV file
        """
        self.groundtruth_path = groundtruth_path
        self.groundtruth_data = None
        self.criteria_mapping = None

        self._load_groundtruth()
        logger.info("RAG Evaluator initialized")

    def _load_groundtruth(self):
        """Load and preprocess groundtruth data"""
        try:
            logger.info(f"Loading groundtruth data from {self.groundtruth_path}")

            # Load the CSV file
            self.groundtruth_data = pd.read_csv(self.groundtruth_path)

            # Extract criteria column names (all columns except post_id and post content)
            criteria_columns = [col for col in self.groundtruth_data.columns
                              if col not in ['post_id'] and not col.startswith('Post:')]

            # Create mapping of criteria names to simplified IDs
            self.criteria_mapping = {col: col for col in criteria_columns}

            logger.info(f"Loaded groundtruth for {len(self.groundtruth_data)} posts with {len(criteria_columns)} criteria")

        except Exception as e:
            logger.error(f"Error loading groundtruth data: {e}")
            raise

    def _extract_criteria_id(self, criteria_text: str) -> str:
        """
        Extract criteria ID from criteria text
        This method needs to map RAG pipeline criteria to groundtruth criteria
        """
        # For now, return the criteria text as is
        # This should be customized based on your criteria mapping
        return criteria_text

    def _get_post_groundtruth(self, post_id: str) -> List[str]:
        """Get groundtruth criteria for a specific post"""
        try:
            # The groundtruth data uses 1-based indexing in the CSV (row 2 = post 1, etc.)
            # But post_id from RAG is 0-based, so we need to adjust
            try:
                post_index = int(post_id)
                # Add 2 because row 1 is header, row 2 is post 0, row 3 is post 1, etc.
                row_index = post_index + 2

                if row_index > len(self.groundtruth_data):
                    logger.warning(f"Post index {post_id} out of range")
                    return []

                post_row = self.groundtruth_data.iloc[post_index]

            except (ValueError, IndexError):
                # Fallback: try to match by string
                post_row = self.groundtruth_data[self.groundtruth_data.iloc[:, 0].astype(str) == str(post_id)]
                if post_row.empty:
                    logger.warning(f"No groundtruth found for post {post_id}")
                    return []
                post_row = post_row.iloc[0]

            # Get criteria that are marked as 1 (positive)
            positive_criteria = []

            for col in self.groundtruth_data.columns:
                if col not in ['post_id'] and not col.startswith('Post:'):
                    if post_row[col] == 1:
                        positive_criteria.append(col)

            return positive_criteria

        except Exception as e:
            logger.error(f"Error getting groundtruth for post {post_id}: {e}")
            return []

    def _get_retrieved_criteria(self, rag_result: RAGResult) -> List[str]:
        """Extract retrieved criteria from RAG result"""
        try:
            retrieved = []
            for match in rag_result.matched_criteria:
                if match.is_match:  # Only consider actual matches
                    # Get the criteria ID from the match
                    criteria_cols = self._map_criteria_to_groundtruth(match.diagnosis, match.criteria_id)
                    if criteria_cols:
                        retrieved.extend(criteria_cols)

            return list(set(retrieved))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error extracting retrieved criteria: {e}")
            return []

    def _map_criteria_to_groundtruth(self, diagnosis: str, criteria_id: str) -> List[str]:
        """
        Map RAG pipeline criteria to groundtruth criteria format
        This maps diagnosis + criteria ID to the groundtruth CSV column format
        """
        try:
            # The groundtruth columns are in format: "Diagnosis - CriteriaID"
            # For example: "Major Depressive Disorder - A"

            matched_columns = []

            # Look for exact matches first
            target_col = f"{diagnosis} - {criteria_id}"
            if target_col in self.criteria_mapping:
                matched_columns.append(target_col)

            # Also look for columns that contain both diagnosis and criteria ID
            for col in self.criteria_mapping.keys():
                if (diagnosis in col and
                    criteria_id in col and
                    col not in matched_columns):
                    matched_columns.append(col)

            return matched_columns

        except Exception as e:
            logger.warning(f"Error mapping criteria {diagnosis} - {criteria_id}: {e}")
            return []

    def evaluate_post(self, rag_result: RAGResult) -> PostEvaluationResult:
        """
        Evaluate RAG result for a single post against groundtruth

        Args:
            rag_result: RAG result for the post

        Returns:
            PostEvaluationResult with metrics
        """
        try:
            post_id = str(rag_result.post_id)

            # Get groundtruth and retrieved criteria
            groundtruth_criteria = self._get_post_groundtruth(post_id)
            retrieved_criteria = self._get_retrieved_criteria(rag_result)

            # Get all possible criteria for binary classification
            all_criteria = list(self.criteria_mapping.keys())

            # Create binary vectors for evaluation
            y_true = [1 if criteria in groundtruth_criteria else 0 for criteria in all_criteria]
            y_pred = [1 if criteria in retrieved_criteria else 0 for criteria in all_criteria]

            # Calculate metrics
            if sum(y_true) == 0 and sum(y_pred) == 0:
                # No positive cases in either true or predicted
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            elif sum(y_true) == 0:
                # No positive cases in true labels
                precision = 0.0 if sum(y_pred) > 0 else 1.0
                recall = 1.0  # No positive cases to miss
                f1 = 0.0
            elif sum(y_pred) == 0:
                # No positive predictions
                precision = 1.0  # No false positives
                recall = 0.0
                f1 = 0.0
            else:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

            accuracy = accuracy_score(y_true, y_pred)

            # Calculate confusion matrix components
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

            return PostEvaluationResult(
                post_id=post_id,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                retrieved_criteria=retrieved_criteria,
                groundtruth_criteria=groundtruth_criteria,
                post_text=rag_result.post_text
            )

        except Exception as e:
            logger.error(f"Error evaluating post {rag_result.post_id}: {e}")
            return PostEvaluationResult(
                post_id=str(rag_result.post_id),
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                retrieved_criteria=[],
                groundtruth_criteria=[],
                post_text=rag_result.post_text
            )

    def evaluate_results(self, rag_results: List[RAGResult]) -> EvaluationSummary:
        """
        Evaluate multiple RAG results against groundtruth

        Args:
            rag_results: List of RAG results

        Returns:
            EvaluationSummary with overall metrics
        """
        try:
            logger.info(f"Evaluating {len(rag_results)} RAG results")

            # Evaluate each post
            post_results = []
            for rag_result in rag_results:
                post_eval = self.evaluate_post(rag_result)
                post_results.append(post_eval)

            # Calculate macro averages (average across posts)
            macro_precision = np.mean([r.precision for r in post_results])
            macro_recall = np.mean([r.recall for r in post_results])
            macro_f1 = np.mean([r.f1_score for r in post_results])
            average_accuracy = np.mean([r.accuracy for r in post_results])

            # Calculate micro averages (aggregate across all predictions)
            total_tp = sum(r.true_positives for r in post_results)
            total_fp = sum(r.false_positives for r in post_results)
            total_fn = sum(r.false_negatives for r in post_results)

            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

            summary = EvaluationSummary(
                total_posts=len(post_results),
                macro_precision=macro_precision,
                macro_recall=macro_recall,
                macro_f1=macro_f1,
                micro_precision=micro_precision,
                micro_recall=micro_recall,
                micro_f1=micro_f1,
                average_accuracy=average_accuracy,
                post_results=post_results
            )

            logger.info(f"Evaluation completed: Macro F1={macro_f1:.4f}, Micro F1={micro_f1:.4f}")
            return summary

        except Exception as e:
            logger.error(f"Error evaluating results: {e}")
            raise

    def save_evaluation_results(self, summary: EvaluationSummary, filepath: Path):
        """Save evaluation results to JSON file"""
        try:
            # Convert to serializable format
            results_data = {
                "summary": {
                    "total_posts": summary.total_posts,
                    "macro_precision": summary.macro_precision,
                    "macro_recall": summary.macro_recall,
                    "macro_f1": summary.macro_f1,
                    "micro_precision": summary.micro_precision,
                    "micro_recall": summary.micro_recall,
                    "micro_f1": summary.micro_f1,
                    "average_accuracy": summary.average_accuracy
                },
                "post_results": []
            }

            for post_result in summary.post_results:
                post_data = {
                    "post_id": post_result.post_id,
                    "precision": post_result.precision,
                    "recall": post_result.recall,
                    "f1_score": post_result.f1_score,
                    "accuracy": post_result.accuracy,
                    "true_positives": post_result.true_positives,
                    "false_positives": post_result.false_positives,
                    "true_negatives": post_result.true_negatives,
                    "false_negatives": post_result.false_negatives,
                    "retrieved_criteria": post_result.retrieved_criteria,
                    "groundtruth_criteria": post_result.groundtruth_criteria,
                    "post_text": post_result.post_text
                }
                results_data["post_results"].append(post_data)

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Evaluation results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise

    def generate_evaluation_report(self, summary: EvaluationSummary, output_dir: Path):
        """Generate visualization and reports for evaluation results"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create visualizations
            self._plot_metrics_distribution(summary, output_dir)
            self._plot_confusion_matrix_summary(summary, output_dir)
            self._create_detailed_report(summary, output_dir)

            logger.info(f"Evaluation report generated in {output_dir}")

        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            raise

    def _plot_metrics_distribution(self, summary: EvaluationSummary, output_dir: Path):
        """Plot distribution of evaluation metrics across posts"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Precision distribution
            precisions = [r.precision for r in summary.post_results]
            axes[0, 0].hist(precisions, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(summary.macro_precision, color='red', linestyle='--',
                             label=f'Mean: {summary.macro_precision:.3f}')
            axes[0, 0].set_title('Precision Distribution')
            axes[0, 0].set_xlabel('Precision')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()

            # Recall distribution
            recalls = [r.recall for r in summary.post_results]
            axes[0, 1].hist(recalls, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(summary.macro_recall, color='red', linestyle='--',
                             label=f'Mean: {summary.macro_recall:.3f}')
            axes[0, 1].set_title('Recall Distribution')
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()

            # F1 Score distribution
            f1_scores = [r.f1_score for r in summary.post_results]
            axes[1, 0].hist(f1_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(summary.macro_f1, color='red', linestyle='--',
                             label=f'Mean: {summary.macro_f1:.3f}')
            axes[1, 0].set_title('F1 Score Distribution')
            axes[1, 0].set_xlabel('F1 Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()

            # Accuracy distribution
            accuracies = [r.accuracy for r in summary.post_results]
            axes[1, 1].hist(accuracies, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].axvline(summary.average_accuracy, color='red', linestyle='--',
                             label=f'Mean: {summary.average_accuracy:.3f}')
            axes[1, 1].set_title('Accuracy Distribution')
            axes[1, 1].set_xlabel('Accuracy')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(output_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting metrics distribution: {e}")

    def _plot_confusion_matrix_summary(self, summary: EvaluationSummary, output_dir: Path):
        """Plot aggregated confusion matrix"""
        try:
            # Aggregate confusion matrix
            total_tp = sum(r.true_positives for r in summary.post_results)
            total_fp = sum(r.false_positives for r in summary.post_results)
            total_tn = sum(r.true_negatives for r in summary.post_results)
            total_fn = sum(r.false_negatives for r in summary.post_results)

            cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted Negative', 'Predicted Positive'],
                       yticklabels=['Actual Negative', 'Actual Positive'])
            plt.title('Aggregated Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def _create_detailed_report(self, summary: EvaluationSummary, output_dir: Path):
        """Create detailed text report"""
        try:
            report_path = output_dir / 'evaluation_report.txt'

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("RAG RETRIEVAL EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Total Posts Evaluated: {summary.total_posts}\n\n")

                f.write("MACRO AVERAGES (average across posts):\n")
                f.write(f"  Precision: {summary.macro_precision:.4f}\n")
                f.write(f"  Recall: {summary.macro_recall:.4f}\n")
                f.write(f"  F1 Score: {summary.macro_f1:.4f}\n")
                f.write(f"  Accuracy: {summary.average_accuracy:.4f}\n\n")

                f.write("MICRO AVERAGES (aggregate across all predictions):\n")
                f.write(f"  Precision: {summary.micro_precision:.4f}\n")
                f.write(f"  Recall: {summary.micro_recall:.4f}\n")
                f.write(f"  F1 Score: {summary.micro_f1:.4f}\n\n")

                # Top performing posts
                f.write("TOP 10 PERFORMING POSTS (by F1 Score):\n")
                top_posts = sorted(summary.post_results, key=lambda x: x.f1_score, reverse=True)[:10]
                for i, post in enumerate(top_posts, 1):
                    f.write(f"{i:2d}. Post {post.post_id}: F1={post.f1_score:.3f}, "
                           f"P={post.precision:.3f}, R={post.recall:.3f}\n")

                f.write("\n")

                # Worst performing posts
                f.write("WORST 10 PERFORMING POSTS (by F1 Score):\n")
                worst_posts = sorted(summary.post_results, key=lambda x: x.f1_score)[:10]
                for i, post in enumerate(worst_posts, 1):
                    f.write(f"{i:2d}. Post {post.post_id}: F1={post.f1_score:.3f}, "
                           f"P={post.precision:.3f}, R={post.recall:.3f}\n")

            logger.info(f"Detailed report saved to {report_path}")

        except Exception as e:
            logger.error(f"Error creating detailed report: {e}")