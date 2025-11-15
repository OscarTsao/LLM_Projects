"""
Enhanced RAG pipeline for comparing negative and positive posts with statistical analysis
"""
# Standard library
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
from .rag_pipeline import RAGPipeline, RAGResult, CriteriaMatch
from ..utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


def calculate_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Simple two-sample t-test implementation
    Returns (t_statistic, p_value)
    """
    import math

    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

    # Pooled standard error
    pooled_se = math.sqrt(var1/n1 + var2/n2)

    if pooled_se == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / pooled_se

    # Degrees of freedom (Welch's formula)
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # Approximate p-value using normal distribution for large samples
    if df > 30:
        p_value = 2 * (1 - normal_cdf(abs(t_stat)))
    else:
        # For small samples, use a conservative approach
        p_value = 2 * (1 - normal_cdf(abs(t_stat) * 0.9))  # Slightly more conservative

    return t_stat, p_value


def normal_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution"""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mann_whitney_u_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Simple Mann-Whitney U test implementation
    Returns (U_statistic, p_value)
    """
    import math

    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and rank all observations
    combined = [(x, 1) for x in sample1] + [(x, 2) for x in sample2]
    combined.sort(key=lambda x: x[0])

    # Assign ranks
    ranks = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Assign average rank for ties
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j

    # Calculate U statistics
    R1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1

    U = min(U1, U2)

    # Approximate p-value for large samples
    if n1 > 8 and n2 > 8:
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        if sigma > 0:
            z = (U - mu) / sigma
            p_value = 2 * (1 - normal_cdf(abs(z)))
        else:
            p_value = 1.0
    else:
        # Conservative p-value for small samples
        p_value = 0.05 if U < n1 * n2 * 0.25 else 0.5

    return U, p_value


@dataclass
class ComparisonStatistics:
    """Statistics for comparing negative and positive posts"""
    negative_avg_score: float
    positive_avg_score: float
    negative_scores: List[float]
    positive_scores: List[float]
    negative_upper_bound: float
    negative_lower_bound: float
    positive_upper_bound: float
    positive_lower_bound: float
    p_value: float
    statistic: float
    significant: bool
    test_type: str
    sample_size_negative: int
    sample_size_positive: int


@dataclass
class PostEvaluationResult:
    """Result for a single post evaluation"""
    post_id: int
    post_text: str
    post_type: str  # 'negative' or 'positive'
    avg_similarity_score: float
    avg_spanbert_score: float
    combined_score: float
    total_matches: int
    processing_time: float
    matched_criteria: List[CriteriaMatch]


class ComparisonRAGPipeline:
    """Enhanced RAG pipeline for comparing negative vs positive posts"""

    def __init__(
        self,
        negative_posts_path: Path,
        positive_posts_path: Path,
        criteria_path: Path,
        embedding_model_name: str = "BAAI/bge-m3",
        spanbert_model_name: str = "SpanBERT/spanbert-base-cased",
        device: str = "cuda",
        similarity_threshold: float = 0.7,
        spanbert_threshold: float = 0.5,
        top_k: int = 10
    ):
        self.negative_posts_path = negative_posts_path
        self.positive_posts_path = positive_posts_path
        self.criteria_path = criteria_path

        # Initialize base RAG pipeline for negative posts
        self.rag_pipeline = RAGPipeline(
            posts_path=negative_posts_path,
            criteria_path=criteria_path,
            embedding_model_name=embedding_model_name,
            spanbert_model_name=spanbert_model_name,
            device=device,
            similarity_threshold=similarity_threshold,
            spanbert_threshold=spanbert_threshold,
            top_k=top_k
        )

        # Create separate data loader for positive posts
        self.positive_data_loader = DataLoader(positive_posts_path, criteria_path)

        logger.info("Comparison RAG Pipeline initialized")

    def build_or_load_index(self, index_path: Optional[Path] = None):
        """Build or load the FAISS index"""
        if index_path and index_path.exists():
            logger.info(f"Loading existing index from {index_path}")
            self.rag_pipeline.load_index(index_path)
        else:
            logger.info("Building new FAISS index...")
            save_path = index_path if index_path else None
            self.rag_pipeline.build_index(save_path)

    def load_negative_posts(self, num_posts: Optional[int] = None) -> List[Dict]:
        """Load and preprocess negative posts"""
        try:
            posts_df = self.rag_pipeline.data_loader.load_posts()
            posts_data = self.rag_pipeline.data_loader.preprocess_posts(posts_df)

            if num_posts:
                posts_data = posts_data[:num_posts]

            logger.info(f"Loaded {len(posts_data)} negative posts")
            return posts_data

        except Exception as e:
            logger.error(f"Error loading negative posts: {e}")
            return []

    def load_positive_posts(self, num_posts: Optional[int] = None) -> List[Dict]:
        """Load and preprocess positive posts"""
        try:
            posts_df = self.positive_data_loader.load_posts()
            posts_data = self.positive_data_loader.preprocess_posts(posts_df)

            if num_posts:
                posts_data = posts_data[:num_posts]

            logger.info(f"Loaded {len(posts_data)} positive posts")
            return posts_data

        except Exception as e:
            logger.error(f"Error loading positive posts: {e}")
            return []

    def evaluate_posts_with_scores(
        self,
        posts_data: List[Dict],
        post_type: str
    ) -> List[PostEvaluationResult]:
        """
        Evaluate posts and extract comprehensive scores

        Args:
            posts_data: List of post dictionaries
            post_type: 'negative' or 'positive'

        Returns:
            List of PostEvaluationResult objects
        """
        results = []

        try:
            logger.info(f"Evaluating {len(posts_data)} {post_type} posts...")

            for post in tqdm(posts_data, desc=f"Processing {post_type} posts"):
                # Process post through RAG pipeline
                rag_result = self.rag_pipeline.process_post(
                    post['text'],
                    post.get('id', 0)
                )

                # Calculate average scores
                if rag_result.matched_criteria:
                    avg_similarity = np.mean([m.similarity_score for m in rag_result.matched_criteria])
                    avg_spanbert = np.mean([m.spanbert_score for m in rag_result.matched_criteria])
                    combined_score = (avg_similarity + avg_spanbert) / 2
                else:
                    avg_similarity = 0.0
                    avg_spanbert = 0.0
                    combined_score = 0.0

                # Create evaluation result
                eval_result = PostEvaluationResult(
                    post_id=rag_result.post_id,
                    post_text=rag_result.post_text,
                    post_type=post_type,
                    avg_similarity_score=avg_similarity,
                    avg_spanbert_score=avg_spanbert,
                    combined_score=combined_score,
                    total_matches=rag_result.total_matches,
                    processing_time=rag_result.processing_time,
                    matched_criteria=rag_result.matched_criteria
                )

                results.append(eval_result)

            logger.info(f"Completed evaluation of {len(results)} {post_type} posts")
            return results

        except Exception as e:
            logger.error(f"Error evaluating {post_type} posts: {e}")
            return results

    def calculate_statistics(
        self,
        negative_results: List[PostEvaluationResult],
        positive_results: List[PostEvaluationResult],
        score_type: str = "combined_score"
    ) -> ComparisonStatistics:
        """
        Calculate comprehensive statistics and significance tests

        Args:
            negative_results: Results from negative posts
            positive_results: Results from positive posts
            score_type: Type of score to compare ('combined_score', 'avg_similarity_score', 'avg_spanbert_score')

        Returns:
            ComparisonStatistics object
        """
        try:
            # Extract scores based on type
            if score_type == "combined_score":
                negative_scores = [r.combined_score for r in negative_results]
                positive_scores = [r.combined_score for r in positive_results]
            elif score_type == "avg_similarity_score":
                negative_scores = [r.avg_similarity_score for r in negative_results]
                positive_scores = [r.avg_similarity_score for r in positive_results]
            elif score_type == "avg_spanbert_score":
                negative_scores = [r.avg_spanbert_score for r in negative_results]
                positive_scores = [r.avg_spanbert_score for r in positive_results]
            else:
                raise ValueError(f"Invalid score_type: {score_type}")

            # Calculate basic statistics
            negative_avg = np.mean(negative_scores) if negative_scores else 0.0
            positive_avg = np.mean(positive_scores) if positive_scores else 0.0

            # Calculate confidence intervals (95%)
            if len(negative_scores) > 1:
                negative_std = np.std(negative_scores, ddof=1)
                negative_se = negative_std / np.sqrt(len(negative_scores))
                negative_margin = 1.96 * negative_se
                negative_lower = negative_avg - negative_margin
                negative_upper = negative_avg + negative_margin
            else:
                negative_lower = negative_upper = negative_avg

            if len(positive_scores) > 1:
                positive_std = np.std(positive_scores, ddof=1)
                positive_se = positive_std / np.sqrt(len(positive_scores))
                positive_margin = 1.96 * positive_se
                positive_lower = positive_avg - positive_margin
                positive_upper = positive_avg + positive_margin
            else:
                positive_lower = positive_upper = positive_avg

            # Perform statistical significance test
            if len(negative_scores) > 1 and len(positive_scores) > 1:
                # Simple normality check based on sample size and variance
                n1, n2 = len(negative_scores), len(positive_scores)
                var1, var2 = np.var(negative_scores), np.var(positive_scores)

                # Use t-test for reasonably sized samples with similar variances
                if n1 >= 5 and n2 >= 5 and var2 > 0 and (var1 / var2 < 4 and var2 / var1 < 4):
                    statistic, p_value = calculate_t_test(negative_scores, positive_scores)
                    test_type = "Independent t-test"
                else:
                    # Use Mann-Whitney U test for small samples or unequal variances
                    statistic, p_value = mann_whitney_u_test(negative_scores, positive_scores)
                    test_type = "Mann-Whitney U test"
            else:
                statistic = 0.0
                p_value = 1.0
                test_type = "Insufficient data for testing"

            # Determine significance (alpha = 0.05)
            significant = p_value < 0.05

            return ComparisonStatistics(
                negative_avg_score=negative_avg,
                positive_avg_score=positive_avg,
                negative_scores=negative_scores,
                positive_scores=positive_scores,
                negative_upper_bound=negative_upper,
                negative_lower_bound=negative_lower,
                positive_upper_bound=positive_upper,
                positive_lower_bound=positive_lower,
                p_value=p_value,
                statistic=statistic,
                significant=significant,
                test_type=test_type,
                sample_size_negative=len(negative_scores),
                sample_size_positive=len(positive_scores)
            )

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            raise

    def run_comparison(
        self,
        num_negative_posts: Optional[int] = None,
        num_positive_posts: Optional[int] = None,
        index_path: Optional[Path] = None
    ) -> Tuple[List[PostEvaluationResult], List[PostEvaluationResult], Dict[str, ComparisonStatistics]]:
        """
        Run complete comparison between negative and positive posts

        Args:
            num_negative_posts: Number of negative posts to evaluate
            num_positive_posts: Number of positive posts to evaluate
            index_path: Path to save/load FAISS index

        Returns:
            Tuple of (negative_results, positive_results, statistics_dict)
        """
        try:
            logger.info("Starting comparison between negative and positive posts")

            # Build or load index
            self.build_or_load_index(index_path)

            # Load posts
            negative_posts = self.load_negative_posts(num_negative_posts)
            positive_posts = self.load_positive_posts(num_positive_posts)

            if not negative_posts or not positive_posts:
                raise ValueError("Unable to load posts data")

            # Evaluate posts
            negative_results = self.evaluate_posts_with_scores(negative_posts, "negative")
            positive_results = self.evaluate_posts_with_scores(positive_posts, "positive")

            # Calculate statistics for different score types
            statistics = {}
            for score_type in ["combined_score", "avg_similarity_score", "avg_spanbert_score"]:
                stats_obj = self.calculate_statistics(
                    negative_results, positive_results, score_type
                )
                statistics[score_type] = stats_obj

                logger.info(f"{score_type} Statistics:")
                logger.info(f"  Negative avg: {stats_obj.negative_avg_score:.4f} "
                          f"[{stats_obj.negative_lower_bound:.4f}, {stats_obj.negative_upper_bound:.4f}]")
                logger.info(f"  Positive avg: {stats_obj.positive_avg_score:.4f} "
                          f"[{stats_obj.positive_lower_bound:.4f}, {stats_obj.positive_upper_bound:.4f}]")
                logger.info(f"  {stats_obj.test_type}: p-value = {stats_obj.p_value:.6f}, "
                          f"significant = {stats_obj.significant}")

            logger.info("Comparison completed successfully")
            return negative_results, positive_results, statistics

        except Exception as e:
            logger.error(f"Error running comparison: {e}")
            raise

    def save_comparison_results(
        self,
        negative_results: List[PostEvaluationResult],
        positive_results: List[PostEvaluationResult],
        statistics: Dict[str, ComparisonStatistics],
        filepath: Path
    ):
        """Save comparison results to JSON file"""
        try:
            # Prepare data for serialization
            data = {
                "timestamp": datetime.now().isoformat(),
                "negative_posts_count": len(negative_results),
                "positive_posts_count": len(positive_results),
                "statistics": {
                    score_type: asdict(stats_obj)
                    for score_type, stats_obj in statistics.items()
                },
                "negative_results": [asdict(result) for result in negative_results],
                "positive_results": [asdict(result) for result in positive_results]
            }

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Comparison results saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")
            raise