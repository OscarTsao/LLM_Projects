"""
Main script for comparing RAG retrieval results between negative and positive posts
"""
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.models.comparison_rag_pipeline import ComparisonRAGPipeline
from src.config.settings import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "comparison_rag_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_comparison_summary(statistics, negative_count, positive_count):
    """Print a comprehensive comparison summary"""
    print("\n" + "="*80)
    print("RAG RETRIEVAL COMPARISON: NEGATIVE vs POSITIVE POSTS")
    print("="*80)
    print(f"Sample Sizes: {negative_count} negative posts, {positive_count} positive posts")
    print("-"*80)

    for score_type, stats in statistics.items():
        print(f"\n{score_type.upper().replace('_', ' ')} ANALYSIS:")
        print(f"  Negative Posts:")
        print(f"    Average Score: {stats.negative_avg_score:.4f}")
        print(f"    95% CI: [{stats.negative_lower_bound:.4f}, {stats.negative_upper_bound:.4f}]")

        print(f"  Positive Posts:")
        print(f"    Average Score: {stats.positive_avg_score:.4f}")
        print(f"    95% CI: [{stats.positive_lower_bound:.4f}, {stats.positive_upper_bound:.4f}]")

        difference = stats.positive_avg_score - stats.negative_avg_score
        print(f"  Difference (Positive - Negative): {difference:+.4f}")

        print(f"  Statistical Test: {stats.test_type}")
        print(f"  Test Statistic: {stats.statistic:.4f}")
        print(f"  P-value: {stats.p_value:.6f}")
        print(f"  Significant at α=0.05: {'YES' if stats.significant else 'NO'}")

        if stats.significant:
            direction = "higher" if difference > 0 else "lower"
            print(f"  → Positive posts have significantly {direction} scores than negative posts")
        else:
            print(f"  → No significant difference between positive and negative posts")
        print("-"*50)

    print("\nINTERPREtation:")
    print("- Similarity Score: How well posts match DSM-5 criteria based on semantic similarity")
    print("- SpanBERT Score: How well posts match criteria based on contextual analysis")
    print("- Combined Score: Average of similarity and SpanBERT scores")
    print("- P-value < 0.05 indicates statistically significant difference")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare RAG retrieval results between negative and positive posts")
    parser.add_argument("--negative_posts", type=str,
                       default="Data/translated_posts.csv",
                       help="Path to negative posts CSV file")
    parser.add_argument("--positive_posts", type=str,
                       default="Data/translated_post_positive_random.csv",
                       help="Path to positive posts CSV file")
    parser.add_argument("--num_negative", type=int, default=1000,
                       help="Number of negative posts to evaluate")
    parser.add_argument("--num_positive", type=int, default=1000,
                       help="Number of positive posts to evaluate")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="Similarity threshold for FAISS search")
    parser.add_argument("--spanbert_threshold", type=float, default=0.5,
                       help="SpanBERT confidence threshold")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top criteria to retrieve")
    parser.add_argument("--save_index", action="store_true",
                       help="Save the built index")
    parser.add_argument("--load_index", type=str, default=None,
                       help="Path to load pre-built index")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (default: auto-generated)")

    args = parser.parse_args()

    try:
        # Resolve paths
        negative_posts_path = Path(args.negative_posts)
        positive_posts_path = Path(args.positive_posts)

        if not negative_posts_path.exists():
            logger.error(f"Negative posts file not found: {negative_posts_path}")
            return

        if not positive_posts_path.exists():
            logger.error(f"Positive posts file not found: {positive_posts_path}")
            return

        # Initialize comparison pipeline
        logger.info("Initializing Comparison RAG Pipeline...")
        comparison_pipeline = ComparisonRAGPipeline(
            negative_posts_path=negative_posts_path,
            positive_posts_path=positive_posts_path,
            criteria_path=CRITERIA_JSON,
            device=DEVICE,
            similarity_threshold=args.similarity_threshold,
            spanbert_threshold=args.spanbert_threshold,
            top_k=args.top_k
        )

        # Determine index path
        if args.load_index:
            index_path = Path(args.load_index)
        elif args.save_index:
            index_path = DATA_DIR / "indices" / "comparison_faiss_index"
            index_path.mkdir(parents=True, exist_ok=True)
        else:
            index_path = None

        # Run comparison
        logger.info("Starting comparison analysis...")
        negative_results, positive_results, statistics = comparison_pipeline.run_comparison(
            num_negative_posts=args.num_negative,
            num_positive_posts=args.num_positive,
            index_path=index_path
        )

        # Print summary
        print_comparison_summary(statistics, len(negative_results), len(positive_results))

        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = RESULTS_DIR / f"comparison_results_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_pipeline.save_comparison_results(
            negative_results, positive_results, statistics, output_path
        )

        logger.info(f"Detailed results saved to: {output_path}")
        print(f"\nDetailed results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()