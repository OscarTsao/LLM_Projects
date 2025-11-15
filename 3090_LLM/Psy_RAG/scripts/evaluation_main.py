"""
Main script for evaluating RAG retrieval accuracy using groundtruth data
"""
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.models.rag_pipeline import RAGPipeline
from src.models.rag_evaluator import RAGEvaluator
from src.config.settings import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "rag_evaluation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_evaluation_summary(summary):
    """Print comprehensive evaluation summary"""
    print("\n" + "="*80)
    print("RAG RETRIEVAL EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Posts Evaluated: {summary.total_posts}")
    print("-"*80)

    print("\nMACRO AVERAGES (average across posts):")
    print(f"  Precision: {summary.macro_precision:.4f}")
    print(f"  Recall:    {summary.macro_recall:.4f}")
    print(f"  F1 Score:  {summary.macro_f1:.4f}")
    print(f"  Accuracy:  {summary.average_accuracy:.4f}")

    print("\nMICRO AVERAGES (aggregate across all predictions):")
    print(f"  Precision: {summary.micro_precision:.4f}")
    print(f"  Recall:    {summary.micro_recall:.4f}")
    print(f"  F1 Score:  {summary.micro_f1:.4f}")

    # Performance breakdown
    high_f1_posts = len([r for r in summary.post_results if r.f1_score >= 0.8])
    medium_f1_posts = len([r for r in summary.post_results if 0.5 <= r.f1_score < 0.8])
    low_f1_posts = len([r for r in summary.post_results if r.f1_score < 0.5])

    print(f"\nPERFORMANCE BREAKDOWN:")
    print(f"  High F1 (≥0.8):     {high_f1_posts:4d} posts ({100*high_f1_posts/summary.total_posts:.1f}%)")
    print(f"  Medium F1 (0.5-0.8): {medium_f1_posts:4d} posts ({100*medium_f1_posts/summary.total_posts:.1f}%)")
    print(f"  Low F1 (<0.5):       {low_f1_posts:4d} posts ({100*low_f1_posts/summary.total_posts:.1f}%)")

    print("\nINTERPRETATION:")
    print("- Precision: How many retrieved criteria were actually relevant")
    print("- Recall: How many relevant criteria were successfully retrieved")
    print("- F1 Score: Harmonic mean of precision and recall")
    print("- Macro averages: Equal weight to each post")
    print("- Micro averages: Weight by total number of predictions")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval accuracy using groundtruth data")
    parser.add_argument("--posts", type=str,
                       default="Data/translated_posts.csv",
                       help="Path to posts CSV file")
    parser.add_argument("--groundtruth", type=str,
                       default="Data/Groundtruth/criteria_evaluation.csv",
                       help="Path to groundtruth CSV file")
    parser.add_argument("--num_posts", type=int, default=100,
                       help="Number of posts to evaluate (None for all)")
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
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: auto-generated)")
    parser.add_argument("--generate_report", action="store_true",
                       help="Generate detailed evaluation report with visualizations")

    args = parser.parse_args()

    try:
        # Resolve paths
        posts_path = Path(args.posts)
        groundtruth_path = Path(args.groundtruth)

        if not posts_path.exists():
            logger.error(f"Posts file not found: {posts_path}")
            return

        if not groundtruth_path.exists():
            logger.error(f"Groundtruth file not found: {groundtruth_path}")
            return

        # Initialize RAG pipeline
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            posts_path=posts_path,
            criteria_path=CRITERIA_JSON,
            device=DEVICE,
            similarity_threshold=args.similarity_threshold,
            spanbert_threshold=args.spanbert_threshold,
            top_k=args.top_k
        )

        # Determine index path
        if args.load_index:
            index_path = Path(args.load_index)
            logger.info(f"Loading pre-built index from {index_path}")
            rag_pipeline.load_index(index_path)
        else:
            logger.info("Building new index...")
            if args.save_index:
                index_path = DATA_DIR / "indices" / "evaluation_faiss_index"
                index_path.mkdir(parents=True, exist_ok=True)
                rag_pipeline.build_index(save_path=index_path)
            else:
                rag_pipeline.build_index()

        # Initialize evaluator
        logger.info("Initializing RAG Evaluator...")
        evaluator = RAGEvaluator(groundtruth_path)

        # Run RAG on posts
        logger.info("Running RAG retrieval...")
        rag_results = rag_pipeline.evaluate_posts(num_posts=args.num_posts)

        if not rag_results:
            logger.error("No RAG results obtained")
            return

        # Evaluate results
        logger.info("Evaluating retrieval accuracy...")
        evaluation_summary = evaluator.evaluate_results(rag_results)

        # Print summary
        print_evaluation_summary(evaluation_summary)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = RESULTS_DIR / f"evaluation_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        evaluator.save_evaluation_results(evaluation_summary, results_file)
        logger.info(f"Detailed results saved to: {results_file}")

        # Generate report if requested
        if args.generate_report:
            logger.info("Generating evaluation report...")
            evaluator.generate_evaluation_report(evaluation_summary, output_dir)
            print(f"\nDetailed evaluation report with visualizations saved to: {output_dir}")

        # Save RAG results for reference
        rag_results_file = output_dir / "rag_results.json"
        rag_pipeline.save_results(rag_results, rag_results_file)

        print(f"\nAll results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()