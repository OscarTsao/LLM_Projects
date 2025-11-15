#!/usr/bin/env python3
"""
Test script for RAG evaluation system
"""
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.rag_evaluator import RAGEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evaluator():
    """Test the RAG evaluator with groundtruth data"""
    try:
        groundtruth_path = Path("Data/Groundtruth/criteria_evaluation.csv")

        if not groundtruth_path.exists():
            logger.error(f"Groundtruth file not found: {groundtruth_path}")
            return False

        # Initialize evaluator
        logger.info("Initializing RAG Evaluator...")
        evaluator = RAGEvaluator(groundtruth_path)

        logger.info(f"Loaded groundtruth for {len(evaluator.groundtruth_data)} posts")
        logger.info(f"Found {len(evaluator.criteria_mapping)} criteria columns")

        # Test getting groundtruth for a specific post
        if len(evaluator.groundtruth_data) > 0:
            first_post_data = evaluator.groundtruth_data.iloc[0]
            post_id = str(first_post_data.iloc[0])  # First column should be post_id

            logger.info(f"Testing with post ID: {post_id}")
            groundtruth_criteria = evaluator._get_post_groundtruth(post_id)
            logger.info(f"Found {len(groundtruth_criteria)} positive criteria for this post")

            if groundtruth_criteria:
                logger.info(f"Sample criteria: {groundtruth_criteria[:3]}...")

        logger.info("RAG Evaluator test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error testing evaluator: {e}")
        return False

if __name__ == "__main__":
    success = test_evaluator()
    if success:
        print("✅ Evaluation system test passed!")
    else:
        print("❌ Evaluation system test failed!")
        sys.exit(1)