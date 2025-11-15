"""
Main script for running the RAG system for DSM-5 criteria matching
"""
import logging
import argparse
from pathlib import Path
import json
from datetime import datetime

from src.models.rag_pipeline import RAGPipeline
from src.config.settings import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / "rag_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG System for DSM-5 Criteria Matching")
    parser.add_argument("--mode", choices=["build_index", "evaluate", "single_post"], 
                       default="evaluate", help="Mode to run")
    parser.add_argument("--num_posts", type=int, default=100, 
                       help="Number of posts to evaluate")
    parser.add_argument("--post_text", type=str, 
                       help="Single post text to evaluate (for single_post mode)")
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
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG pipeline
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            posts_path=POSTS_CSV,
            criteria_path=CRITERIA_JSON,
            device=DEVICE,
            similarity_threshold=args.similarity_threshold,
            spanbert_threshold=args.spanbert_threshold,
            top_k=args.top_k
        )
        
        if args.mode == "build_index":
            # Build and optionally save index
            logger.info("Building FAISS index...")
            save_path = DATA_DIR / "indices" / "faiss_index" if args.save_index else None
            rag_pipeline.build_index(save_path)
            logger.info("Index building completed")
            
        elif args.mode == "evaluate":
            # Load index if specified
            if args.load_index:
                logger.info(f"Loading index from {args.load_index}")
                rag_pipeline.load_index(Path(args.load_index))
            else:
                # Build index if not loading
                logger.info("Building index...")
                rag_pipeline.build_index()
            
            # Evaluate posts
            logger.info(f"Evaluating {args.num_posts} posts...")
            results = rag_pipeline.evaluate_posts(args.num_posts)
            
            # Get statistics
            stats = rag_pipeline.get_statistics(results)
            logger.info(f"Evaluation completed. Statistics: {json.dumps(stats, indent=2)}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = RESULTS_DIR / f"rag_results_{timestamp}.json"
            rag_pipeline.save_results(results, results_file)
            logger.info(f"Results saved to {results_file}")
            
        elif args.mode == "single_post":
            if not args.post_text:
                logger.error("post_text is required for single_post mode")
                return
            
            # Load index if specified
            if args.load_index:
                logger.info(f"Loading index from {args.load_index}")
                rag_pipeline.load_index(Path(args.load_index))
            else:
                # Build index if not loading
                logger.info("Building index...")
                rag_pipeline.build_index()
            
            # Process single post
            logger.info("Processing single post...")
            result = rag_pipeline.process_post(args.post_text)
            
            # Print results
            print(f"\nPost: {result.post_text}")
            print(f"Total matches: {result.total_matches}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print("\nMatched criteria:")
            
            for i, match in enumerate(result.matched_criteria, 1):
                print(f"\n{i}. {match.diagnosis} - {match.criteria_id}")
                print(f"   Similarity: {match.similarity_score:.3f}")
                print(f"   SpanBERT: {match.spanbert_score:.3f}")
                print(f"   Match: {match.is_match}")
                print(f"   Text: {match.criterion_text}")
                
                if match.supporting_spans:
                    print("   Supporting spans:")
                    for span in match.supporting_spans:
                        print(f"     - '{span.text}' (confidence: {span.confidence:.3f})")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
