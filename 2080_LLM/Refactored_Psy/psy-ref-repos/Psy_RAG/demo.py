"""
Demonstration script for the RAG system
"""
import logging
from pathlib import Path
import json

from src.models.rag_pipeline import RAGPipeline
from src.config.settings import *

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def demo_single_post():
    """Demonstrate single post analysis"""
    print("=" * 60)
    print("RAG SYSTEM DEMO - Single Post Analysis")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        posts_path=POSTS_CSV,
        criteria_path=CRITERIA_JSON,
        device=DEVICE,
        similarity_threshold=0.7,
        spanbert_threshold=0.5,
        top_k=5
    )
    
    # Build index
    print("Building FAISS index...")
    pipeline.build_index()
    
    # Sample posts for demonstration
    sample_posts = [
        "I feel very depressed and hopeless about my future. I can't sleep, I've lost my appetite, and I can't concentrate on anything. I feel worthless and think about suicide every day.",
        "I've been having panic attacks for weeks now. My heart races, I can't breathe, and I feel like I'm going to die. I'm constantly worried about everything and can't control my anxiety.",
        "I've been feeling really irritable and angry lately. I have outbursts over small things and feel restless all the time. My mood swings are extreme and I can't seem to control them.",
        "I've been having trouble sleeping and I feel tired all the time. I've lost interest in activities I used to enjoy and I feel empty inside. Sometimes I think about hurting myself.",
        "I've been experiencing hallucinations and hearing voices that aren't there. I feel paranoid and suspicious of everyone around me. I can't tell what's real anymore."
    ]
    
    for i, post in enumerate(sample_posts, 1):
        print(f"\n--- Sample Post {i} ---")
        print(f"Text: {post}")
        print("\nAnalyzing...")
        
        # Process post
        result = pipeline.process_post(post, post_id=i)
        
        # Display results
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Total matches: {result.total_matches}")
        
        if result.matched_criteria:
            print("\nMatched criteria:")
            for j, match in enumerate(result.matched_criteria[:3], 1):  # Show top 3
                print(f"\n{j}. {match.diagnosis} - {match.criteria_id}")
                print(f"   Similarity: {match.similarity_score:.3f}")
                print(f"   SpanBERT: {match.spanbert_score:.3f}")
                print(f"   Match: {'✓' if match.is_match else '✗'}")
                print(f"   Text: {match.criterion_text[:100]}...")
                
                if match.supporting_spans:
                    print("   Supporting evidence:")
                    for span in match.supporting_spans[:2]:  # Show top 2 spans
                        print(f"     - '{span.text}' (confidence: {span.confidence:.3f})")
        else:
            print("No criteria matches found.")
        
        print("-" * 40)


def demo_batch_processing():
    """Demonstrate batch processing"""
    print("\n" + "=" * 60)
    print("RAG SYSTEM DEMO - Batch Processing")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        posts_path=POSTS_CSV,
        criteria_path=CRITERIA_JSON,
        device=DEVICE,
        similarity_threshold=0.6,
        spanbert_threshold=0.4,
        top_k=3
    )
    
    # Build index
    print("Building FAISS index...")
    pipeline.build_index()
    
    # Load and process a small sample of posts
    print("Loading posts from dataset...")
    posts_df = pipeline.data_loader.load_posts()
    posts_data = pipeline.data_loader.preprocess_posts(posts_df)
    
    # Process first 10 posts
    sample_posts = posts_data[:10]
    print(f"Processing {len(sample_posts)} posts...")
    
    results = pipeline.process_posts_batch(sample_posts, batch_size=5)
    
    # Get statistics
    stats = pipeline.get_statistics(results)
    
    print(f"\nBatch Processing Results:")
    print(f"Total posts processed: {stats['total_posts']}")
    print(f"Total matches found: {stats['total_matches']}")
    print(f"Average matches per post: {stats['avg_matches_per_post']:.2f}")
    print(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    print(f"Posts with matches: {stats['posts_with_matches']}")
    
    print(f"\nTop diagnoses found:")
    for diagnosis, count in sorted(stats['diagnosis_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {diagnosis}: {count} matches")


def demo_performance():
    """Demonstrate performance optimizations"""
    print("\n" + "=" * 60)
    print("RAG SYSTEM DEMO - Performance Analysis")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        posts_path=POSTS_CSV,
        criteria_path=CRITERIA_JSON,
        device=DEVICE
    )
    
    # Build index
    print("Building FAISS index...")
    pipeline.build_index()
    
    # Test different batch sizes
    test_posts = [
        {"text": "I feel depressed and hopeless", "id": 1},
        {"text": "I have anxiety and panic attacks", "id": 2},
        {"text": "I feel irritable and angry", "id": 3},
        {"text": "I can't sleep and feel tired", "id": 4},
        {"text": "I hear voices and feel paranoid", "id": 5}
    ]
    
    print(f"\nTesting performance with {len(test_posts)} posts...")
    
    import time
    start_time = time.time()
    results = pipeline.process_posts_batch(test_posts, batch_size=2)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_post = total_time / len(test_posts)
    
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Average time per post: {avg_time_per_post:.3f}s")
    print(f"Posts per second: {len(test_posts) / total_time:.2f}")
    
    # Show memory usage if available
    try:
        from src.utils.performance_optimizer import PerformanceOptimizer
        optimizer = PerformanceOptimizer()
        memory_info = optimizer.get_memory_usage()
        
        print(f"\nMemory usage:")
        print(f"CPU memory: {memory_info.get('cpu_memory_percent', 'N/A')}%")
        if 'gpu_memory_allocated_gb' in memory_info:
            print(f"GPU memory allocated: {memory_info['gpu_memory_allocated_gb']:.2f} GB")
            print(f"GPU memory reserved: {memory_info['gpu_memory_reserved_gb']:.2f} GB")
    except Exception as e:
        print(f"Memory info not available: {e}")


def main():
    """Run all demonstrations"""
    print("RAG System for DSM-5 Criteria Matching")
    print("Using BGE-M3 embeddings and SpanBERT filtering")
    print(f"Device: {DEVICE}")
    print(f"Data: {POSTS_CSV.name} ({POSTS_CSV.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Criteria: {CRITERIA_JSON.name}")
    
    try:
        # Run demonstrations
        demo_single_post()
        demo_batch_processing()
        demo_performance()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo run the full system:")
        print("1. Build index: python main.py --mode build_index --save_index")
        print("2. Evaluate posts: python main.py --mode evaluate --num_posts 100")
        print("3. Single post: python main.py --mode single_post --post_text 'Your post here'")
        print("\nFor more options: python main.py --help")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")
        print("Please check the logs for more details.")


if __name__ == "__main__":
    main()
