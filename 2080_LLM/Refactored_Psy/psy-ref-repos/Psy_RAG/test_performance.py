#!/usr/bin/env python3
"""
Performance test script for the RAG system
"""
import time
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import DataLoader
from src.config.settings import *

def test_data_loading_performance():
    """Test data loading performance"""
    print("\n🔍 Testing Data Loading Performance...")
    
    data_loader = DataLoader(POSTS_CSV, CRITERIA_JSON)
    
    # Test posts loading
    start_time = time.time()
    posts_df = data_loader.load_posts()
    posts_load_time = time.time() - start_time
    
    # Test criteria loading
    start_time = time.time()
    criteria_data = data_loader.load_criteria()
    criteria_load_time = time.time() - start_time
    
    # Test preprocessing
    start_time = time.time()
    processed_posts = data_loader.preprocess_posts(posts_df.head(100))
    posts_preprocess_time = time.time() - start_time
    
    start_time = time.time()
    processed_criteria = data_loader.preprocess_criteria(criteria_data)
    criteria_preprocess_time = time.time() - start_time
    
    print(f"✅ Posts loaded: {len(posts_df)} in {posts_load_time:.3f}s")
    print(f"✅ Criteria loaded: {len(criteria_data)} disorders in {criteria_load_time:.3f}s")
    print(f"✅ Posts preprocessed: {len(processed_posts)} in {posts_preprocess_time:.3f}s")
    print(f"✅ Criteria preprocessed: {len(processed_criteria)} in {criteria_preprocess_time:.3f}s")
    
    return {
        'posts_count': len(posts_df),
        'criteria_count': len(criteria_data),
        'posts_load_time': posts_load_time,
        'criteria_load_time': criteria_load_time,
        'posts_preprocess_time': posts_preprocess_time,
        'criteria_preprocess_time': criteria_preprocess_time
    }

def test_system_info():
    """Display system information"""
    print("\n💻 System Information...")
    
    import platform
    import psutil
    
    print(f"🖥️  Platform: {platform.platform()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"💾 RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"🔧 CPU: {psutil.cpu_count()} cores")
    
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("   No CUDA devices available")
    except ImportError:
        print("⚠️  PyTorch not available - some features may not work")
    
    try:
        import faiss
        print(f"🔍 FAISS: Available with {faiss.get_num_gpus()} GPU(s)")
    except ImportError:
        print("⚠️  FAISS not available")

def test_configuration():
    """Test configuration settings"""
    print("\n⚙️  Configuration Settings...")
    
    print(f"📊 Device: {DEVICE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🎯 Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"🔍 Top-K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"🏎️  Mixed Precision: {MIXED_PRECISION}")
    print(f"⚡ Torch Compile: {TORCH_COMPILE}")
    
    print(f"\n📁 Data Paths:")
    print(f"   Posts: {POSTS_CSV.exists()} - {POSTS_CSV}")
    print(f"   Criteria: {CRITERIA_JSON.exists()} - {CRITERIA_JSON}")
    
    print(f"\n📂 Directories:")
    for name, path in [("Data", DATA_DIR), ("Results", RESULTS_DIR), ("Logs", LOGS_DIR)]:
        print(f"   {name}: {path.exists()} - {path}")

def main():
    """Main performance test function"""
    print("🚀 RAG System Performance Test")
    print("=" * 50)
    
    # Test system info
    test_system_info()
    
    # Test configuration
    test_configuration()
    
    # Test data loading
    try:
        performance_stats = test_data_loading_performance()
        
        print(f"\n📈 Performance Summary:")
        print(f"   Total data load time: {performance_stats['posts_load_time'] + performance_stats['criteria_load_time']:.3f}s")
        print(f"   Total preprocess time: {performance_stats['posts_preprocess_time'] + performance_stats['criteria_preprocess_time']:.3f}s")
        print(f"   Posts processing rate: {performance_stats['posts_count'] / performance_stats['posts_preprocess_time']:.0f} posts/sec")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False
    
    print(f"\n✅ All performance tests completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)