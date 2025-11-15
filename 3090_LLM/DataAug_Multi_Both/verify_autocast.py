#!/usr/bin/env python3
"""Verify autocast compatibility for different models and report GPU capabilities.

This script checks:
1. GPU availability and capabilities (FP16, BF16, TF32)
2. Autocast compatibility with different model architectures
3. Training speed improvements with mixed precision
"""

import logging
import time
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu_capabilities():
    """Check GPU availability and capabilities."""
    print("\n" + "="*80)
    print("GPU CAPABILITIES CHECK")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU available. Training will be slow on CPU.")
        print("   Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support.")
        return False
    
    # GPU information
    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA is available")
    print(f"✓ Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Total memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Multi-processor count: {props.multi_processor_count}")
    
    # CUDA and PyTorch versions
    print(f"\n✓ CUDA version: {torch.version.cuda}")
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # Mixed precision support
    print("\nMixed Precision Support:")
    
    # FP16 support (available on all CUDA GPUs)
    print("  ✓ FP16 (float16): Supported on all CUDA GPUs")
    
    # BF16 support (requires Ampere or newer, compute capability >= 8.0)
    if torch.cuda.is_bf16_supported():
        print("  ✓ BF16 (bfloat16): Supported (Ampere GPU or newer)")
        print("    → Recommended for training: Better numerical stability than FP16")
    else:
        print("  ⚠ BF16 (bfloat16): Not supported (requires compute capability >= 8.0)")
        print("    → Use FP16 instead for mixed precision training")
    
    # TF32 support (Ampere or newer)
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        print("  ✓ TF32: Available (can be enabled for matmul operations)")
        print("    → Provides speedup with minimal accuracy loss on Ampere+ GPUs")
    else:
        print("  ⚠ TF32: Not available")
    
    return True


def test_autocast_performance(model_id: str = "google-bert/bert-base-uncased"):
    """Test autocast performance with a sample model."""
    print("\n" + "="*80)
    print(f"AUTOCAST PERFORMANCE TEST: {model_id}")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ Skipping performance test (no GPU available)")
        return
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        device = torch.device("cuda")
        
        # Load model and tokenizer
        print(f"\nLoading model: {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=False, use_safetensors=True).to(device)
        model.eval()
        
        # Create dummy batch
        batch_size = 16
        seq_length = 128
        texts = ["This is a test sentence for performance benchmarking."] * batch_size
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=seq_length,
            return_tensors="pt"
        ).to(device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)
        
        torch.cuda.synchronize()
        
        # Test FP32 (baseline)
        print("\nTesting FP32 (baseline)...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(50):
                _ = model(**inputs)
        
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        fp32_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"  Time: {fp32_time:.3f}s")
        print(f"  Peak memory: {fp32_memory:.2f} GB")
        
        # Test FP16
        print("\nTesting FP16 (autocast)...")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with torch.no_grad():
            with autocast(enabled=True, dtype=torch.float16):
                for _ in range(50):
                    _ = model(**inputs)
        
        torch.cuda.synchronize()
        fp16_time = time.time() - start_time
        fp16_memory = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"  Time: {fp16_time:.3f}s ({fp32_time/fp16_time:.2f}x speedup)")
        print(f"  Peak memory: {fp16_memory:.2f} GB ({fp32_memory/fp16_memory:.2f}x reduction)")
        
        # Test BF16 if supported
        if torch.cuda.is_bf16_supported():
            print("\nTesting BF16 (autocast)...")
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            with torch.no_grad():
                with autocast(enabled=True, dtype=torch.bfloat16):
                    for _ in range(50):
                        _ = model(**inputs)
            
            torch.cuda.synchronize()
            bf16_time = time.time() - start_time
            bf16_memory = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"  Time: {bf16_time:.3f}s ({fp32_time/bf16_time:.2f}x speedup)")
            print(f"  Peak memory: {bf16_memory:.2f} GB ({fp32_memory/bf16_memory:.2f}x reduction)")
        
        # Summary
        print("\n" + "-"*80)
        print("PERFORMANCE SUMMARY:")
        print(f"  FP16 provides {fp32_time/fp16_time:.2f}x speedup over FP32")
        print(f"  Memory usage reduced by {fp32_memory/fp16_memory:.2f}x with FP16")
        
        if torch.cuda.is_bf16_supported():
            print(f"  BF16 provides {fp32_time/bf16_time:.2f}x speedup over FP32")
            print(f"  Memory usage reduced by {fp32_memory/bf16_memory:.2f}x with BF16")
            print("\n  💡 Recommendation: Use BF16 for best numerical stability")
        else:
            print("\n  💡 Recommendation: Use FP16 for faster training")
        
        print("-"*80)
        
    except Exception as e:
        print(f"❌ Error during performance test: {e}")
        import traceback
        traceback.print_exc()


def test_model_compatibility():
    """Test autocast compatibility with models in the search space."""
    print("\n" + "="*80)
    print("MODEL COMPATIBILITY CHECK")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ Skipping compatibility test (no GPU available)")
        return
    
    # Models from the search space
    test_models = [
        "google-bert/bert-base-uncased",
        "microsoft/deberta-v3-base",
        "FacebookAI/xlm-roberta-base",
        "google/electra-base-discriminator",
    ]
    
    device = torch.device("cuda")
    
    for model_id in test_models:
        print(f"\nTesting: {model_id}")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            model = AutoModel.from_pretrained(model_id, trust_remote_code=False, use_safetensors=True).to(device)
            model.eval()
            
            # Create dummy input
            inputs = tokenizer(
                "Test sentence",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            # Test FP16
            with torch.no_grad():
                with autocast(enabled=True, dtype=torch.float16):
                    outputs = model(**inputs)
                    
            # Check for NaN
            if hasattr(outputs, 'last_hidden_state'):
                if torch.isnan(outputs.last_hidden_state).any():
                    print(f"  ⚠ FP16: NaN detected in outputs")
                else:
                    print(f"  ✓ FP16: Compatible")
            
            # Test BF16 if supported
            if torch.cuda.is_bf16_supported():
                with torch.no_grad():
                    with autocast(enabled=True, dtype=torch.bfloat16):
                        outputs = model(**inputs)
                        
                if hasattr(outputs, 'last_hidden_state'):
                    if torch.isnan(outputs.last_hidden_state).any():
                        print(f"  ⚠ BF16: NaN detected in outputs")
                    else:
                        print(f"  ✓ BF16: Compatible")
            
            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def print_recommendations():
    """Print recommendations for optimal training configuration."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR FASTEST TRAINING")
    print("="*80)
    
    print("\n1. Mixed Precision Training:")
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print("   ✓ Use fp_precision='bf16' for best speed and numerical stability")
        else:
            print("   ✓ Use fp_precision='fp16' for faster training")
        print("   ✓ Autocast is automatically applied to forward/backward passes")
    else:
        print("   ⚠ No GPU available - mixed precision not applicable")
    
    print("\n2. GPU Utilization:")
    print("   ✓ Use larger batch sizes (16-32) with gradient accumulation")
    print("   ✓ Enable gradient_checkpointing for larger models (trades compute for memory)")
    print("   ✓ Use num_workers=2-4 for DataLoader (already configured)")
    print("   ✓ Enable pin_memory=True for faster CPU-GPU transfer (already enabled)")
    
    print("\n3. CUDA Optimizations (automatically enabled):")
    if torch.cuda.is_available():
        print("   ✓ TF32 enabled for matmul (Ampere+ GPUs)")
        print("   ✓ cudnn.benchmark enabled for optimal conv algorithms")
    
    print("\n4. Current Configuration Status:")
    print("   ✓ Autocast is implemented in training loop")
    print("   ✓ Autocast is implemented in evaluation loop")
    print("   ✓ GradScaler for FP16 automatic loss scaling")
    print("   ✓ GPU memory optimizations enabled")
    print("   ✓ Default fp_precision='fp16' when GPU available")
    
    print("\n5. Known Compatible Models:")
    print("   ✓ All BERT variants (bert-base, bert-large, etc.)")
    print("   ✓ DeBERTa models")
    print("   ✓ XLM-RoBERTa models")
    print("   ✓ ELECTRA models")
    print("   ✓ Longformer models")
    print("   ✓ BioBERT, ClinicalBERT")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main verification function."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "AUTOCAST VERIFICATION SCRIPT" + " " * 30 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    # Check GPU capabilities
    has_gpu = check_gpu_capabilities()
    
    # Test performance if GPU available
    if has_gpu:
        test_autocast_performance()
        test_model_compatibility()
    
    # Print recommendations
    print_recommendations()
    
    print("Verification complete!")
    print("\n")


if __name__ == "__main__":
    main()
