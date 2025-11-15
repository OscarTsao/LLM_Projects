"""
Performance optimization utilities for RTX 3090
"""
import logging
from typing import Optional, Dict, Any
import gc
import psutil
import os

# Try to import torch, fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Performance optimization utilities for RTX 3090"""
    
    def __init__(self):
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        if TORCH_AVAILABLE:
            self.optimize_for_rtx3090()
        else:
            logger.warning("PyTorch not available, skipping GPU optimizations")
    
    def optimize_for_rtx3090(self):
        """Apply RTX 3090 specific optimizations"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, skipping optimizations")
                return
                
            if self.device == "cuda":
                # Enable TensorFloat-32 for better performance
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable cuDNN benchmark for consistent input sizes
                torch.backends.cudnn.benchmark = True
                
                # Enable cuDNN deterministic mode for reproducibility
                torch.backends.cudnn.deterministic = False
                
                # Set memory fraction to avoid OOM
                torch.cuda.set_per_process_memory_fraction(0.9)
                
                # Enable memory efficient attention if available
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    torch.backends.cuda.enable_flash_sdp(True)
                
                logger.info("RTX 3090 optimizations applied")
            else:
                logger.warning("CUDA not available, skipping GPU optimizations")
                
        except Exception as e:
            logger.error(f"Error applying RTX 3090 optimizations: {e}")
    
    def optimize_model_for_inference(self, model) -> object:
        """Optimize model for inference"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, returning model unchanged")
                return model
                
            if self.device == "cuda":
                # Move to GPU
                model = model.to(self.device)
                
                # Use half precision for better performance
                model = model.half()
                
                # Enable evaluation mode
                model.eval()
                
                # Compile model if PyTorch 2.0+
                if hasattr(torch, 'compile'):
                    try:
                        model = torch.compile(model, mode="max-autotune")
                        logger.info("Model compiled with torch.compile")
                    except Exception as e:
                        logger.warning(f"Could not compile model: {e}")
                
                logger.info("Model optimized for inference")
            
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return model
    
    def clear_memory(self):
        """Clear GPU and CPU memory"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage"""
        try:
            memory_info = {
                "cpu_percent": psutil.cpu_percent(),
                "cpu_memory_percent": psutil.virtual_memory().percent,
                "cpu_memory_available_gb": psutil.virtual_memory().available / (1024**3)
            }
            
            if self.device == "cuda":
                memory_info.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
                })
            
            return memory_info
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}
    
    def optimize_batch_size(self, model: torch.nn.Module, input_shape: tuple) -> int:
        """Find optimal batch size for the model"""
        try:
            if self.device != "cuda":
                return 1
            
            # Start with a small batch size
            batch_size = 1
            max_batch_size = 64
            
            while batch_size <= max_batch_size:
                try:
                    # Create dummy input
                    dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                    
                    # Test forward pass
                    with torch.no_grad():
                        _ = model(dummy_input)
                    
                    # Clear memory
                    del dummy_input
                    self.clear_memory()
                    
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # Reduce batch size and break
                        batch_size = max(1, batch_size // 2)
                        break
                    else:
                        raise
            
            logger.info(f"Optimal batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return 1
    
    def enable_mixed_precision(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enable mixed precision training/inference"""
        try:
            if self.device == "cuda":
                # Use automatic mixed precision
                model = model.half()
                logger.info("Mixed precision enabled")
            
            return model
            
        except Exception as e:
            logger.error(f"Error enabling mixed precision: {e}")
            return model
    
    def profile_model(self, model: torch.nn.Module, input_shape: tuple, num_iterations: int = 10):
        """Profile model performance"""
        try:
            if self.device != "cuda":
                logger.warning("Profiling only available on CUDA")
                return
            
            # Warm up
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            for _ in range(5):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            # Profile
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / num_iterations
            
            logger.info(f"Model profiling - Average inference time: {elapsed_time:.2f}ms")
            
            # Clean up
            del dummy_input
            self.clear_memory()
            
        except Exception as e:
            logger.error(f"Error profiling model: {e}")
    
    def set_environment_variables(self):
        """Set environment variables for optimal performance"""
        try:
            # Set CUDA environment variables
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            os.environ["CUDA_CACHE_DISABLE"] = "0"
            os.environ["CUDA_CACHE_MAXSIZE"] = "268435456"  # 256MB
            
            # Set PyTorch environment variables
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
            os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
            
            logger.info("Environment variables set for optimal performance")
            
        except Exception as e:
            logger.error(f"Error setting environment variables: {e}")
