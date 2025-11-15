"""
LLM Models and pipeline generator - Optimized Version
Models: TAIDE, Gemini Api, GPT oss
"""

import os
import time
import functools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

def monitor_performance(func):
    """Decorator to monitor execution time and memory usage"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        start_memory = self.get_memory_usage() if hasattr(self, 'get_memory_usage') else {}
        
        result = func(self, *args, **kwargs)
        
        end_time = time.time()
        end_memory = self.get_memory_usage() if hasattr(self, 'get_memory_usage') else {}
        
        print(f"{func.__name__} took {end_time - start_time:.2f}s")
        if start_memory and end_memory:
            print(f"Memory: {start_memory.get('allocated', 0):.1f}GB → {end_memory.get('allocated', 0):.1f}GB")
        
        return result
    return wrapper

class TAIDE:#"taide/Gemma-3-TAIDE-12b-Chat"
    def __init__(self, use_quantization: bool = True, model_name: str = "taide/Llama-3.1-TAIDE-LX-8B-Chat", hf_token: str = os.getenv("HF_TOKEN")):
        # Enable hardware optimizations first
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self.device = 0 if torch.cuda.is_available() else -1
        self.model_name = model_name
        
        # Setup quantization config
        self.quantization_config = None
        if use_quantization and torch.cuda.is_available():
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Fixed: was compute_type
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8
            )
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True, 
            token=hf_token  # Fixed: updated from deprecated use_auth_token
        )
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config,
            token=hf_token,  # Fixed: updated parameter name
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device == 0 else torch.float32,  # Better than float16
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        
        # Apply model optimizations AFTER loading
        # if hasattr(torch, 'compile'):
        #     print("Compiling model for faster inference...")
        #     # self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Better for generation
        
        # Configure model generation settings
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.use_cache = True
        self.model.config.use_cache = True
        self.model.eval()  # Set to evaluation mode
        
        # Create optimized pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # pad_to_multiple_of=8,  # Hardware optimization
        )
        
        print(f"Model loaded successfully!")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")

    def _format_messages_to_text(self, messages):
        """Convert chat messages to text format for text-generation pipeline"""
        if isinstance(messages, str):
            return messages
        
        if isinstance(messages, list) and len(messages) > 0:
            # Use the tokenizer's chat template if available
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
            except:
                # Fallback to manual formatting
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        text += f"System: {content}\n"
                    elif role == "user":
                        text += f"User: {content}\n"
                    elif role == "assistant":
                        text += f"Assistant: {content}\n"
                text += "Assistant:"
                return text
        
        return str(messages)

    @monitor_performance
    def generate(
        self,
        prompt,  # list[{"role","content"}] or str
        *,
        max_new_tokens: int = 256,  # Reduced for faster generation
        do_sample: bool = False,
        temperature: float = 0.1,  # Lower for more consistent results
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        return_full_text: bool = False,
    ) -> str:  # Fixed return type
        """Generate text with optimized parameters"""
        
        # Format input correctly
        formatted_input = self._format_messages_to_text(prompt)
        
        # Generate with memory optimization
        with torch.no_grad():
            outputs = self.pipeline(
                formatted_input,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                return_full_text=return_full_text,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = outputs[0]["generated_text"]
        
        # Clean output if needed
        if not return_full_text and formatted_input in generated_text:
            generated_text = generated_text.replace(formatted_input, "").strip()
        
        return generated_text

    @monitor_performance
    def generate_batch(
        self,
        prompts: list,
        *,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        return_full_text: bool = False,
    ) -> list:
        """Process multiple prompts efficiently in batches"""
        
        all_results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Format all prompts in batch
            formatted_batch = [self._format_messages_to_text(prompt) for prompt in batch]
            
            # Process batch with memory management
            with torch.no_grad():
                try:
                    batch_outputs = self.pipeline(
                        formatted_batch,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        return_full_text=return_full_text,
                        batch_size=len(formatted_batch),
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Extract and clean results
                    batch_results = []
                    for j, output in enumerate(batch_outputs):
                        generated = output["generated_text"]
                        if not return_full_text and formatted_batch[j] in generated:
                            generated = generated.replace(formatted_batch[j], "").strip()
                        batch_results.append(generated)
                    
                    all_results.extend(batch_results)
                    
                except Exception as e:
                    print(f"Batch processing failed, falling back to individual processing: {e}")
                    # Fallback to individual processing
                    for prompt in batch:
                        result = self.generate(prompt, 
                                             max_new_tokens=max_new_tokens,
                                             do_sample=do_sample,
                                             temperature=temperature,
                                             top_p=top_p,
                                             top_k=top_k,
                                             repetition_penalty=repetition_penalty,
                                             return_full_text=return_full_text)
                        all_results.append(result)
            
            # Clear cache periodically
            if i % (batch_size * 5) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results

    def get_memory_usage(self) -> dict:
        """Monitor GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {"allocated": 0, "cached": 0, "max_allocated": 0}

###############################################################################################################
# Not Finished yet
###############################################################################################################

class GeminiAPI:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        from google.generativeai import configure
        import google.generativeai as genai

        configure(api_key=api_key)

class GPT_Oss:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=device, do_sample=False)

        return self.pipeline, self.tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.75, repetition_penalty: float = 1.1):
        return self.pipeline(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)[0]['generated_text']