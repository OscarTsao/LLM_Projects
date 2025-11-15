"""
This Agent uses RAG method or Prompt based method to extract the DSM-5 criteria that matches the patient's post.
LLM model: TAIDE
RAG method: Hybrid Search
"""
import os
import sys
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach project root (/home/oscartsao/Developer/Psy_Agent)
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)

# Add to Python path
sys.path.insert(0, project_root)

from src.Utils.LLM_pipeline import TAIDE
from src.Utils.RAG import SparseRetriever, DenseRetriever, HybridRetriever
from src.Dataloader.criteria_reader import extract_complete_named_tuples
from src.Dataloader.posts_loader import load_posts

def create_retriever(retriever_type: str, criteria: list):
    """Factory function to create different types of retrievers"""
    print(f"Initializing {retriever_type.upper()} retriever...")
    
    if retriever_type == "sparse":
        return SparseRetriever(criteria)
    
    elif retriever_type == "dense":
        return DenseRetriever(
            criteria,
            model_name="BAAI/bge-base-en-v1.5",
            cache_dir="cache/dense_retriever"
        )
    
    elif retriever_type == "hybrid":
        return HybridRetriever(
            criteria,
            sparse_weight=0.4,
            dense_weight=0.6,
            model_name="BAAI/bge-base-en-v1.5",
            cache_dir="cache/hybrid_retriever"
        )
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

def get_retrieved_criteria(retriever, query: str, top_k: int, retriever_type: str):
    """Unified function to get retrieved criteria in consistent format"""
    
    if retriever_type == "sparse":
        # SparseRetriever returns [(index, score), ...]
        indices_scores = retriever.retrieve(query, top_k)
        # Convert to text format
        return [(retriever.criteria_texts[idx].text, score) for idx, score in indices_scores]
    
    elif retriever_type == "dense":
        # DenseRetriever has retrieve_with_texts method
        return retriever.retrieve_with_texts(query, top_k)
    
    elif retriever_type == "hybrid":
        # HybridRetriever already returns [(text, score), ...]
        return retriever.retrieve(query, top_k)
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

def build_baseline_prompt(question: str) -> str:
    sys_content = """You are a helpful assistant that helps to identify the DSM-5 criteria that matches the patient's post.
    You will be provided with a patient's post, and you need to identify the most relevant DSM-5 criteria from your knowledge.
    If you are unsure about the answer, you should say "I don't know".
    The following is the list of DSM-5 criteria you can refer to:
    [
  {
    "diagnosis": "Disruptive Mood Dysregulation Disorder",
    "criteria": [
      {
        "id": "A",
        "text": "Severe recurrent temper outbursts manifested verbally (e.g., verbal rages) and/or behaviorally (e.g., physical aggression toward people or property) that are grossly out of proportion in intensity or duration to the situation or provocation."
      },
      {
        "id": "B",
        "text": "The temper outbursts are inconsistent with developmental level."
      },
      {
        "id": "D",
        "text": "The mood between temper outbursts is persistently irritable or angry most of the day, nearly every day, and is observable by others (e.g., parents, teachers, peers)."
      },
      {
        "id": "G",
        "text": "The diagnosis should not be made for the first time before age 6 years or after age 18 years."
      },
      {
        "id": "I",
        "text": "There has never been a distinct period lasting more than 1 day during which the full symptom criteria, except duration, for a manic or hypomanic episode have been met. Note: Developmentally appropriate mood elevation, such as occurs in the context of a highly positive event or its anticipation, should not be considered as a symptom of mania or hypomania."
      },
      {
        "id": "J",
        "text": "The behaviors do not occur exclusively during an episode of major depressive disorder and are not better explained by another mental disorder (e.g., autism spectrum disorder, posttraumatic stress disorder, separation anxiety disorder, persistent depressive disorder [dysthymia])."
      },
      {
        "id": "K",
        "text": "The symptoms are not attributable to the physiological effects of a substance or to another medical or neurological condition."
      }
    ]
  """

    user_content = f"""Patient's post: {question}
    Please provide relevant disorder and DSM-5 criteria of each disorder that matches the patient's post.
    Generate the answer in the following format:
    1. Criteria: <relevant disorder and DSM-5 criteria of this disorder matched in the patient's post>
    3. Final Answer: <relevant DSM-5 criteria in csv format>
    """
    
    prompt = [
        {"role": "system", "content": f"{sys_content}"},
        {"role": "user", "content": f"{user_content}"},
    ]
    return prompt

def build_rag_prompt(question: str, retrieved_criteria: list) -> str:
    criteria_texts = "\n".join([f"{i+1}. {text} (Score: {score:.4f})" for i, (text, score) in enumerate(retrieved_criteria)])
    
    sys_content = """You are a helpful assistant that helps to identify the DSM-5 criteria of each disorder that matches the patient's post.
    You will be provided with a patient's post and a list of relevant DSM-5 criteria retrieved from a database.
    You need to identify relevant DSM-5 criteria from the provided list.
    If you are unsure about the answer, you should say "I don't know"."""

    user_content = f"""Patient's post: {question}
    Relevant DSM-5 criteria:
    {criteria_texts}

    Please provide relevant DSM-5 criteria that matches the patient's post. Organize the final answer by disorder but also list the matched criteria under the disorder. 
    Generate the answer in the following format:
    1. Criteria: <relevant disorder and DSM-5 criteria of this disorder matched in the patient's post>
    3. Final Answer: <relevant DSM-5 criteria in csv format>
    """
    
    prompt = [
        {"role": "system", "content": f"{sys_content}"},
        {"role": "user", "content": f"{user_content}"},
    ]
    return prompt

def run_baseline(model, posts, batch_size=4):
    """Run baseline method without RAG"""
    print("Running BASELINE method...")
    
    all_prompts = []
    print("Preparing baseline prompts...")
    for post in tqdm(posts, desc="Preparing baseline prompts"):
        prompt = build_baseline_prompt(post)
        all_prompts.append(prompt)
    
    # Process in batches
    print("Generating baseline responses...")
    try:
        results = model.generate_batch(
            all_prompts,
            batch_size=batch_size,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    except Exception as e:
        print(f"Batch processing failed, using individual processing: {e}")
        results = []
        for prompt in tqdm(all_prompts, desc="Individual baseline processing"):
            result = model.generate(prompt, max_new_tokens=256, temperature=0.1)
            results.append(result)
    
    return results, [None] * len(results)  # No retrieved criteria for baseline

def run_rag(model, posts, retriever, retriever_type, batch_size=4, top_k=20):
    """Run RAG method with retrieval - now supports all retriever types"""
    print(f"Running RAG method with {retriever_type.upper()} retriever...")
    
    all_prompts = []
    all_retrieved_criteria = []
    
    print("Preparing RAG prompts...")
    for post in tqdm(posts, desc="Preparing RAG prompts"):
        # Use unified retrieval function
        retrieved_criteria = get_retrieved_criteria(retriever, post, top_k, retriever_type)
        prompt = build_rag_prompt(post, retrieved_criteria)
        all_prompts.append(prompt)
        all_retrieved_criteria.append(retrieved_criteria)
    
    # Process in batches (remove unsupported parameters)
    print("Generating RAG responses...")
    try:
        results = model.generate_batch(
            all_prompts,
            batch_size=batch_size,
            max_new_tokens=512,
            do_sample=False
        )
    except Exception as e:
        print(f"Batch processing failed, using individual processing: {e}")
        results = []
        for prompt in tqdm(all_prompts, desc="Individual RAG processing"):
            result = model.generate(prompt, max_new_tokens=512, do_sample=False)
            results.append(result)
    
    return results, all_retrieved_criteria

def save_results(posts, results, retrieved_criteria_list, method_name, output_dir="results"):
    """Save results to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{method_name}_{timestamp}_results.csv")
    
    print("Saving results...")
    import csv
    
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        
        if method_name == "baseline":
            writer.writerow(["Post", "Baseline_Result"])
            for i, (post, result) in enumerate(zip(posts, results)):
                try:
                    writer.writerow([post, result])
                except Exception as e:
                    print(f"Error saving row {i}: {e}")
                    continue
        else:  # RAG method
            writer.writerow(["Post", "RAG_Result", "Retrieved_Criteria"])
            for i, (post, result, retrieved) in enumerate(zip(posts, results, retrieved_criteria_list)):
                try:
                    retrieved_str = str(retrieved)[:500] if retrieved else ""  # Truncate long criteria
                    writer.writerow([post, result, retrieved_str])
                except Exception as e:
                    print(f"Error saving row {i}: {e}")
                    continue
    
    print(f"Results saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="DSM-5 Criteria Agent")
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["baseline", "rag", "both"], 
        default="both",
        help="Choose method: 'baseline' for baseline only, 'rag' for RAG only, 'both' for both methods"
    )
    parser.add_argument(
        "--retriever", 
        type=str, 
        choices=["sparse", "dense", "hybrid"], 
        default="dense",
        help="Choose retriever: 'sparse' for BM25, 'dense' for BGE+FAISS, 'hybrid' for combined"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,  # Changed default for memory safety
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--num_posts", 
        type=int, 
        default=-1,
        help="Number of posts to process (default: -1, use -1 for all posts)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Number of criteria to retrieve for RAG method (default: 50)"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("DSM-5 Criteria Agent")
    print("="*50)
    print(f"Method: {args.method}")
    print(f"Retriever: {args.retriever}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of posts: {args.num_posts}")
    if args.method in ["rag", "both"]:
        print(f"Top-k retrieval: {args.top_k}")
    print("="*50)
    
    # Initialize model
    print("Initializing TAIDE model...")
    model = TAIDE(use_quantization=True)
    
    # Load data
    print("Loading data...")
    posts = load_posts("Data/translated_posts.csv")
    
    # Limit number of posts if specified
    if args.num_posts > 0:
        posts = posts[:args.num_posts]
    
    print(f"Processing {len(posts)} posts...")
    
    # Initialize RAG components if needed
    retriever = None
    if args.method in ["rag", "both"]:
        criteria = extract_complete_named_tuples("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
        retriever = create_retriever(args.retriever, criteria)
    
    # Run methods based on choice
    if args.method == "baseline":
        results, retrieved_criteria = run_baseline(model, posts, args.batch_size)
        save_results(posts, results, retrieved_criteria, "baseline")
        
    elif args.method == "rag":
        results, retrieved_criteria = run_rag(model, posts, retriever, args.retriever, args.batch_size, args.top_k)
        save_results(posts, results, retrieved_criteria, f"rag_{args.retriever}")
        
    elif args.method == "both":
        # Run baseline
        print("\n" + "="*50)
        baseline_results, _ = run_baseline(model, posts, args.batch_size)
        save_results(posts, baseline_results, [None] * len(baseline_results), "baseline")
        
        # Run RAG
        print("\n" + "="*50)
        rag_results, retrieved_criteria = run_rag(model, posts, retriever, args.retriever, args.batch_size, args.top_k)
        save_results(posts, rag_results, retrieved_criteria, f"rag_{args.retriever}")
        
        # Compare results
        print("\n" + "="*50)
        print("COMPARISON SUMMARY:")
        print(f"Baseline method processed {len(baseline_results)} posts")
        print(f"RAG ({args.retriever}) method processed {len(rag_results)} posts")
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"Final GPU memory usage: {model.get_memory_usage()['allocated']:.1f}GB")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()