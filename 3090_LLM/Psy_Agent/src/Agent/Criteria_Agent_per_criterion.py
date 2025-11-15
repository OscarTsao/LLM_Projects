"""
This Agent evaluates each DSM-5 criterion individually for each patient post.
For N criteria and M patients, it performs N×M evaluations.
LLM model: TAIDE
Method: Baseline (per-criterion evaluation)
"""
import os
import sys
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
import csv

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach project root
project_root = os.path.join(current_dir, '..', '..')
project_root = os.path.abspath(project_root)

# Add to Python path
sys.path.insert(0, project_root)

from src.Utils.LLM_pipeline import TAIDE
from src.Dataloader.criteria_reader import extract_complete_named_tuples
from src.Dataloader.posts_loader import load_posts

def build_baseline_prompt(post: str, criterion: str, diagnosis: str, criterion_id: str) -> list:
    """Build prompt for evaluating a single criterion against a patient post"""
    sys_content = """You are a clinical psychology assistant that evaluates whether patient posts meet specific DSM-5 criteria.
    
    Your task is to determine if a patient's post indicates that they meet a specific DSM-5 criterion.
    
    Instructions:
    - Carefully read the patient's post and the DSM-5 criterion
    - Determine if the post contains evidence that the criterion is met
    - Answer with "1" if the criterion appears to be met based on the post
    - Answer with "0" if the criterion does not appear to be met or there is insufficient evidence
    - Be conservative in your evaluation - only answer "1" if there is clear evidence
    
    Format your response as: [0 or 1]"""

    user_content = f"""Patient's post: {post}

Diagnosis: {diagnosis}
Criterion ID: {criterion_id}
Criterion: {criterion}

Based on the patient's post, does this person meet the above DSM-5 criterion?
Answer with 1 (meets criterion) or 0 (does not meet criterion): """
    
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]

def run_baseline_per_criterion(model, posts, criterion_data, batch_size=4):
    """
    Run baseline evaluation for a single criterion across all posts
    
    Args:
        model: TAIDE model instance
        posts: List of patient posts
        criterion_data: Single criterion object with text, diagnosis, id
        batch_size: Batch size for processing
    
    Returns:
        results: List of model responses
        metadata: List of metadata for each evaluation
    """
    diagnosis = criterion_data.diagnosis
    criterion_id = criterion_data.criterion_id
    criterion_text = criterion_data.text
    
    print(f"Processing criterion {criterion_id} from {diagnosis}")
    print(f"Criterion: {criterion_text[:100]}...")
    
    all_prompts = []
    metadata = []
    
    # Build prompts for all posts with this criterion
    for post in posts:
        prompt = build_baseline_prompt(post, criterion_text, diagnosis, criterion_id)
        all_prompts.append(prompt)
        metadata.append({
            'diagnosis': diagnosis,
            'criterion_id': criterion_id,
            'criterion_text': criterion_text,
            'post': post
        })
    
    # Process in batches
    results = []
    try:
        print(f"Generating responses for {len(all_prompts)} post-criterion pairs...")
        results = model.generate_batch(
            all_prompts,
            batch_size=batch_size,
            max_new_tokens=10,  # Short responses (just 0 or 1)
            do_sample=False
        )
    except Exception as e:
        print(f"Batch processing failed, using individual processing: {e}")
        for prompt in tqdm(all_prompts, desc=f"Processing {criterion_id}"):
            result = model.generate(prompt, max_new_tokens=10, do_sample=False)
            results.append(result)
    
    return results, metadata

def save_per_criterion_results(all_results, all_metadata, output_dir="results"):
    """Save results in a comprehensive format"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed results file
    detailed_file = os.path.join(output_dir, f"per_criterion_detailed_{timestamp}.csv")
    
    # Summary results file (pivot table format)
    summary_file = os.path.join(output_dir, f"per_criterion_summary_{timestamp}.csv")
    
    print("Saving detailed results...")
    
    # Save detailed results
    with open(detailed_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "Post_ID", "Post_Text", "Diagnosis", "Criterion_ID", 
            "Criterion_Text", "Model_Response", "Extracted_Score"
        ])
        
        for i, (result, meta) in enumerate(zip(all_results, all_metadata)):
            try:
                # Extract binary score from model response
                extracted_score = extract_binary_score(result)
                
                writer.writerow([
                    i % len(set(m['post'] for m in all_metadata)),  # Post ID
                    meta['post'][:200] + "..." if len(meta['post']) > 200 else meta['post'],
                    meta['diagnosis'],
                    meta['criterion_id'],
                    meta['criterion_text'][:100] + "..." if len(meta['criterion_text']) > 100 else meta['criterion_text'],
                    result.strip(),
                    extracted_score
                ])
            except Exception as e:
                print(f"Error saving row {i}: {e}")
                continue
    
    print("Saving summary results...")
    
    # Create summary pivot table
    create_summary_table(all_results, all_metadata, summary_file)
    
    print(f"Detailed results saved to {detailed_file}")
    print(f"Summary results saved to {summary_file}")
    
    return detailed_file, summary_file

def extract_binary_score(response: str) -> int:
    """Extract binary score (0 or 1) from model response"""
    response = response.strip().lower()
    
    # Look for explicit 0 or 1
    if '1' in response and '0' not in response:
        return 1
    elif '0' in response and '1' not in response:
        return 0
    elif response.startswith('1') or '[1]' in response:
        return 1
    elif response.startswith('0') or '[0]' in response:
        return 0
    else:
        # Default to 0 if unclear
        return 0

def create_summary_table(all_results, all_metadata, output_file):
    """Create a pivot table format summary"""
    # Group data by post and criterion
    posts = list(set(m['post'] for m in all_metadata))
    criteria = list(set((m['diagnosis'], m['criterion_id']) for m in all_metadata))
    
    # Create pivot table
    pivot_data = {}
    for i, (result, meta) in enumerate(zip(all_results, all_metadata)):
        post = meta['post']
        criterion_key = (meta['diagnosis'], meta['criterion_id'])
        score = extract_binary_score(result)
        
        if post not in pivot_data:
            pivot_data[post] = {}
        pivot_data[post][criterion_key] = score
    
    # Write summary file
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        
        # Header
        header = ["Post_ID", "Post_Text"]
        for diagnosis, criterion_id in sorted(criteria):
            header.append(f"{diagnosis}_{criterion_id}")
        writer.writerow(header)
        
        # Data rows
        for post_id, post in enumerate(posts):
            row = [post_id, post[:100] + "..." if len(post) > 100 else post]
            for diagnosis, criterion_id in sorted(criteria):
                criterion_key = (diagnosis, criterion_id)
                score = pivot_data[post].get(criterion_key, -1)  # -1 for missing
                row.append(score)
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="DSM-5 Per-Criterion Evaluation Agent")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--num_posts", 
        type=int, 
        default=-1,
        help="Number of posts to process (default: -1 for all posts)"
    )
    parser.add_argument(
        "--num_criteria", 
        type=int, 
        default=-1,
        help="Number of criteria to process (default: -1 for all criteria)"
    )
    parser.add_argument(
        "--start_criterion", 
        type=int, 
        default=0,
        help="Starting criterion index (default: 0)"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("DSM-5 Per-Criterion Evaluation Agent")
    print("="*50)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of posts: {args.num_posts}")
    print(f"Number of criteria: {args.num_criteria}")
    print(f"Starting criterion: {args.start_criterion}")
    print("="*50)
    
    # Initialize model
    print("Initializing TAIDE model...")
    model = TAIDE(use_quantization=True)
    
    # Load data
    print("Loading patient posts...")
    posts = load_posts("Data/translated_posts.csv")
    if args.num_posts > 0:
        posts = posts[:args.num_posts]
    
    print("Loading DSM-5 criteria...")
    all_criteria = extract_complete_named_tuples("Data/DSM-5/DSM_Criteria_Array_Fixed.json")
    
    # Apply criteria filtering
    start_idx = args.start_criterion
    end_idx = start_idx + args.num_criteria if args.num_criteria > 0 else len(all_criteria)
    criteria_to_process = all_criteria[start_idx:end_idx]
    
    print(f"Processing {len(posts)} posts against {len(criteria_to_process)} criteria")
    print(f"Total evaluations: {len(posts)} × {len(criteria_to_process)} = {len(posts) * len(criteria_to_process)}")
    print("="*50)
    
    # Process each criterion
    all_results = []
    all_metadata = []
    
    for criterion_idx, criterion in enumerate(criteria_to_process):
        print(f"\nProcessing criterion {criterion_idx + 1}/{len(criteria_to_process)}")
        
        # Run baseline for this criterion across all posts
        results, metadata = run_baseline_per_criterion(
            model, posts, criterion, args.batch_size
        )
        
        all_results.extend(results)
        all_metadata.extend(metadata)
        
        # Print progress
        total_processed = (criterion_idx + 1) * len(posts)
        total_expected = len(criteria_to_process) * len(posts)
        print(f"Progress: {total_processed}/{total_expected} evaluations completed")
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Total evaluations completed: {len(all_results)}")
    print(f"Posts processed: {len(posts)}")
    print(f"Criteria processed: {len(criteria_to_process)}")
    
    detailed_file, summary_file = save_per_criterion_results(all_results, all_metadata)
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"Final GPU memory usage: {model.get_memory_usage()['allocated']:.1f}GB")
    
    print("Processing completed!")
    print(f"Check results in: {detailed_file} and {summary_file}")

if __name__ == "__main__":
    main()