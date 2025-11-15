#!/usr/bin/env python3
"""
Quick verification script showing the NSP format implementation.
Run this to see the new input format in action.
"""

from transformers import AutoTokenizer
from datasets import Dataset as HFDataset
from src.dataaug_multi_both.data.dataset import RedSM5Dataset


def main():
    print("=" * 80)
    print("NSP Format Verification")
    print("=" * 80)
    
    # Sample post
    sample_data = [{
        "post_id": "1",
        "post_text": "I can't sleep at night and feel so tired during the day",
        "criteria_labels": [1, 0, 0, 1, 0, 0, 0, 0, 0],
        "evidence_spans": []
    }]
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hf_dataset = HFDataset.from_list(sample_data)
    
    print("\nOriginal Post:")
    print(f"  '{sample_data[0]['post_text']}'")
    print(f"\nGround Truth Labels: {sample_data[0]['criteria_labels']}")
    print("  (1 = SLEEP_ISSUES, 4 = FATIGUE)")
    
    # Binary Pairs Format
    print("\n" + "-" * 80)
    print("BINARY PAIRS FORMAT (9 examples per post)")
    print("-" * 80)
    
    dataset_binary = RedSM5Dataset(hf_dataset, tokenizer, input_format="binary_pairs", max_length=64)
    
    for i in range(3):  # Show first 3 examples
        example = dataset_binary[i]
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
        
        # Find SEP positions
        sep_pos = [j for j, t in enumerate(tokens) if t == "[SEP]"]
        
        # Extract segments
        post_tokens = tokens[1:sep_pos[0]]  # Skip [CLS]
        criterion_tokens = tokens[sep_pos[0]+1:sep_pos[1]]
        
        print(f"\nExample {i} (Criterion {i}):")
        print(f"  Post: {' '.join(post_tokens)}")
        print(f"  Criterion: {' '.join(criterion_tokens)}")
        print(f"  Label: {example['criteria_labels'][i].item()}")
        print(f"  Token Type IDs: {example['token_type_ids'][:len(post_tokens)+len(criterion_tokens)+3].tolist()}")
    
    # Multi-Label Format
    print("\n" + "-" * 80)
    print("MULTI-LABEL FORMAT (1 example per post)")
    print("-" * 80)
    
    dataset_multi = RedSM5Dataset(hf_dataset, tokenizer, input_format="multi_label", max_length=128)
    
    example = dataset_multi[0]
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    
    # Find SEP positions
    sep_pos = [j for j, t in enumerate(tokens) if t == "[SEP]"]
    
    # Extract segments
    post_tokens = tokens[1:sep_pos[0]]
    criteria_tokens = tokens[sep_pos[0]+1:sep_pos[1]]
    
    print(f"\nPost: {' '.join(post_tokens)}")
    print(f"\nAll Criteria (concatenated):")
    print(f"  {' '.join(criteria_tokens[:50])}...")
    print(f"\nLabels: {example['criteria_labels'].tolist()}")
    print(f"\nToken Type IDs (first 30): {example['token_type_ids'][:30].tolist()}")
    print("  (0 = post segment, 1 = criteria segment)")
    
    print("\n" + "=" * 80)
    print("✓ NSP Format Successfully Implemented!")
    print("=" * 80)


if __name__ == "__main__":
    main()
