#!/usr/bin/env python3
"""
Test script to verify data loading functionality
"""

import pandas as pd
import json

def test_data_loading():
    """Test data loading without PyTorch dependencies"""

    print("Testing data loading...")

    # Load DSM-5 criteria
    with open('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json', 'r', encoding='utf-8') as f:
        criteria_data = json.load(f)

    print(f"✓ Loaded {len(criteria_data)} disorders")

    # Create criteria lookup
    criteria_lookup = {}
    total_criteria = 0
    for disorder in criteria_data:
        disorder_name = disorder['diagnosis']
        for criterion in disorder['criteria']:
            key = f"{disorder_name} - {criterion['id']}"
            criteria_lookup[key] = criterion['text']
            total_criteria += 1

    print(f"✓ Created lookup for {total_criteria} criteria")

    # Load ground truth
    gt_df = pd.read_csv('Data/Groundtruth/criteria_evaluation.csv')
    print(f"✓ Loaded {len(gt_df)} posts from ground truth file")

    # Test data preparation
    training_examples = []

    print("Preparing training examples...")
    for idx, row in gt_df.iterrows():
        if idx >= 10:  # Limit to first 10 for testing
            break

        post_text = row['post_id']
        criteria_columns = [col for col in gt_df.columns if col != 'post_id']

        for criterion_col in criteria_columns:
            if criterion_col in criteria_lookup:
                criterion_text = criteria_lookup[criterion_col]
                label = int(row[criterion_col])

                training_examples.append({
                    'post': post_text,
                    'criterion': criterion_text,
                    'label': label,
                    'criterion_name': criterion_col
                })

    df = pd.DataFrame(training_examples)
    print(f"✓ Created {len(df)} training examples (from first 10 posts)")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Show sample data
    print("\nSample training example:")
    sample = df.iloc[0]
    print(f"Post (first 100 chars): {sample['post'][:100]}...")
    print(f"Criterion: {sample['criterion'][:100]}...")
    print(f"Label: {sample['label']}")

    return df

if __name__ == "__main__":
    try:
        df = test_data_loading()
        print("\n🎉 Data loading test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during data loading: {str(e)}")
        import traceback
        traceback.print_exc()