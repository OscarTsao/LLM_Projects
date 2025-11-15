#!/usr/bin/env python3
"""
Demo script for making predictions with trained SpanBERT model
"""

# Standard library imports
import json
import os
import sys

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.spanbert_classifier import DSMClassificationTrainer

def load_sample_criteria():
    """Load sample criteria for demonstration"""
    with open('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json', 'r') as f:
        criteria_data = json.load(f)

    # Get first few criteria as examples
    sample_criteria = []
    for disorder in criteria_data[:3]:  # First 3 disorders
        for criterion in disorder['criteria'][:2]:  # First 2 criteria each
            sample_criteria.append({
                'disorder': disorder['diagnosis'],
                'criterion_id': criterion['id'],
                'text': criterion['text']
            })
    return sample_criteria

def main():
    print("=== SpanBERT DSM-5 Criteria Classification Demo ===")

    # Sample posts for testing
    sample_posts = [
        "I've been feeling really sad and hopeless for the past few weeks. I can't seem to enjoy anything anymore and I have no energy.",
        "I get extremely anxious in social situations and avoid talking to people. My heart races and I start sweating.",
        "I've been having trouble sleeping and concentrating. I feel worthless and have thoughts about death.",
        "I have sudden panic attacks where I feel like I can't breathe and my heart pounds really fast."
    ]

    # Load sample criteria
    sample_criteria = load_sample_criteria()

    # Initialize trainer
    trainer = DSMClassificationTrainer()

    print("\nMaking predictions on sample post-criterion pairs...")
    print("Note: This requires a trained model at './spanbert_dsm_model'")

    try:
        for i, post in enumerate(sample_posts[:2]):  # Limit to 2 posts for demo
            print(f"\n--- Post {i+1} ---")
            print(f"Post: {post[:100]}...")

            for j, criterion in enumerate(sample_criteria[:3]):  # Limit to 3 criteria
                result = trainer.predict(post, criterion['text'])
                match_prob = result['probability'][1]  # Probability of match (class 1)
                prediction = "MATCH" if result['prediction'] == 1 else "NO MATCH"

                print(f"\nCriterion: {criterion['disorder']} - {criterion['criterion_id']}")
                print(f"Prediction: {prediction} (confidence: {match_prob:.3f})")

    except FileNotFoundError:
        print("\nERROR: Trained model not found at './spanbert_dsm_model'")
        print("Please run 'python train_and_evaluate.py' first to train the model.")
    except Exception as e:
        print(f"\nERROR during prediction: {str(e)}")

if __name__ == "__main__":
    main()