#!/usr/bin/env python3
"""
Test script for the RAG-enhanced BERT classification system
"""

import os
import pandas as pd
from rag_retrieval import DSMCriteriaRetriever


def test_retrieval_system():
    """Test the RAG retrieval system with sample posts"""
    print("Testing RAG Retrieval System")
    print("=" * 50)

    # Initialize retriever
    retriever = DSMCriteriaRetriever()

    # Load criteria and build/load index
    try:
        retriever.load_index()
        print("✓ Loaded existing retrieval index")
    except FileNotFoundError:
        print("Building new retrieval index...")
        retriever.load_criteria('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json')
        retriever.build_index()
        print("✓ Built new retrieval index")

    # Test with sample posts
    test_posts = [
        "I feel very sad and hopeless every day. I can't sleep and have lost my appetite.",
        "I have been having panic attacks with heart racing and difficulty breathing.",
        "I keep checking the door multiple times to make sure it's locked.",
        "I feel extremely energetic and don't need much sleep. I have so many great ideas!",
        "I can't concentrate on anything and feel restless all the time."
    ]

    print(f"\nTesting retrieval with {len(test_posts)} sample posts:")
    print("-" * 60)

    for i, post in enumerate(test_posts, 1):
        print(f"\nPost {i}: '{post[:50]}...'")
        retrieved = retriever.retrieve(post, top_k=3, threshold=0.3)

        if retrieved:
            print(f"Retrieved {len(retrieved)} relevant criteria:")
            for j, result in enumerate(retrieved, 1):
                print(f"  {j}. [{result['score']:.3f}] {result['disorder']} - {result['criterion_id']}")
                print(f"     {result['text'][:80]}...")
        else:
            print("  No criteria retrieved above threshold")

    return retriever


def test_data_loading():
    """Test data loading and preparation"""
    print("\n\nTesting Data Loading")
    print("=" * 50)

    # Check if data files exist
    files_to_check = [
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False

    # Load and check ground truth data
    try:
        gt_df = pd.read_csv('Data/Groundtruth/criteria_evaluation.csv')
        print(f"✓ Loaded ground truth data: {len(gt_df)} posts")
        print(f"  Columns: {list(gt_df.columns[:5])}... (showing first 5)")

        # Check label distribution
        criteria_columns = [col for col in gt_df.columns if col != 'post_id']
        total_positive_labels = 0
        total_labels = 0

        for col in criteria_columns:
            positive = gt_df[col].sum()
            total = len(gt_df)
            total_positive_labels += positive
            total_labels += total

        print(f"  Total criteria: {len(criteria_columns)}")
        print(f"  Positive labels: {total_positive_labels}/{total_labels} ({total_positive_labels/total_labels*100:.1f}%)")

        return True

    except Exception as e:
        print(f"✗ Error loading ground truth data: {e}")
        return False


def test_rag_integration():
    """Test RAG integration with classification"""
    print("\n\nTesting RAG Integration")
    print("=" * 50)

    try:
        from rag_spanbert_classifier import RAGDSMClassificationTrainer

        # Initialize trainer
        trainer = RAGDSMClassificationTrainer(
            retrieval_top_k=5,
            retrieval_threshold=0.3
        )
        print("✓ Initialized RAG classification trainer")

        # Setup retriever
        trainer.setup_retriever('Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json')
        print("✓ Setup RAG retriever")

        # Test prediction for a single post
        test_post = "I feel very sad and hopeless every day. I can't sleep and have lost my appetite."

        # Get retrieved criteria (without full model prediction)
        retrieved_criteria = trainer.retriever.get_retrieved_criteria_for_classification(
            test_post, top_k=5, threshold=0.3
        )

        print(f"✓ Retrieved {len(retrieved_criteria)} criteria for test post")
        print("  Retrieved criteria:")
        for criterion_key, criterion_text in retrieved_criteria.items():
            print(f"    - {criterion_key}: {criterion_text[:60]}...")

        return True

    except Exception as e:
        print(f"✗ Error in RAG integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("RAG-Enhanced BERT Classification System Tests")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    # Test 1: Retrieval system
    try:
        test_retrieval_system()
        tests_passed += 1
        print("✓ RAG retrieval test PASSED")
    except Exception as e:
        print(f"✗ RAG retrieval test FAILED: {e}")

    # Test 2: Data loading
    try:
        if test_data_loading():
            tests_passed += 1
            print("✓ Data loading test PASSED")
        else:
            print("✗ Data loading test FAILED")
    except Exception as e:
        print(f"✗ Data loading test FAILED: {e}")

    # Test 3: RAG integration
    try:
        if test_rag_integration():
            tests_passed += 1
            print("✓ RAG integration test PASSED")
        else:
            print("✗ RAG integration test FAILED")
    except Exception as e:
        print(f"✗ RAG integration test FAILED: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"TESTS SUMMARY: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! RAG system is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)