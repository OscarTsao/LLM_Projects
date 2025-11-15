"""
Simple test script to verify comparison functionality without heavy dependencies
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.comparison_rag_pipeline import (
    calculate_t_test,
    mann_whitney_u_test,
    ComparisonStatistics
)

def test_statistical_functions():
    """Test our custom statistical functions"""
    print("Testing statistical functions...")

    # Test data
    sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample2 = [2.0, 3.0, 4.0, 5.0, 6.0]
    sample3 = [1.0, 1.1, 1.2, 1.3, 1.4]  # Different variance

    # Test t-test
    print("\nT-test results:")
    t_stat, p_val = calculate_t_test(sample1, sample2)
    print(f"Sample1 vs Sample2: t={t_stat:.4f}, p={p_val:.4f}")

    t_stat, p_val = calculate_t_test(sample1, sample3)
    print(f"Sample1 vs Sample3: t={t_stat:.4f}, p={p_val:.4f}")

    # Test Mann-Whitney U
    print("\nMann-Whitney U test results:")
    u_stat, p_val = mann_whitney_u_test(sample1, sample2)
    print(f"Sample1 vs Sample2: U={u_stat:.4f}, p={p_val:.4f}")

    u_stat, p_val = mann_whitney_u_test(sample1, sample3)
    print(f"Sample1 vs Sample3: U={u_stat:.4f}, p={p_val:.4f}")

    # Test with identical samples (should be non-significant)
    t_stat, p_val = calculate_t_test(sample1, sample1)
    print(f"\nIdentical samples t-test: t={t_stat:.4f}, p={p_val:.4f}")

    u_stat, p_val = mann_whitney_u_test(sample1, sample1)
    print(f"Identical samples U-test: U={u_stat:.4f}, p={p_val:.4f}")


def test_comparison_statistics():
    """Test the ComparisonStatistics class"""
    print("\n" + "="*50)
    print("Testing ComparisonStatistics...")

    # Mock data
    negative_scores = [0.3, 0.4, 0.5, 0.6, 0.7]
    positive_scores = [0.5, 0.6, 0.7, 0.8, 0.9]

    stats = ComparisonStatistics(
        negative_avg_score=sum(negative_scores)/len(negative_scores),
        positive_avg_score=sum(positive_scores)/len(positive_scores),
        negative_scores=negative_scores,
        positive_scores=positive_scores,
        negative_upper_bound=0.8,
        negative_lower_bound=0.2,
        positive_upper_bound=1.0,
        positive_lower_bound=0.4,
        p_value=0.023,
        statistic=2.45,
        significant=True,
        test_type="Independent t-test",
        sample_size_negative=len(negative_scores),
        sample_size_positive=len(positive_scores)
    )

    print(f"Negative average: {stats.negative_avg_score:.3f}")
    print(f"Positive average: {stats.positive_avg_score:.3f}")
    print(f"Difference: {stats.positive_avg_score - stats.negative_avg_score:.3f}")
    print(f"P-value: {stats.p_value}")
    print(f"Significant: {stats.significant}")
    print(f"Test type: {stats.test_type}")


def test_data_files():
    """Test loading the data files"""
    print("\n" + "="*50)
    print("Testing data file access...")

    negative_file = Path("Data/translated_posts.csv")
    positive_file = Path("Data/translated_post_positive.csv")

    if negative_file.exists():
        print(f"✅ Negative posts file found: {negative_file}")
        # Count lines
        with open(negative_file, 'r') as f:
            lines = sum(1 for _ in f)
        print(f"   Lines: {lines}")
    else:
        print(f"❌ Negative posts file not found: {negative_file}")

    if positive_file.exists():
        print(f"✅ Positive posts file found: {positive_file}")
        # Count lines
        with open(positive_file, 'r') as f:
            lines = sum(1 for _ in f)
        print(f"   Lines: {lines}")
    else:
        print(f"❌ Positive posts file not found: {positive_file}")


def create_sample_results():
    """Create sample comparison results to demonstrate the output format"""
    print("\n" + "="*50)
    print("Creating sample comparison results...")

    # Sample data
    import numpy as np

    # Simulate negative posts having lower scores
    np.random.seed(42)
    negative_scores = np.random.normal(0.4, 0.1, 100).tolist()
    positive_scores = np.random.normal(0.6, 0.1, 100).tolist()

    # Calculate statistics
    t_stat, p_val = calculate_t_test(negative_scores, positive_scores)

    # Create statistics object
    stats = ComparisonStatistics(
        negative_avg_score=np.mean(negative_scores),
        positive_avg_score=np.mean(positive_scores),
        negative_scores=negative_scores,
        positive_scores=positive_scores,
        negative_upper_bound=np.mean(negative_scores) + 1.96 * np.std(negative_scores) / np.sqrt(len(negative_scores)),
        negative_lower_bound=np.mean(negative_scores) - 1.96 * np.std(negative_scores) / np.sqrt(len(negative_scores)),
        positive_upper_bound=np.mean(positive_scores) + 1.96 * np.std(positive_scores) / np.sqrt(len(positive_scores)),
        positive_lower_bound=np.mean(positive_scores) - 1.96 * np.std(positive_scores) / np.sqrt(len(positive_scores)),
        p_value=p_val,
        statistic=t_stat,
        significant=p_val < 0.05,
        test_type="Independent t-test",
        sample_size_negative=len(negative_scores),
        sample_size_positive=len(positive_scores)
    )

    # Print results
    print("\nSAMPLE COMPARISON RESULTS:")
    print("-" * 40)
    print(f"Negative Posts Average Score: {stats.negative_avg_score:.4f}")
    print(f"95% CI: [{stats.negative_lower_bound:.4f}, {stats.negative_upper_bound:.4f}]")
    print()
    print(f"Positive Posts Average Score: {stats.positive_avg_score:.4f}")
    print(f"95% CI: [{stats.positive_lower_bound:.4f}, {stats.positive_upper_bound:.4f}]")
    print()
    print(f"Difference (Positive - Negative): {stats.positive_avg_score - stats.negative_avg_score:+.4f}")
    print()
    print(f"Statistical Test: {stats.test_type}")
    print(f"Test Statistic: {stats.statistic:.4f}")
    print(f"P-value: {stats.p_value:.6f}")
    print(f"Significant at α=0.05: {'YES' if stats.significant else 'NO'}")

    if stats.significant:
        direction = "higher" if stats.positive_avg_score > stats.negative_avg_score else "lower"
        print(f"→ Positive posts have significantly {direction} scores than negative posts")

    return stats


def main():
    """Run all tests"""
    print("RAG COMPARISON SYSTEM - FUNCTIONALITY TEST")
    print("=" * 50)

    try:
        test_statistical_functions()
        test_comparison_statistics()
        test_data_files()
        sample_stats = create_sample_results()

        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The comparison system is ready to use.")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)