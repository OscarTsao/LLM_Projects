"""
Test runner script for the RAG system
"""
import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests with pytest"""
    try:
        # Add src to Python path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Run pytest with coverage
        cmd = [
            "python", "-m", "pytest", 
            "tests/", 
            "-v", 
            "--cov=src", 
            "--cov-report=html", 
            "--cov-report=term-missing",
            "--tb=short"
        ]
        
        print("Running tests...")
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("Coverage report generated in htmlcov/index.html")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
