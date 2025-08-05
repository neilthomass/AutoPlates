#!/usr/bin/env python3
"""
Test runner for the License Plate Detector project.

This script runs all tests and provides a summary of results.
"""

import unittest
import sys
import os
import subprocess
from pathlib import Path


def run_unit_tests():
    """Run all unit tests."""
    print("🧪 Running Unit Tests")
    print("=" * 40)
    
    # Add paths for imports
    tests_dir = Path(__file__).parent / "tests"
    src_dir = Path(__file__).parent.parent / "src"
    database_dir = Path(__file__).parent.parent / "database"
    
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(database_dir))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = str(tests_dir)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful(), len(result.failures), len(result.errors)


def run_setup_test():
    """Run the setup verification test."""
    print("\n🔧 Running Setup Verification")
    print("=" * 40)
    
    try:
        # Run the setup test
        setup_test_path = Path(__file__).parent / "tests" / "test_setup.py"
        result = subprocess.run([sys.executable, str(setup_test_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Setup verification passed")
            return True
        else:
            print("Setup verification failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running setup test: {e}")
        return False


def run_examples():
    """Run example scripts."""
    print("\nRunning Examples")
    print("=" * 40)
    
    examples_dir = Path(__file__).parent / "examples"
    examples = list(examples_dir.glob("*.py"))
    
    results = []
    
    for example in examples:
        print(f"\nRunning {example.name}...")
        try:
            result = subprocess.run([sys.executable, str(example)], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"{example.name} completed successfully")
                results.append(True)
            else:
                print(f"{example.name} failed")
                print(result.stdout)
                print(result.stderr)
                results.append(False)
                
        except subprocess.TimeoutExpired:
            print(f"{example.name} timed out")
            results.append(False)
        except Exception as e:
            print(f"Error running {example.name}: {e}")
            results.append(False)
    
    return results


def main():
    """Run all tests and examples."""
    print("License Plate Detector - Test Suite")
    print("=" * 50)
    
    # Run unit tests
    unit_success, unit_failures, unit_errors = run_unit_tests()
    
    # Run setup test
    setup_success = run_setup_test()
    
    # Run examples
    example_results = run_examples()
    example_success = all(example_results)
    example_count = len(example_results)
    example_passed = sum(example_results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
    if not unit_success:
        print(f"  - Failures: {unit_failures}")
        print(f"  - Errors: {unit_errors}")
    
    print(f"Setup Verification: {'PASSED' if setup_success else 'FAILED'}")
    print(f"Examples: {example_passed}/{example_count} passed")
    
    # Overall result
    overall_success = unit_success and setup_success and example_success
    
    if overall_success:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main()) 