#!/usr/bin/env python3

import subprocess
import sys
import os

def test_analysis_basic():
    """Test basic functionality of analysis.py"""
    print("Testing analysis.py basic functionality...")
    
    # Test with a simple case
    try:
        result = subprocess.run(
            ["python3", "src/analysis.py", "2", "input_1.txt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"✅ Basic test passed")
            print(f"Output: {output}")
            
            # Check if output has expected format
            lines = output.split('\n')
            if len(lines) == 2:
                if lines[0].startswith("nmf:") and lines[1].startswith("kmeans:"):
                    print("✅ Output format is correct")
                    return True
                else:
                    print("❌ Output format is incorrect")
                    return False
            else:
                print("❌ Expected 2 lines of output")
                return False
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_analysis_with_test_data():
    """Test analysis.py with test data files"""
    print("\nTesting analysis.py with test data...")
    
    test_cases = [
        ("2", "tests/tests/input_1.txt"),
        ("7", "tests/tests/input_2.txt"),
        ("15", "tests/tests/input_3.txt"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for k, input_file in test_cases:
        try:
            result = subprocess.run(
                ["python3", "src/analysis.py", k, input_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                lines = output.split('\n')
                if len(lines) == 2 and lines[0].startswith("nmf:") and lines[1].startswith("kmeans:"):
                    print(f"✅ k={k}, file={input_file}: {output}")
                    passed += 1
                else:
                    print(f"❌ k={k}, file={input_file}: Invalid output format")
            else:
                print(f"❌ k={k}, file={input_file}: Failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ k={k}, file={input_file}: Timed out")
        except Exception as e:
            print(f"❌ k={k}, file={input_file}: Exception {e}")
    
    print(f"\nTest results: {passed}/{total} passed")
    return passed == total

def test_symnmf_basic():
    """Test symnmf.py basic functionality"""
    print("\nTesting symnmf.py basic functionality...")
    
    try:
        result = subprocess.run(
            ["python3", "src/symnmf.py", "sym", "input_1.txt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ SymNMF sym test passed")
            return True
        else:
            print(f"❌ SymNMF sym test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ SymNMF test failed: {e}")
        return False

def test_kmeans_basic():
    """Test kmeans.py basic functionality"""
    print("\nTesting kmeans.py basic functionality...")
    
    try:
        # Create a simple input file for testing
        with open("test_input.txt", "w") as f:
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")
        
        result = subprocess.run(
            ["python3", "src/kmeans.py", "2", "3"],
            input="1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n",
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ K-means test passed")
            return True
        else:
            print(f"❌ K-means test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ K-means test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_input.txt"):
            os.remove("test_input.txt")

def main():
    print("=== Testing Your Project Implementation ===\n")
    
    tests = [
        test_analysis_basic,
        test_analysis_with_test_data,
        test_symnmf_basic,
        test_kmeans_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Summary ===")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your implementation is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 