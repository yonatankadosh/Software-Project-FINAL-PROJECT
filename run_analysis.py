#!/usr/bin/env python3
"""
Main script to run the analysis.py program.
This script allows running analysis.py from the root directory.
"""

import sys
import subprocess
import os

def main():
    """Run analysis.py with the provided arguments."""
    if len(sys.argv) < 3:
        print("Usage: python3 run_analysis.py <k> <file_name.txt>")
        print("Example: python3 run_analysis.py 5 input_1.txt")
        return 1
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_script = os.path.join(script_dir, "src", "analysis.py")
    
    # Check if analysis.py exists
    if not os.path.exists(analysis_script):
        print(f"Error: {analysis_script} not found")
        return 1
    
    # Run analysis.py with the provided arguments
    try:
        result = subprocess.run([sys.executable, analysis_script] + sys.argv[1:], 
                              check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        return e.returncode
    except FileNotFoundError:
        print(f"Error: Python executable not found")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 