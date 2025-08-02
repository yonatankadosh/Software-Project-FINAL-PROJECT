# SymNMF and K-means Analysis Project

This project implements and compares SymNMF (Symmetric Non-negative Matrix Factorization) and K-means clustering algorithms using silhouette scores.

## Project Structure

```
FINAL-PROJECT/
├── src/                    # Source code files
│   ├── analysis.py         # Main analysis script
│   ├── symnmf.py          # SymNMF implementation
│   ├── kmeans.py          # K-means implementation
│   ├── symnmf.c           # C implementation of SymNMF
│   ├── symnmf.h           # Header file for SymNMF
│   ├── symnmfmodule.c     # Python C extension module
│   ├── setup.py           # Build configuration
│   └── Makefile           # Build instructions
├── tests/                  # All test files and data
│   ├── simple_test.py     # Simple test script
│   ├── test_my_project.sh # Comprehensive test script
│   ├── run_tests.sh       # Original test suite
│   ├── tests/             # SymNMF test data
│   └── kmeans_tests/      # K-means test data
├── build/                  # Build artifacts
│   ├── symnmf             # Compiled SymNMF executable
│   ├── symnmf.o           # Object files
│   └── symnmf.cpython-*.so # Python extension modules
├── data/                   # Data files
├── run_analysis.py         # Main entry point script
└── README.md              # This file
```

## Usage

### Running Analysis

From the root directory, you can run the analysis in two ways:

1. **Using the main script (recommended):**
   ```bash
   python3 run_analysis.py <k> <file_name.txt>
   ```

2. **Directly from src directory:**
   ```bash
   python3 src/analysis.py <k> <file_name.txt>
   ```

### Examples

```bash
# Run analysis with k=5 on input_1.txt
python3 run_analysis.py 5 input_1.txt

# Run analysis with k=3 on a test file
python3 run_analysis.py 3 tests/tests/input_1.txt
```

### Expected Output

The program outputs silhouette scores for both algorithms:
```
nmf: 0.8856
kmeans: 0.8856
```

## Testing

### Running Tests

1. **Simple test (recommended):**
   ```bash
   python3 tests/simple_test.py
   ```

2. **Comprehensive test:**
   ```bash
   cd tests
   ./test_my_project.sh
   ```

3. **Original test suite:**
   ```bash
   cd tests
   ./run_tests.sh
   ```

## Implementation Details

### Convergence Parameters
- **ε (epsilon)**: 1e-4 (0.0001)
- **max_iter**: 300
- **Random seed**: 1234 (for reproducible results)

### Algorithms
- **SymNMF**: Symmetric Non-negative Matrix Factorization
- **K-means**: Standard K-means clustering
- **Silhouette Score**: Manual implementation

### Cluster Assignment
- **SymNMF**: Assigns each point to the cluster corresponding to the maximum value in the H matrix row
- **K-means**: Assigns each point to the nearest centroid

## Building

To build the C extensions:

```bash
cd src
make
```

## Requirements

- Python 3.x
- NumPy
- C compiler (for building extensions)

## Notes

- All random operations use seed 1234 for reproducible results
- The implementation avoids code duplication by reusing existing SymNMF and K-means modules
- Test data is included in the tests/ directory
- Build artifacts are stored in the build/ directory 