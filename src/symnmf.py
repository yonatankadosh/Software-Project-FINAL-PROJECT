import sys
import numpy as np
import symnmf

np.random.seed(1234)

def print_matrix(matrix):
    for row in matrix:
        print(','.join(f"{val:.4f}" for val in row))

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 symnmf.py <goal> <input_file> [<k> if goal is symnmf]")
        return

    goal = sys.argv[1]
    input_file = sys.argv[2]

    try:
        X = np.loadtxt(input_file, delimiter=',')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if goal == "sym":
        A = symnmf.compute_similarity_matrix(X)
        print_matrix(A)

    elif goal == "ddg":
        A = symnmf.compute_similarity_matrix(X)
        D = symnmf.compute_diagonal_degree_matrix(A)
        print_matrix(D)

    elif goal == "norm":
        A = symnmf.compute_similarity_matrix(X)
        D = symnmf.compute_diagonal_degree_matrix(A)
        W = symnmf.compute_normalized_similarity_matrix(A, D)
        print_matrix(W)

    elif goal == "symnmf":
        if len(sys.argv) < 4:
            print("Usage for symnmf: python3 symnmf.py symnmf <input_file> <k>")
            return
        k = int(sys.argv[3])
        A = symnmf.compute_similarity_matrix(X)
        D = symnmf.compute_diagonal_degree_matrix(A)
        W = symnmf.compute_normalized_similarity_matrix(A, D)

        H_init = np.random.uniform(0, 2 * np.sqrt(1 / k), (X.shape[0], k))
        H_final = symnmf.symnmf(W, H_init, 300, 1e-4)
        print_matrix(H_final)

    else:
        print("Invalid goal. Choose from: sym, ddg, norm, symnmf.")

if __name__ == "__main__":
    main()