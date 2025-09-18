#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <math.h>


/* Helper function to compute the convergence difference between two iterations */
double compute_convergence_diff(double **H, double **WH, double **denominator, int n, int k, double beta) {
    double diff = 0.0;
    int i;
    int j;
    int l;
    double h_old_il = 0.0;
    double h_old_jl = 0.0;
    double delta = 0.0;
    double dot_new = 0.0;
    double dot_old = 0.0;
    double h_new_il = 0.0;
    double h_new_jl = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            delta = 0.0;
            dot_new = 0.0;
            dot_old = 0.0;
            for (l = 0; l < k; l++) {
                h_new_il = H[i][l];
                h_new_jl = H[j][l];
                dot_new += h_new_il * h_new_jl;

                h_old_il = H[i][l] / pow(WH[i][l] / denominator[i][l], beta);
                h_old_jl = H[j][l] / pow(WH[j][l] / denominator[j][l], beta);
                dot_old += h_old_il * h_old_jl;
            }
            delta = dot_new - dot_old;
            diff += (i == j) ? delta * delta : 2 * delta * delta;
        }
    }
    return diff;
}

/**
 * Computes the normalized similarity matrix W = D^(-1/2) * A * D^(-1/2)
 * 
 * @param A The similarity matrix (n x n)
 * @param D The diagonal degree matrix (n x n)
 * @param n The number of data points
 * @return The normalized similarity matrix W (n x n)
 */
double** compute_normalized_similarity_matrix(double **A, double **D, int n) {
    double **W = (double **)malloc(n * sizeof(double *));
    double *D_sqrt_inv = (double *)malloc(n * sizeof(double));
    int i;
    int j;
    for (i = 0; i < n; i++) {
        W[i] = (double *)calloc(n, sizeof(double));
        D_sqrt_inv[i] = (D[i][i] > 0) ? 1.0 / sqrt(D[i][i]) : 0.0;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = D_sqrt_inv[i] * A[i][j] * D_sqrt_inv[j];
        }
    }

    free(D_sqrt_inv);
    return W;
}
/* Function to compute the squared Euclidean distance between two vectors */
double squared_euclidean_distance(double *x1, double *x2, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return sum;
}

/* Function to compute the similarity matrix A */
double** compute_similarity_matrix(double **X, int n, int d) {
    double **A = (double **)malloc(n * sizeof(double *));
    int i;
    int j;
    for (i = 0; i < n; i++) {
        A[i] = (double *)calloc(n, sizeof(double)); /* initialize with zeros */
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                double dist_sq = squared_euclidean_distance(X[i], X[j], d);
                A[i][j] = exp(-dist_sq / 2.0);
            }
        }
    }
    return A;
}

/**
 * Computes the diagonal degree matrix D from a similarity matrix A.
 * Each diagonal element d_i is the sum of the i-th row of A.
 * 
 * @param A The similarity matrix (n x n)
 * @param n The number of data points
 * @return A diagonal matrix D (n x n) where D[i][i] = sum of A[i][*]
 */
double** compute_diagonal_degree_matrix(double **A, int n) {
    double **D = (double **)malloc(n * sizeof(double *));
    int i;
    int j;
    for (i = 0; i < n; i++) {
        D[i] = (double *)calloc(n, sizeof(double));
    }

    for (i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (j = 0; j < n; j++) {
            row_sum += A[i][j];
        }
        D[i][i] = row_sum;
    }

    return D;
}

void compute_WH(double **W, double **H, int n, int k, double **WH) {
    int i;
    int j;
    int l;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            WH[i][j] = 0.0;
            for (l = 0; l < n; l++) {
                WH[i][j] += W[i][l] * H[l][j];
            }
        }
    }
}

void compute_HTH(double **H, int n, int k, double **HTH) {
    int i;
    int j;
    int l;
    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            HTH[i][j] = 0.0;
            for (l = 0; l < n; l++) {
                HTH[i][j] += H[l][i] * H[l][j];
            }
        }
    }
}

void compute_denominator(double **H, double **HTH, int n, int k, double **denominator) {
    int i;
    int j;
    int l;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            denominator[i][j] = 0.0;
            for (l = 0; l < k; l++) {
                denominator[i][j] += H[i][l] * HTH[l][j];
            }
        }
    }
}

void update_H(double **H, double **WH, double **denominator, int n, int k, double beta) {
    int i;
    int j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (denominator[i][j] != 0.0) {
                H[i][j] = H[i][j] * (1 - beta + beta * (WH[i][j] / denominator[i][j]));
            }
        }
    }
}

void symnmf(double **H, double **W, int n, int k, int max_iter, double epsilon) {
    double beta = 0.5;
    double **WH = (double **)malloc(n * sizeof(double *));
    double **HTH = (double **)malloc(k * sizeof(double *));
    double **numerator = (double **)malloc(n * sizeof(double *));
    double **denominator = (double **)malloc(n * sizeof(double *));
    double **H_prev = (double **)malloc(n * sizeof(double *));
    int i, j;
    int iter;
    double diff = 0.0;

    for (i = 0; i < n; i++) {
        WH[i] = (double *)calloc(k, sizeof(double));
        numerator[i] = (double *)calloc(k, sizeof(double));
        denominator[i] = (double *)calloc(k, sizeof(double));
        H_prev[i] = (double *)calloc(k, sizeof(double));
    }
    for (i = 0; i < k; i++) {
        HTH[i] = (double *)calloc(k, sizeof(double));
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* Store previous H for convergence check */
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_prev[i][j] = H[i][j];
            }
        }
        
        compute_WH(W, H, n, k, WH);
        compute_HTH(H, n, k, HTH);
        compute_denominator(H, HTH, n, k, denominator);
        update_H(H, WH, denominator, n, k, beta);
        
        /* Simple Frobenius norm convergence check like Yahel's reference */
        diff = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                double delta = H[i][j] - H_prev[i][j];
                diff += delta * delta;
            }
        }
        if (diff < epsilon) {
            break;
        }
    }

    /* Free temporary matrices */
    for (i = 0; i < n; i++) {
        free(WH[i]);
        free(numerator[i]);
        free(denominator[i]);
        free(H_prev[i]);
    }
    for (i = 0; i < k; i++) {
        free(HTH[i]);
    }
    free(WH);
    free(numerator);
    free(denominator);
    free(H_prev);
    free(HTH);
}

/* Declare function prototypes (implementations to be added) */
double** read_input_file(const char *filename, int *n, int *d);
void print_matrix(double **mat, int rows, int cols);
void free_matrix(double **mat, int rows);

int main(int argc, char *argv[]) {
    const char *goal;
    const char *input_path;
    double **X, **A, **D, **W;
    int n = 0, d = 0;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s [sym|ddg|norm] <input_file>\n", argv[0]);
        return 1;
    }

    goal = argv[1];
    input_path = argv[2];

    X = read_input_file(input_path, &n, &d);
    if (!X) {
        fprintf(stderr, "Error reading input file.\n");
        return 1;
    }

    if (strcmp(goal, "sym") == 0) {
        A = compute_similarity_matrix(X, n, d);
        print_matrix(A, n, n);
        free_matrix(A, n);
    } else if (strcmp(goal, "ddg") == 0) {
        A = compute_similarity_matrix(X, n, d);
        D = compute_diagonal_degree_matrix(A, n);
        print_matrix(D, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
    } else if (strcmp(goal, "norm") == 0) {
        A = compute_similarity_matrix(X, n, d);
        D = compute_diagonal_degree_matrix(A, n);
        W = compute_normalized_similarity_matrix(A, D, n);
        print_matrix(W, n, n);
        free_matrix(A, n);
        free_matrix(D, n);
        free_matrix(W, n);
    } else {
        fprintf(stderr, "Invalid goal: %s\n", goal);
        free_matrix(X, n);
        return 1;
    }

    free_matrix(X, n);
    return 0;
}
/* Implementation of read_input_file, print_matrix, and free_matrix */
double** read_input_file(const char *filename, int *n, int *d) {
    FILE *fp = fopen(filename, "r");
    double **data = NULL;
    double value;
    int c;
    int rows = 0, cols = 0, i, j;

    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return NULL;
    }

    /* Count rows and columns */
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') rows++;
        if (rows == 0 && c == ',') cols++;
    }
    cols++; /* number of columns is number of commas + 1 */

    rewind(fp);

    data = (double **)malloc(rows * sizeof(double *));
    for (i = 0; i < rows; i++) {
        data[i] = (double *)malloc(cols * sizeof(double));
        for (j = 0; j < cols; j++) {
            if (fscanf(fp, "%lf", &value) != 1) {
                fprintf(stderr, "Invalid data format in file.\n");
                fclose(fp);
                free_matrix(data, i); /* free already allocated rows */
                return NULL;
            }
            data[i][j] = value;
            if (j < cols - 1) fgetc(fp); /* skip comma */
        }
    }

    fclose(fp);
    *n = rows;
    *d = cols;
    return data;
}

void print_matrix(double **mat, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            if (j == cols - 1)
                printf("%.4f", mat[i][j]);
            else
                printf("%.4f,", mat[i][j]);
        }
        printf("\n");
    }
}

void free_matrix(double **mat, int rows) {
    int i;
    for (i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}