#ifndef SYMNMF_H
#define SYMNMF_H

double** compute_similarity_matrix(double **X, int n, int d);
double** compute_diagonal_degree_matrix(double **A, int n);
double** compute_normalized_similarity_matrix(double **A, double **D, int n);

void compute_WH(double **W, double **H, int n, int k, double **WH);
void compute_HTH(double **H, int n, int k, double **HTH);
void compute_denominator(double **H, double **HTH, int n, int k, double **denominator);
void update_H(double **H, double **WH, double **denominator, int n, int k, double beta);
double compute_convergence_diff(double **H, double **WH, double **denominator, int n, int k, double beta);

void symnmf(double **H, double **W, int n, int k, int max_iter, double epsilon);

#endif // SYMNMF_H
