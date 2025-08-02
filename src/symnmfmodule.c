

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

// Helper: Convert PyArrayObject to C double**
double** pyarray_to_c_matrix(PyArrayObject *array, int n, int d) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(d * sizeof(double));
        for (int j = 0; j < d; j++) {
            matrix[i][j] = *(double *)PyArray_GETPTR2(array, i, j);
        }
    }
    return matrix;
}

// Helper: Convert C double** to NumPy 2D array
PyObject* c_matrix_to_pyarray(double **matrix, int n, int m) {
    npy_intp dims[2] = {n, m};
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            *(double *)PyArray_GETPTR2((PyArrayObject *)array, i, j) = matrix[i][j];
        }
    }
    return array;
}

// Python wrapper for compute_similarity_matrix
static PyObject* py_compute_similarity_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *X_in;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_in)) {
        return NULL;
    }

    int n = (int)PyArray_DIM(X_in, 0);
    int d = (int)PyArray_DIM(X_in, 1);
    double **X = pyarray_to_c_matrix(X_in, n, d);

    double **A = compute_similarity_matrix(X, n, d);
    PyObject *result = c_matrix_to_pyarray(A, n, n);

    // Free C matrix X and A
    for (int i = 0; i < n; i++) {
        free(X[i]);
        free(A[i]);
    }
    free(X);
    free(A);

    return result;
}

// Python wrapper for compute_diagonal_degree_matrix
static PyObject* py_compute_diagonal_degree_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *A_in;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &A_in)) {
        return NULL;
    }
    int n = (int)PyArray_DIM(A_in, 0);
    double **A = pyarray_to_c_matrix(A_in, n, n);
    double **D = compute_diagonal_degree_matrix(A, n);
    PyObject *result = c_matrix_to_pyarray(D, n, n);
    for (int i = 0; i < n; i++) { free(A[i]); free(D[i]); }
    free(A); free(D);
    return result;
}

// Python wrapper for compute_normalized_similarity_matrix
static PyObject* py_compute_normalized_similarity_matrix(PyObject *self, PyObject *args) {
    PyArrayObject *A_in, *D_in;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A_in, &PyArray_Type, &D_in)) {
        return NULL;
    }
    int n = (int)PyArray_DIM(A_in, 0);
    double **A = pyarray_to_c_matrix(A_in, n, n);
    double **D = pyarray_to_c_matrix(D_in, n, n);
    double **W = compute_normalized_similarity_matrix(A, D, n);
    PyObject *result = c_matrix_to_pyarray(W, n, n);
    for (int i = 0; i < n; i++) { free(A[i]); free(D[i]); free(W[i]); }
    free(A); free(D); free(W);
    return result;
}

// Python wrapper for symnmf
static PyObject* py_symnmf(PyObject *self, PyObject *args) {
    PyArrayObject *W_in, *H_in;
    int max_iter;
    double epsilon;
    if (!PyArg_ParseTuple(args, "O!O!id", &PyArray_Type, &W_in, &PyArray_Type, &H_in, &max_iter, &epsilon)) {
        return NULL;
    }
    int n = (int)PyArray_DIM(W_in, 0);
    int k = (int)PyArray_DIM(H_in, 1);
    double **W = pyarray_to_c_matrix(W_in, n, n);
    double **H = pyarray_to_c_matrix(H_in, n, k);
    symnmf(H, W, n, k, max_iter, epsilon);
    PyObject *result = c_matrix_to_pyarray(H, n, k);
    for (int i = 0; i < n; i++) { free(W[i]); free(H[i]); }
    free(W); free(H);
    return result;
}

// Method table
static PyMethodDef SymnmfMethods[] = {
    {"compute_similarity_matrix", py_compute_similarity_matrix, METH_VARARGS, "Compute similarity matrix"},
    {"compute_diagonal_degree_matrix", py_compute_diagonal_degree_matrix, METH_VARARGS, "Compute diagonal degree matrix"},
    {"compute_normalized_similarity_matrix", py_compute_normalized_similarity_matrix, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Run symnmf optimization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    NULL,
    -1,
    SymnmfMethods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    import_array();  // Required for NumPy
    return PyModule_Create(&symnmfmodule);
}