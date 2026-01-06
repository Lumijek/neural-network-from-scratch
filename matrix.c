#include <stdio.h>
#include <assert.h>
#include "matrix.h"

/** Return a random double in [min, max). */
double randfrom(double min, double max) {
  double range = (max - min);
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

/** Allocate a row-major matrix (row_size x column_size), zero-initialized. */
matrix create_matrix(uint64_t row_size, uint64_t column_size) {
  matrix mat;
  mat.row_size = row_size;
  mat.column_size = column_size;
  mat.array = calloc(mat.row_size * mat.column_size, sizeof(double));
  if (!mat.array) {
    printf("Failed to allocate memory for matrix of size (%llu, %llu)\n", row_size, column_size);
  }
  return mat;
}

/** Free the heap storage for a matrix (does not free the struct itself). */
void free_matrix(matrix* mat) {
  free(mat->array);
}

/** Matrix multiply wrapper around cblas_dgemm; returns a newly-allocated matrix. */
matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta, uint8_t tranpose) {
  uint8_t t1 = CblasNoTrans;
  uint8_t t2 = CblasNoTrans;
  uint64_t M = C->row_size;
  uint64_t N = C->column_size;
  uint64_t K = 0;

  if (tranpose == 0) {
    assert(A->column_size == B->row_size);
    assert(C->row_size == A->row_size && C->column_size == B->column_size);
    K = A->column_size;
  } else if (tranpose == 1) {
    assert(A->row_size == B->row_size);
    assert(A->column_size == C->row_size && C->column_size == B->column_size);
    t1 = CblasTrans;
    K = B->row_size;
  } else if (tranpose == 2) {
    assert(A->column_size == B->column_size);
    assert(A->row_size == C->row_size && C->column_size == B->row_size);
    t2 = CblasTrans;
    K = A->column_size;
  } else if (tranpose == 3) {
    assert(A->row_size == B->column_size);
    assert(A->column_size == C->row_size && C->column_size == B->row_size);
    t1 = CblasTrans;
    t2 = CblasTrans;
    K = A->row_size;
  }

  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));

  cblas_dgemm(
    CblasRowMajor, t1, t2,
    M, N, K,
    alpha,
    A->array, A->column_size,
    B->array, B->column_size,
    beta,
    C_copy.array, C->column_size
  );

  return C_copy;
}

/** Compute A*B + bias (C interpreted as bias vector). Returns a newly-allocated matrix. */
matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
  assert(A->column_size == B->row_size);
  assert(A->row_size == C->row_size);

  matrix temp = create_matrix(A->row_size, B->column_size);
  matrix m = matrix_m_multiply(A, B, &temp, alpha, 1.0, 0);
  free_matrix(&temp);

  for (int i = 0; i < m.row_size; i++) {
    for (int j = 0; j < m.column_size; j++) {
      m.array[i * m.column_size + j] += beta * C->array[i];
    }
  }
  return m;
}

/** Elementwise multiply A and B. Returns a newly-allocated matrix. */
matrix hadamard(matrix* A, matrix* B) {
  assert(A->row_size == B->row_size && A->column_size == B->column_size);
  matrix C = create_matrix(A->row_size, A->column_size);
  for (int i = 0; i < A->row_size * A->column_size; i++) {
    C.array[i] = A->array[i] * B->array[i];
  }
  return C;
}

/** In-place subtraction A -= B. */
void matrix_subtract(matrix* A, matrix* B) {
  assert(A->row_size == B->row_size && A->column_size == B->column_size);
  for (int i = 0; i < A->row_size * A->column_size; i++) {
    A->array[i] = A->array[i] - B->array[i];
  }
}

/** Return a newly-allocated copy of A scaled by 'scale'. */
matrix matrix_scale(matrix* A, double scale) {
  matrix A_copy = create_matrix(A->row_size, A->column_size);
  memcpy(A_copy.array, A->array, A->row_size * A->column_size * sizeof(double));
  cblas_dscal((int)(A_copy.row_size * A_copy.column_size), scale, A_copy.array, 1);
  return A_copy;
}

/** Print (row_size, column_size). */
void shape(matrix* mat) {
  printf("(%llu, %llu)\n", mat->row_size, mat->column_size);
}

/** Row-sum reduction over columns; returns (row_size x 1). */
matrix row_sum(matrix* A) {
  matrix B = create_matrix(A->column_size, 1);
  matrix C = create_matrix(A->row_size, 1);
  set_matrix(&B, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, A->row_size, A->column_size, 1.0, A->array, A->column_size, B.array, 1, 1.0, C.array, 1);
  free_matrix(&B);
  return C;
}

/** Fill matrix with random values in [min, max). */
void fill_matrix(matrix* mat, double min, double max) {
  for (int i = 0; i < mat->row_size * mat->column_size; i++) {
    mat->array[i] = randfrom(min, max);
  }
}

/** Fill matrix with a constant value. */
void set_matrix(matrix* mat, double val) {
  for (int i = 0; i < mat->row_size * mat->column_size; i++) {
    mat->array[i] = val;
  }
}