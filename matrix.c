#include <stdio.h>
#include <assert.h>
#include "matrix.h"

double randfrom(double min, double max) {
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

matrix create_matrix(uint64_t row_size, uint64_t column_size) {
  matrix mat;
  mat.row_size = row_size;
  mat.column_size = column_size;
  mat.array = calloc(mat.row_size * mat.column_size, sizeof(double));
  if(!mat.array) {
    printf("Failed to allocate memory for matrix of size (%llu, %llu)\n", row_size, column_size);
  }
  return mat;
}

void free_matrix(matrix* mat) {
  free(mat->array);
}
matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta, uint8_t tranpose) {
  uint8_t t1 = CblasNoTrans;
  uint8_t t2 = CblasNoTrans;
  uint64_t M = 0, N = 0, K = 0;
  if (tranpose == 0) {
    assert(A->column_size == B->row_size);
    assert(C->row_size == A->row_size && C->column_size == B->column_size);
    M = A->row_size;
    N = B->column_size;
    K = A->column_size;
  }
  else if(tranpose == 1) {
    assert(A->row_size == B->row_size);
    assert(A->column_size == C->row_size && C->column_size == B->column_size);
    t1 = CblasTrans;
    M = A->column_size;
    N = B->column_size;
    K = A->row_size;
  }
  else if(tranpose == 2) {
    assert(A->column_size == B->column_size);
    assert(A->row_size == C->row_size && C->column_size == B->row_size);
    t2 = CblasTrans;
    M = A->row_size;
    N = B->row_size;
    K = A->column_size;
  }
  else if(tranpose == 3) {
    assert(A->row_size == B->column_size);
    assert(A->column_size == C->row_size && C->column_size == B->row_size);
    t1 = CblasTrans;
    t2 = CblasTrans;
    M = A->column_size;
    N = B->row_size;
    K = A->row_size;
  }
  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A->array, K, B->array, N, beta, C_copy.array, N);
  return C_copy;
}

matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
  assert(A->column_size == B->row_size);
  assert(A->row_size == C->row_size);
  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));
  cblas_dgemv(CblasRowMajor, CblasNoTrans, A->row_size, A->column_size, alpha, A->array, A->column_size, B->array, 1, beta, C_copy.array, 1);
  return C_copy;
}
matrix hadamard(matrix* A, matrix* B) {
   assert(A->row_size == B->row_size && A->column_size == B->column_size);
   matrix C = create_matrix(A->row_size, A->column_size);
   for(int i = 0; i < A->row_size * A->column_size; i++) {
      C.array[i] = A->array[i] * B->array[i];
   }
   return C;
}

void matrix_subtract(matrix* A, matrix* B) {
   assert(A->row_size == B->row_size && A->column_size == B->column_size);
   for(int i = 0; i < A->row_size * A->column_size; i ++) {
      A->array[i] = A->array[i] * B->array[i];
   }
}

matrix matrix_scale(matrix* A, double scale) {
  matrix A_copy = create_matrix(A->row_size, A->column_size);
  memcpy(A_copy.array, A->array, A->row_size * A->column_size * sizeof(double));
  cblas_dscal(A_copy.row_size, scale, A_copy.array, 1);
  return A_copy;
}
void shape(matrix* mat) {
  printf("(%llu, %llu)\n", mat->row_size, mat->column_size);
}

matrix row_sum(matrix* mat) {
  matrix A = create_matrix(mat->row_size, 1);
  for(int i = 0; i < mat->row_size; i++) {
    for(int j = 0; j < mat->column_size; j++) {
      A.array[i] += mat->array[i * mat->row_size + j];
    }
  }
  return A;
}
void fill_matrix(matrix* mat, double min, double max) {
  for(int i = 0; i < mat->row_size * mat->column_size; i++) {
    mat->array[i] = randfrom(min, max);
  }
}