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
  mat.array = malloc(sizeof(double) * mat.row_size * mat.column_size);
  if(!mat.array) {
    printf("Failed to allocate memory for matrix of size (%llu, %llu)\n", row_size, column_size);
  }
  return mat;
}

matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
  assert(A->column_size == B->row_size);
  assert(C->row_size == A->row_size && C->column_size == B->column_size);
  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->row_size, B->column_size, A->column_size, alpha, A->array, A->column_size, B->array, B->column_size, beta, C_copy.array, C->column_size);
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

void fill_matrix(matrix* mat, double min, double max) {
  for(int i = 0; i < mat->row_size * mat->column_size; i++) {
    mat->array[i] = randfrom(min, max);
  }
}