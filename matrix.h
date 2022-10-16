#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cblas.h>


typedef struct {
  uint64_t row_size;
  uint64_t column_size;
  double* array;
} matrix;

double randfrom(double min, double max);
matrix create_matrix(uint64_t row_size, uint64_t column_size);
matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta);
matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta);
void fill_matrix(matrix* mat, double min, double max);

#endif