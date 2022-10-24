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
void free_matrix(matrix* mat);
matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta, uint8_t tranpose);
matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta);
matrix hadamard(matrix* A, matrix* B);
void matrix_subtract(matrix* A, matrix* B);
matrix row_sum(matrix* A);
matrix matrix_scale(matrix* A, double scale);
void fill_matrix(matrix* mat, double min, double max);
void set_matrix(matrix* mat, double val);
void shape(matrix* mat);

#endif