#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#if defined(__has_include)
  #if __has_include(<cblas.h>)
    #include <cblas.h>
  #elif __has_include(<vecLib/cblas.h>)
    #include <vecLib/cblas.h>
  #elif __has_include(<Accelerate/Accelerate.h>)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#else
  #include <cblas.h>
#endif

typedef struct {
  uint64_t row_size;
  uint64_t column_size;
  double* array;
} matrix;

/** Return a random double in [min, max). */
double randfrom(double min, double max);

/** Allocate a row-major matrix (row_size x column_size), zero-initialized. */
matrix create_matrix(uint64_t row_size, uint64_t column_size);

/** Free the heap storage for a matrix (does not free the struct itself). */
void free_matrix(matrix* mat);

/**
 * Matrix multiply wrapper around cblas_dgemm.
 * IMPORTANT: this returns a newly-allocated matrix (it does not write into C).
 * tranpose:
 *   0: C = alpha*A*B + beta*C
 *   1: C = alpha*A^T*B + beta*C
 *   2: C = alpha*A*B^T + beta*C
 *   3: C = alpha*A^T*B^T + beta*C
 */
matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta, uint8_t tranpose);

/** Compute A*B + bias (C interpreted as bias vector). Returns a newly-allocated matrix. */
matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta);

/** Elementwise multiply A and B. Returns a newly-allocated matrix. */
matrix hadamard(matrix* A, matrix* B);

/** In-place subtraction A -= B. */
void matrix_subtract(matrix* A, matrix* B);

/** Row-sum reduction over columns; returns (row_size x 1). */
matrix row_sum(matrix* A);

/** Return a newly-allocated copy of A scaled by 'scale'. */
matrix matrix_scale(matrix* A, double scale);

/** Fill matrix with random values in [min, max). */
void fill_matrix(matrix* mat, double min, double max);

/** Fill matrix with a constant value. */
void set_matrix(matrix* mat, double val);

/** Print (row_size, column_size). */
void shape(matrix* mat);

#endif