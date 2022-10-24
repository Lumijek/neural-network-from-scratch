#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "mnist_loader.h"
#include "matrix.h"
#include "neural_network.h"
#include <unistd.h>


matrix naive(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
   assert(A->column_size == B->row_size);
   assert(A->row_size == C->row_size);
   matrix temp = create_matrix(A->row_size, B->column_size);
   matrix m = matrix_m_multiply(A, B, &temp, alpha, 1.0, 0);
   free_matrix(&temp);
   for(int i = 0; i < m.row_size; i++) {
      for(int j = 0; j < m.column_size; j++) {
         m.array[i * m.column_size + j] += beta * C->array[i];
      }
   }
  return m;
}
int main(int argc, char *argv[])
{

   setbuf(stdout, NULL);
   printf("Process ID: %d\n", getpid());
   srand(0);
   load_mnist_data();

   neural_network network = create_network();
   network.learning_rate = 0.01;
   uint16_t batch_size = 5;
   add_layer(&network, linear(1, 2, "relu"));
   add_layer(&network, linear(2, 1, "relu"));

   printf("---------------------------------\n");

   matrix input = create_matrix(1, batch_size);
   matrix y_pred = create_matrix(1, batch_size);

   int data_size = 5;
   for(int i = 0; i < data_size; i++) {
      input.array[i] = i + 1;
      y_pred.array[i] = (2 * i + 4) + randfrom(-0.01, 0.01);
   }

   clock_t begin = clock();
   int l = 20;
   matrix output;
   for(int i = 0; i < l; i++) {
      if(i % 100 == 0) {
         //printf("%f%%\n", ((double) i / (double) l) * 100);
      }


      output = forward_pass(&network, &input);
      back_propagate(&network, &input, &y_pred);
      double t = l2cost(&output, &y_pred);
      printf("Cost: %f\n", t);

   }
   for(int i = 0; i < data_size; i++) {
    printf("X: %f, Y_pred: %f, Y_actual: %f\n", input.array[i], output.array[i], y_pred.array[i]);
   }
   



   clock_t end = clock();

   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("Time Taken: %f seconds\n", time_spent);
   return 0;

   
}














