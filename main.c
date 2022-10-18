#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "mnist_loader.h"
#include "matrix.h"
#include "neural_network.h"
#include <unistd.h>
int main(int argc, char *argv[])
{
   setbuf(stdout, NULL);

   printf("%d\n", getpid());
   srand(time(NULL));
   load_mnist_data();

   neural_network network = create_network();
   network.learning_rate = 0.01;
   uint16_t batch_size = 100;
   add_layer(&network, linear(784, 16, batch_size, "relu"));
   add_layer(&network, linear(16, 24, batch_size, "relu"));

   add_layer(&network, linear(24, 10, batch_size, "relu"));

   matrix input = create_matrix(784, batch_size);
   matrix y_pred = create_matrix(10, batch_size);

   /*
   int data_size = 100;
   double x[data_size];
   double y[data_size];
   for(int i = 0; i < data_size; i++) {
      input.array[i] = i;
      y_pred.array[i] = (2 * i + 4) + randfrom(-0.02, 0.02);
   }
   */

   clock_t begin = clock();
   int l = 5000;
   matrix output;
   for(int i = 0; i < l; i++) {
      if(i % 500 == 0) {
         printf("%f%%\n", ((double) i / (double) l) * 100);
      }


      output = forward_pass(&network, &input);
      back_propagate(&network, &input, &y_pred);
      free_network_memory(&network);

   }

   clock_t end = clock();

   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("%f\n", time_spent);
   return 0;
}