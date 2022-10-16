#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mnist_loader.h"
#include "matrix.h"
#include "neural_network.h"

int main(int argc, char *argv[])
{
   srand(time(NULL));
   load_mnist_data();

   neural_network network = create_network();
   uint16_t batch_size = 248;
   add_layer(&network, linear(784, 16, batch_size));
   add_layer(&network, ReLU());
   add_layer(&network, linear(16, 16, batch_size));
   add_layer(&network, ReLU());
   add_layer(&network, linear(16, 10, batch_size));


   matrix out = create_matrix(784, batch_size);

   clock_t begin = clock();
   matrix output = forward_pass(&network, &out);

   clock_t end = clock();

   printf("Shape: (%llu, %llu)\n", output.row_size, output.column_size);
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("%f\n", time_spent);

   return 0;
}