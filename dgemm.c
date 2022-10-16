#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <string.h>


typedef struct {
  uint64_t row_size;
  uint64_t column_size;
  double* array;
} matrix;

typedef struct layer{
   uint64_t neurons;
   matrix weights;
   matrix biases;
   matrix (*compute_activations)(struct layer* l, matrix* activations);
} layer;

typedef struct {
   uint16_t number_of_layers;
   layer* layers;
} neural_network;

matrix create_matrix(uint64_t row_size, uint64_t column_size) {
  matrix mat;
  mat.row_size = row_size;
  mat.column_size = column_size;
  mat.array = malloc(sizeof(double) * mat.row_size * mat.column_size);
  return mat;
}

matrix matrix_m_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->row_size, B->column_size, A->column_size, alpha, A->array, A->column_size, B->array, B->column_size, beta, C_copy.array, C->column_size);
  return C_copy;

}

matrix matrix_v_multiply(matrix* A, matrix* B, matrix* C, double alpha, double beta) {
  matrix C_copy = create_matrix(C->row_size, C->column_size);
  memcpy(C_copy.array, C->array, C->row_size * C->column_size * sizeof(double));
  cblas_dgemv(CblasRowMajor, CblasNoTrans, A->row_size, A->column_size, alpha, A->array, A->column_size, B->array, 1, beta, C_copy.array, 1);
  return C_copy;
}

neural_network create_network() {
   neural_network network = {.number_of_layers = 0, .layers = NULL};
   return network;
}

void add_layer(neural_network* network, layer* l) {
   network->layers = realloc(network->layers, ++network->number_of_layers);
   network->layers[network->number_of_layers - 1] = *l;
}

matrix forward_pass(neural_network* network, matrix* inputs) {
   matrix activations = *inputs;
   for(int i = 0; i < network->number_of_layers; i++) {
      activations = network->layers[i].compute_activations(&network->layers[i], &activations);
   }
   return activations;
}
matrix linear_function(layer* linear_layer, matrix* activations) {
   return matrix_v_multiply(&linear_layer->weights, activations, &linear_layer->biases, 1.0, 1.0);
}

layer linear(uint64_t in, uint64_t out) {
   layer linear_layer;
   linear_layer.neurons = out;
   linear_layer.weights = create_matrix(out, in); // Don't have to transpose matrix when doing vector product with inputs
   linear_layer.biases = create_matrix(out, 1);
   linear_layer.compute_activations = linear_function;
   return linear_layer;

}


int main(int argc, char *argv[])
{
   srand(0);
   layer l = linear(2, 3);
   neural_network network = create_network();
   add_layer(&network, &l);
   printf("After: %llu\n", network.layers[0].weights.column_size);
   matrix i = create_matrix(2, 1);
   printf("After: %llu\n", network.layers[0].weights.column_size);
   matrix output = forward_pass(&network, &i);

   return 0;
}