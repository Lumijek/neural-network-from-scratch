#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <stdint.h>
#include <stdlib.h>
#include "matrix.h"

/* Activation function signature: takes a matrix and returns a newly allocated matrix. */
typedef matrix (*activation_function)(matrix* activations);

void print(matrix m);

typedef struct layer{
  uint64_t neurons;
  matrix weights;
  matrix biases;
  matrix zs;
  matrix activations;
  matrix (*z)(struct layer* l, matrix* last_activations);
  matrix (*a) (matrix* activations);
  matrix (*a_prime) (matrix* activations);
} layer;

typedef struct {
  uint16_t number_of_layers;
  layer* layers;
  double learning_rate;
} neural_network;

activation_function get_activation(char* activation);
activation_function get_derivative_activation(char* activation);

neural_network create_network();
void add_layer(neural_network* network, layer l);

/* Forward pass caches z and a per layer in network->layers[i].{zs,activations}. */
matrix forward_pass(neural_network* network, matrix* inputs);

/* Backprop assumes y_true is shaped like the network output: (classes x batch). */
void back_propagate(neural_network* network, matrix* inputs, matrix* y_true);

matrix linear_function(layer* linear_layer, matrix* activations);
layer linear(uint64_t in, uint64_t out, char* activation);

matrix relu(matrix* activations);
matrix softmax(matrix* activations);

/* For classification with softmax outputs */
double cross_entropy(matrix* activations, matrix* y_true);
matrix cross_entropy_prime(matrix* activations, matrix* y_true);

/* L2 cost */
double l2cost(matrix* activations, matrix* y_true);
matrix l2cost_prime(matrix* activations, matrix* y_true);

void free_network_memory(neural_network* network);

#endif
