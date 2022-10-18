#ifndef NEURAL_NETWORK_C_
#define NEURAL_NETWORK_C_
#include <stdint.h>
#include <stdlib.h>
#include "matrix.h"

typedef matrix (*activation_function)(matrix* activations);


double randfrom(double min, double max);

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
matrix forward_pass(neural_network* network, matrix* inputs);
void back_propagate(neural_network* network, matrix* inputs, matrix* y_true);

matrix linear_function(layer* linear_layer, matrix* activations);
layer linear(uint64_t in, uint64_t out, uint16_t batch_size, char* activation);

matrix relu(matrix* activations);
layer ReLU();

double l2cost(matrix* activations, matrix* y_true);
matrix l2cost_prime(matrix* activations, matrix* y_true);

void free_network_memory(neural_network* network);
#endif