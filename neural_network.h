#ifndef NEURAL_NETWORK_C_
#define NEURAL_NETWORK_C_
#include <stdint.h>
#include <stdlib.h>
#include "matrix.h"

double randfrom(double min, double max);

typedef struct layer{
	uint64_t neurons;
	matrix weights;
	matrix biases;
	matrix activations;
	matrix (*compute_activations)(struct layer* l, matrix* last_activations);
} layer;

typedef struct {
	uint16_t number_of_layers;
	layer* layers;
} neural_network;


neural_network create_network();
void add_layer(neural_network* network, layer l);
matrix forward_pass(neural_network* network, matrix* inputs);

matrix linear_function(layer* linear_layer, matrix* activations);
layer linear(uint64_t in, uint64_t out, uint16_t batch_size);

matrix relu(layer* linear_layer, matrix* activations);
layer ReLU();


#endif