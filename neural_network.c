#include <stdio.h>
#include <sys/param.h>
#include <math.h>

#include "neural_network.h"

neural_network create_network() {
	neural_network network = {.number_of_layers = 0, .layers = NULL};
	return network;
}

void add_layer(neural_network* network, layer l) {
	network->layers = realloc(network->layers, ++network->number_of_layers * sizeof(layer));
	if(!network->layers) {
		printf("Failed to reallocate memory for %huth layer\n", network->number_of_layers);
	}
	network->layers[network->number_of_layers - 1] = l;
}

matrix forward_pass(neural_network* network, matrix* inputs) {
	matrix last_activations = *inputs;
	for(int i = 0; i < network->number_of_layers; i++) {
		last_activations = network->layers[i].compute_activations(&network->layers[i], &last_activations);
		network->layers[i].activations = last_activations;
	}
	return last_activations;
}


matrix linear_function(layer* linear_layer, matrix* activations) {
	return matrix_m_multiply(&linear_layer->weights, activations, &linear_layer->biases, 1.0, 1.0);
}

layer linear(uint64_t in, uint64_t out, uint16_t batch_size) {
	layer linear_layer;
	linear_layer.neurons = out;

	double min = -1.0f / sqrt(out);
	double max = 1.0f / sqrt(out);
	linear_layer.weights = create_matrix(out, in);
	fill_matrix(&linear_layer.weights, min, max);
	linear_layer.biases = create_matrix(out, batch_size);
	fill_matrix(&linear_layer.biases, min, max);
	linear_layer.compute_activations = linear_function;
	return linear_layer;
}

matrix relu(layer* linear_layer, matrix* activations) {
	matrix activated_matrix = create_matrix(activations->row_size, activations->column_size);
	for(int i = 0; i < activations->row_size * activations->column_size; i++) {
		activated_matrix.array[i] = MAX(0, activations->array[i]);
	}
	return activated_matrix;
}

layer ReLU() {
	layer relu_layer;
	relu_layer.compute_activations = relu;
	return relu_layer;
}

