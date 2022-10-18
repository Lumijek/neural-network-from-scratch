#include <stdio.h>
#include <sys/param.h>
#include <math.h>
#include <assert.h>

#include "neural_network.h"

matrix relu(matrix* activations) {
	matrix activated_matrix = create_matrix(activations->row_size, activations->column_size);
	for(int i = 0; i < activations->row_size * activations->column_size; i++) {
		activated_matrix.array[i] = MAX(0, activations->array[i]);
	}
	return activated_matrix;
}

matrix relu_prime(matrix* activations) {
	matrix relu_prime_matrix = create_matrix(activations->row_size, activations->column_size);
	for(int i = 0; i < activations->row_size * activations->column_size; i++) {
		relu_prime_matrix.array[i] = (activations->array[i] > 0.0) ? 1.0 : 0.0; 
	}
	return relu_prime_matrix;
}

activation_function get_activation(char* activation) {
	//TODO: make activation lowercase
	if(strcmp(activation, "relu") == 0) {
		return relu;
	}
	else {
		printf("%s is not a valid activation function.\n", activation);
		exit(EXIT_FAILURE);
	}
}

activation_function get_derivative_activation(char* activation) {
	//TODO: make activation lowercase
	if(strcmp(activation, "relu") == 0) {
		return relu_prime;
	}
	else {
		printf("%s is not a valid activation function.\n", activation);
		exit(EXIT_FAILURE);
	}
}
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
		last_activations = network->layers[i].z(&network->layers[i], &last_activations);
		network->layers[i].zs = last_activations;
		last_activations = network->layers[i].a(&last_activations);
		network->layers[i].activations = last_activations;
	}
	return last_activations;
}

void back_propagate(neural_network* network, matrix* inputs, matrix* y_true) {
	uint16_t layer_size = network->number_of_layers - 1; // subtract 1 for indexing
	layer* l = &network->layers[layer_size];
	matrix l_activations = l->zs;
	uint16_t batch_size = l_activations.column_size;
	long double scale_factor = (long double) network->learning_rate / (long double) batch_size;

	matrix dC_da = l2cost_prime(&l_activations, y_true);
	matrix da_dz = l->a_prime(&l_activations);

	matrix delta_l_prev = hadamard(&dC_da, &da_dz);
	free_matrix(&dC_da);
	free_matrix(&da_dz);
	matrix weights_prev = l->weights;

	matrix C = create_matrix(delta_l_prev.row_size, network->layers[layer_size - 1].activations.row_size);
	matrix weight_sub = matrix_m_multiply(&delta_l_prev, &network->layers[layer_size - 1].activations, &C, scale_factor, 0.0, 2);
	free_matrix(&C);
	matrix bias_sub = matrix_scale(&delta_l_prev, scale_factor);

	matrix_subtract(&l->weights, &weight_sub);
	matrix_subtract(&l->biases, &bias_sub);

	free_matrix(&weight_sub);
	free_matrix(&bias_sub);
	free_matrix(&delta_l_prev);

	for(int i = layer_size - 1; i >= 1; i--) {
		l = &network->layers[i];
		l_activations = l->zs;
		da_dz = l->a_prime(&l_activations);

		matrix temp = create_matrix(weights_prev.column_size, delta_l_prev.column_size);
		matrix dl_new = matrix_m_multiply(&weights_prev, &delta_l_prev, &temp, 1.0, 1.0, 1);
		free_matrix(&temp);


		delta_l_prev = hadamard(&dl_new, &da_dz);
		weights_prev = l->weights;
		free_matrix(&dl_new);
		free_matrix(&da_dz);

		C = create_matrix(delta_l_prev.row_size, network->layers[i - 1].activations.row_size);
		weight_sub = matrix_m_multiply(&delta_l_prev, &network->layers[i - 1].activations, &C, scale_factor, 0.0, 2);
		bias_sub = matrix_scale(&delta_l_prev, scale_factor);
		matrix_subtract(&l->weights, &weight_sub);
		matrix_subtract(&l->biases, &bias_sub);

		free_matrix(&C);
		free_matrix(&weight_sub);
		free_matrix(&bias_sub);
		free_matrix(&delta_l_prev);


	}
	l = &network->layers[0];
	l_activations = l->zs;
	da_dz = l->a_prime(&l_activations);

	matrix temp = create_matrix(weights_prev.column_size, delta_l_prev.column_size);
	matrix dl_new = matrix_m_multiply(&weights_prev, &delta_l_prev, &temp, 1.0, 1.0, 1);
	free_matrix(&temp);


	delta_l_prev = hadamard(&dl_new, &da_dz);
	weights_prev = l->weights;
	free_matrix(&dl_new);
	free_matrix(&da_dz);

	C = create_matrix(delta_l_prev.row_size, inputs->row_size);
	weight_sub = matrix_m_multiply(&delta_l_prev, inputs, &C, scale_factor, 0.0, 2);
	bias_sub = matrix_scale(&delta_l_prev, scale_factor);
	matrix_subtract(&l->weights, &weight_sub);
	matrix_subtract(&l->biases, &bias_sub);

	free_matrix(&C);
	free_matrix(&weight_sub);
	free_matrix(&bias_sub);
	free_matrix(&delta_l_prev);
}

double l2cost(matrix* activations, matrix* y_true) {
	assert(activations->row_size == y_true->row_size && activations->column_size == y_true->column_size);
	double total = 0;
	for(int i = 0; i < activations->row_size; i++) {
		total += (activations->array[i] - y_true->array[i]) * (activations->array[i] - y_true->array[i]);
	}
	return total / (2.0 * activations->column_size); // column size = batch size
}

matrix l2cost_prime(matrix* activations, matrix* y_true) {
	assert(activations->row_size == y_true->row_size && activations->column_size == y_true->column_size);
	matrix cost_prime = create_matrix(activations->row_size, activations->column_size);
	for(int i = 0; i < activations->row_size * activations->column_size; i++) {
		cost_prime.array[i] = activations->array[i] - y_true->array[i];
	}
	return cost_prime;
}
matrix linear_function(layer* linear_layer, matrix* activations) {
	return matrix_m_multiply(&linear_layer->weights, activations, &linear_layer->biases, 1.0, 1.0, 0);
}

layer linear(uint64_t in, uint64_t out, uint16_t batch_size, char* activation) {
	layer linear_layer;
	linear_layer.neurons = out;
	double min = -1.0f / sqrt(out);
	double max = 1.0f / sqrt(out);

	linear_layer.weights = create_matrix(out, in);
	fill_matrix(&linear_layer.weights, min, max);
	linear_layer.biases = create_matrix(out, batch_size);
	fill_matrix(&linear_layer.biases, min, max);

	linear_layer.z = linear_function;
	linear_layer.a = get_activation(activation);
	linear_layer.a_prime = get_derivative_activation(activation);
	return linear_layer;
}

void free_network_memory(neural_network* network) {
	for(int i = 0; i < network->number_of_layers; i++) {
		free(network->layers[i].zs.array);
		free(network->layers[i].activations.array);
	}
}

