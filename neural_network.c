#include <stdio.h>
#include <sys/param.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <float.h>

#include "neural_network.h"

void print(matrix m) {
  for(uint64_t i = 0; i < m.row_size; i++) {
    for(uint64_t j = 0; j < m.column_size; j++) {
      printf("%.2f ", m.array[i * m.column_size + j]);
    }
    printf("\n");
  }
  printf("\n");
}

matrix relu(matrix* activations) {
  matrix activated_matrix = create_matrix(activations->row_size, activations->column_size);
  for(uint64_t i = 0; i < activations->row_size * activations->column_size; i++) {
    activated_matrix.array[i] = MAX(0, activations->array[i]);
  }
  return activated_matrix;
}

matrix softmax(matrix* z) {
  /* softmax over rows, per column (sample) */
  matrix out = create_matrix(z->row_size, z->column_size);

  for(uint64_t col = 0; col < z->column_size; col++) {
    /* max for stability */
    double mx = z->array[col];
    for(uint64_t i = 1; i < z->row_size; i++) {
      double v = z->array[i * z->column_size + col];
      if(v > mx) mx = v;
    }

    double sum = 0.0;
    for(uint64_t i = 0; i < z->row_size; i++) {
      double e = exp(z->array[i * z->column_size + col] - mx);
      out.array[i * out.column_size + col] = e;
      sum += e;
    }

    double inv = (sum > 0.0) ? (1.0 / sum) : 0.0;
    for(uint64_t i = 0; i < z->row_size; i++) {
      out.array[i * out.column_size + col] *= inv;
    }
  }

  return out;
}

double cross_entropy(matrix* p, matrix* y_true) {
  assert(p->row_size == y_true->row_size && p->column_size == y_true->column_size);

  const double eps = 1e-12;
  double sum = 0.0;

  for(uint64_t col = 0; col < p->column_size; col++) {
    for(uint64_t i = 0; i < p->row_size; i++) {
      double y = y_true->array[i * y_true->column_size + col];
      if(y != 0.0) {
        double prob = p->array[i * p->column_size + col];
        if(prob < eps) prob = eps;
        sum += -y * log(prob);
      }
    }
  }

  return sum / (double)p->column_size;
}

matrix cross_entropy_prime(matrix* p, matrix* y_true) {
  assert(p->row_size == y_true->row_size && p->column_size == y_true->column_size);

  matrix out = create_matrix(p->row_size, p->column_size);
  for(uint64_t i = 0; i < p->row_size * p->column_size; i++) {
    out.array[i] = p->array[i] - y_true->array[i];
  }
  return out;
}

matrix relu_prime(matrix* activations) {
  matrix relu_prime_matrix = create_matrix(activations->row_size, activations->column_size);
  for(uint64_t i = 0; i < activations->row_size * activations->column_size; i++) {
    relu_prime_matrix.array[i] = (activations->array[i] > 0.0) ? 1.0 : 0.0;
  }
  return relu_prime_matrix;
}

activation_function get_activation(char* activation) {
  if(strcmp(activation, "relu") == 0) {
    return relu;
  }
  else if(strcmp(activation, "softmax") == 0) {
    return softmax;
  }
  else {
    printf("%s is not a valid activation function.\n", activation);
    exit(EXIT_FAILURE);
  }
}

activation_function get_derivative_activation(char* activation) {
  if(strcmp(activation, "relu") == 0) {
    return relu_prime;
  }
  else if(strcmp(activation, "softmax") == 0) {
    /*
      Not a true softmax derivative; by convention this project uses:
        - softmax + cross-entropy on the output layer
        - cross_entropy_prime() handles the gradient, so this isn't used in that case
    */
    return softmax;
  }
  else {
    printf("%s is not a valid activation function.\n", activation);
    exit(EXIT_FAILURE);
  }
}

neural_network create_network() {
  neural_network network = {.number_of_layers = 0, .layers = NULL, .learning_rate = 0.01};
  return network;
}

void add_layer(neural_network* network, layer l) {
  network->layers = realloc(network->layers, ++network->number_of_layers * sizeof(layer));
  if(!network->layers) {
    printf("Failed to reallocate memory for %huth layer\n", network->number_of_layers);
    exit(EXIT_FAILURE);
  }
  network->layers[network->number_of_layers - 1] = l;

#ifdef NN_DEBUG
  printf("Weights %d: \n", network->number_of_layers);
  print(l.weights);
  printf("Biases %d: \n", network->number_of_layers);
  print(l.biases);
#endif
}

matrix forward_pass(neural_network* network, matrix* inputs) {
  matrix last_activations = *inputs;

  for(int i = 0; i < network->number_of_layers; i++) {
    /* free cached matrices from the previous forward pass */
    if(network->layers[i].zs.array) {
      free_matrix(&network->layers[i].zs);
    }
    if(network->layers[i].activations.array) {
      free_matrix(&network->layers[i].activations);
    }

#ifdef NN_DEBUG
    printf("Inputs: \n");
    print(*inputs);
#endif

    matrix z = network->layers[i].z(&network->layers[i], &last_activations);
    network->layers[i].zs = z;

    matrix a = network->layers[i].a(&z);
    network->layers[i].activations = a;

    last_activations = a;

#ifdef NN_DEBUG
    printf("Z_%d:\n", i + 1);
    print(z);
    printf("A_%d:\n", i + 1);
    print(a);
#endif
  }

  return last_activations;
}

void back_propagate(neural_network* network, matrix* inputs, matrix* y_true) {
  assert(network->number_of_layers > 0);

  int last = (int)network->number_of_layers - 1;
  layer* l = &network->layers[last];

  uint16_t batch_size = (uint16_t)l->activations.column_size;
  double scale_factor = network->learning_rate / (double)batch_size;

  matrix delta;
  matrix dC_da;
  matrix da_dz;

  /* Output-layer delta */
  if(l->a == softmax) {
    dC_da = cross_entropy_prime(&l->activations, y_true);
    delta = dC_da; /* reuse the allocated buffer */
  } else {
    dC_da = l2cost_prime(&l->activations, y_true);
    da_dz = l->a_prime(&l->zs);
    delta = hadamard(&dC_da, &da_dz);
    free_matrix(&dC_da);
    free_matrix(&da_dz);
  }

  /* Iterate layers from last -> first */
  for(int i = last; i >= 0; i--) {
    layer* cur = &network->layers[i];

    matrix* prev_a = (i == 0) ? inputs : &network->layers[i - 1].activations;

    /* Weight update: W -= lr/batch * (delta * prev_a^T) */
    matrix Cw = create_matrix(delta.row_size, prev_a->row_size);
    matrix weight_sub = matrix_m_multiply(&delta, prev_a, &Cw, scale_factor, 0.0, 2);

    /* Bias update: b -= lr/batch * sum(delta across batch) */
    matrix bias_reduce = row_sum(&delta);
    matrix bias_sub = matrix_scale(&bias_reduce, scale_factor);
    free_matrix(&bias_reduce);

    matrix_subtract(&cur->weights, &weight_sub);
    matrix_subtract(&cur->biases, &bias_sub);

    free_matrix(&Cw);
    free_matrix(&weight_sub);
    free_matrix(&bias_sub);

    /* Prepare delta for the next layer (if any) */
    if(i > 0) {
      matrix temp = create_matrix(cur->weights.column_size, delta.column_size);
      matrix dl_new = matrix_m_multiply(&cur->weights, &delta, &temp, 1.0, 0.0, 1);
      free_matrix(&temp);

      da_dz = network->layers[i - 1].a_prime(&network->layers[i - 1].zs);

      matrix new_delta = hadamard(&dl_new, &da_dz);

      free_matrix(&dl_new);
      free_matrix(&da_dz);

      /* free old delta, then replace */
      free_matrix(&delta);
      delta = new_delta;
    }
  }

  free_matrix(&delta);
}

double l2cost(matrix* activations, matrix* y_true) {
  assert(activations->row_size == y_true->row_size && activations->column_size == y_true->column_size);

  double total = 0.0;
  for(uint64_t i = 0; i < activations->row_size * activations->column_size; i++) {
    double d = activations->array[i] - y_true->array[i];
    total += d * d;
  }

  return total / (2.0 * (double)activations->column_size);
}

matrix l2cost_prime(matrix* activations, matrix* y_true) {
  assert(activations->row_size == y_true->row_size && activations->column_size == y_true->column_size);

  matrix cost_prime = create_matrix(activations->row_size, activations->column_size);
  for(uint64_t i = 0; i < activations->row_size * activations->column_size; i++) {
    cost_prime.array[i] = activations->array[i] - y_true->array[i];
  }
  return cost_prime;
}

matrix linear_function(layer* linear_layer, matrix* activations) {
  assert(linear_layer->weights.column_size == activations->row_size);
  assert(linear_layer->weights.row_size == linear_layer->biases.row_size);
  return matrix_v_multiply(&linear_layer->weights, activations, &linear_layer->biases, 1.0, 1.0);
}

layer linear(uint64_t in, uint64_t out, char* activation) {
  layer linear_layer;
  linear_layer.neurons = out;

  /* Xavier/He-ish uniform init scale (small weights help stability) */
  double limit = 1.0 / sqrt((double)in);

  linear_layer.weights = create_matrix(out, in);
  fill_matrix(&linear_layer.weights, -limit, limit);

  linear_layer.biases = create_matrix(out, 1);
  for(uint64_t i = 0; i < out; i++) {
    linear_layer.biases.array[i] = 0.0;
  }

  /* Filled during forward_pass; keep empty to avoid freeing garbage. */
  linear_layer.zs.row_size = 0;
  linear_layer.zs.column_size = 0;
  linear_layer.zs.array = NULL;

  linear_layer.activations.row_size = 0;
  linear_layer.activations.column_size = 0;
  linear_layer.activations.array = NULL;

  linear_layer.z = linear_function;
  linear_layer.a = get_activation(activation);
  linear_layer.a_prime = get_derivative_activation(activation);

  return linear_layer;
}

void free_network_memory(neural_network* network) {
  if(!network) return;

  for(int i = 0; i < network->number_of_layers; i++) {
    if(network->layers[i].weights.array) free_matrix(&network->layers[i].weights);
    if(network->layers[i].biases.array) free_matrix(&network->layers[i].biases);
    if(network->layers[i].zs.array) free_matrix(&network->layers[i].zs);
    if(network->layers[i].activations.array) free_matrix(&network->layers[i].activations);
  }

  free(network->layers);
  network->layers = NULL;
  network->number_of_layers = 0;
}