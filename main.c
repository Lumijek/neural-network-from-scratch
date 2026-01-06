#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

#include "idx_loader.h"
#include "matrix.h"
#include "neural_network.h"

#define IMAGE_SIZE 28
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define MNIST_INPUTS (IMAGE_SIZE * IMAGE_SIZE)
#define MNIST_CLASSES 10

static uint8_t* train_image[TRAIN_SIZE];
static uint8_t* test_image[TEST_SIZE];
static uint8_t* train_label = NULL;
static uint8_t* test_label = NULL;

static idx_u8_images train_images_raw;
static idx_u8_images test_images_raw;
static idx_u8_labels train_labels_raw;
static idx_u8_labels test_labels_raw;

/**
 * Print an error message and exit.
 */
static void die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  exit(EXIT_FAILURE);
}

/**
 * Load MNIST from ./data using the generic IDX loader.
 * MNIST-specific filenames and expected shapes are kept in this file (main.c) only.
 */
static void load_mnist_data(void) {
  if (idx_read_u8_images("data/train-images-idx3-ubyte", &train_images_raw) != 0) die("Failed to read train images");
  if (idx_read_u8_labels("data/train-labels-idx1-ubyte", &train_labels_raw) != 0) die("Failed to read train labels");
  if (idx_read_u8_images("data/t10k-images-idx3-ubyte", &test_images_raw) != 0) die("Failed to read test images");
  if (idx_read_u8_labels("data/t10k-labels-idx1-ubyte", &test_labels_raw) != 0) die("Failed to read test labels");

  if (train_images_raw.count != TRAIN_SIZE || train_images_raw.rows != IMAGE_SIZE || train_images_raw.cols != IMAGE_SIZE) {
    die("Train images shape mismatch");
  }
  if (test_images_raw.count != TEST_SIZE || test_images_raw.rows != IMAGE_SIZE || test_images_raw.cols != IMAGE_SIZE) {
    die("Test images shape mismatch");
  }
  if (train_labels_raw.count != TRAIN_SIZE || test_labels_raw.count != TEST_SIZE) {
    die("Labels count mismatch");
  }

  train_label = train_labels_raw.data;
  test_label = test_labels_raw.data;

  for (uint32_t i = 0; i < TRAIN_SIZE; i++) {
    train_image[i] = train_images_raw.data + (uint64_t)i * MNIST_INPUTS;
  }
  for (uint32_t i = 0; i < TEST_SIZE; i++) {
    test_image[i] = test_images_raw.data + (uint64_t)i * MNIST_INPUTS;
  }
}

/**
 * Free MNIST buffers allocated by load_mnist_data().
 */
static void free_mnist_data(void) {
  idx_free(train_images_raw.data);
  idx_free(test_images_raw.data);
  idx_free(train_labels_raw.data);
  idx_free(test_labels_raw.data);
  memset(&train_images_raw, 0, sizeof(train_images_raw));
  memset(&test_images_raw, 0, sizeof(test_images_raw));
  memset(&train_labels_raw, 0, sizeof(train_labels_raw));
  memset(&test_labels_raw, 0, sizeof(test_labels_raw));
  train_label = NULL;
  test_label = NULL;
}

/**
 * Return the row index of the maximum value in a given column.
 */
static uint64_t argmax_col(const matrix* m, uint64_t col) {
  uint64_t best = 0;
  double bestv = m->array[col];
  for (uint64_t i = 1; i < m->row_size; i++) {
    double v = m->array[i * m->column_size + col];
    if (v > bestv) {
      bestv = v;
      best = i;
    }
  }
  return best;
}

/**
 * Fill an input batch matrix X and one-hot label matrix Y from an image/label dataset.
 * X has shape (MNIST_INPUTS x batch_size), Y has shape (MNIST_CLASSES x batch_size).
 */
static void build_batch_inputs(
  matrix* x,
  const uint8_t* const* images,
  const uint8_t* labels,
  const uint32_t* idx,
  uint32_t start,
  uint32_t bs,
  matrix* y_onehot
) {
  for (uint64_t i = 0; i < x->row_size * x->column_size; i++) x->array[i] = 0.0;
  for (uint64_t i = 0; i < y_onehot->row_size * y_onehot->column_size; i++) y_onehot->array[i] = 0.0;

  for (uint32_t col = 0; col < bs; col++) {
    uint32_t k = idx[start + col];
    const uint8_t* img = images[k];

    for (uint32_t p = 0; p < MNIST_INPUTS; p++) {
      x->array[p * bs + col] = (double)img[p] / 255.0;
    }

    uint8_t y = labels[k];
    if (y >= MNIST_CLASSES) die("Invalid label");
    y_onehot->array[(uint32_t)y * bs + col] = 1.0;
  }
}

/**
 * Fisher-Yates shuffle for an array of uint32_t.
 */
static void shuffle_u32(uint32_t* a, uint32_t n) {
  for (uint32_t i = n - 1; i > 0; i--) {
    uint32_t j = (uint32_t)(rand() % (i + 1));
    uint32_t t = a[i];
    a[i] = a[j];
    a[j] = t;
  }
}

/**
 * Evaluate test-set accuracy using the current model.
 */
static double eval_test_accuracy(neural_network* net, uint32_t batch_size) {
  uint64_t correct = 0;

  uint32_t* idx = (uint32_t*)malloc(sizeof(uint32_t) * TEST_SIZE);
  if (!idx) die("malloc failed");
  for (uint32_t i = 0; i < TEST_SIZE; i++) idx[i] = i;

  for (uint32_t start = 0; start < TEST_SIZE; start += batch_size) {
    uint32_t bs = batch_size;
    if (start + bs > TEST_SIZE) bs = TEST_SIZE - start;

    matrix x = create_matrix(MNIST_INPUTS, bs);
    matrix y = create_matrix(MNIST_CLASSES, bs);

    build_batch_inputs(&x, (const uint8_t* const*)test_image, test_label, idx, start, bs, &y);

    matrix out = forward_pass(net, &x);

    for (uint32_t col = 0; col < bs; col++) {
      uint64_t pred = argmax_col(&out, col);
      if ((uint8_t)pred == test_label[idx[start + col]]) correct++;
    }

    free_matrix(&x);
    free_matrix(&y);
  }

  free(idx);
  return (double)correct / (double)TEST_SIZE;
}

int main(int argc, char** argv) {
  uint32_t epochs = 5;
  uint32_t batch_size = 128;
  double lr = 0.01;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = (uint32_t)atoi(argv[++i]);
    else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch_size = (uint32_t)atoi(argv[++i]);
    else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = atof(argv[++i]);
    else {
      fprintf(stderr, "Usage: %s [--epochs N] [--batch N] [--lr X]\n", argv[0]);
      return 1;
    }
  }

  srand((unsigned)time(NULL));
  load_mnist_data();

  neural_network net = create_network();
  net.learning_rate = lr;

  add_layer(&net, linear(MNIST_INPUTS, 128, "relu"));
  add_layer(&net, linear(128, 64, "relu"));
  add_layer(&net, linear(64, MNIST_CLASSES, "softmax"));

  uint32_t* train_idx = (uint32_t*)malloc(sizeof(uint32_t) * TRAIN_SIZE);
  if (!train_idx) die("malloc failed");
  for (uint32_t i = 0; i < TRAIN_SIZE; i++) train_idx[i] = i;

  uint32_t steps_per_epoch = (TRAIN_SIZE + batch_size - 1) / batch_size;

  for (uint32_t e = 0; e < epochs; e++) {
    shuffle_u32(train_idx, TRAIN_SIZE);

    double epoch_loss = 0.0;

    for (uint32_t start = 0; start < TRAIN_SIZE; start += batch_size) {
      uint32_t bs = batch_size;
      if (start + bs > TRAIN_SIZE) bs = TRAIN_SIZE - start;

      matrix x = create_matrix(MNIST_INPUTS, bs);
      matrix y = create_matrix(MNIST_CLASSES, bs);

      build_batch_inputs(&x, (const uint8_t* const*)train_image, train_label, train_idx, start, bs, &y);

      matrix out = forward_pass(&net, &x);
      epoch_loss += cross_entropy(&out, &y);

      back_propagate(&net, &x, &y);

      free_matrix(&x);
      free_matrix(&y);
    }

    printf("epoch %u | loss %.6f | test acc %.4f\n", e + 1, epoch_loss / (double)steps_per_epoch, eval_test_accuracy(&net, batch_size));
  }

  free(train_idx);
  free_network_memory(&net);
  free_mnist_data();
  return 0;
}