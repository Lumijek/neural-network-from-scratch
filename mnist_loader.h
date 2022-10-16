#ifndef MNIST_LOADER_H_
#define MNIST_LOADER_H_

#define SWAP_UINT32(x) (((x) >> 24) | (((x) & 0x00FF0000) >> 8) | (((x) & 0x0000FF00) << 8) | ((x) << 24))//#include "mnist_loader.h"
#define IMAGE_SIZE 28
#define TEST_SIZE 10000
#define TRAIN_SIZE 60000

uint8_t* train_image[TRAIN_SIZE];
uint8_t train_label[TRAIN_SIZE];

uint8_t* test_image[TEST_SIZE];
uint8_t test_label[TEST_SIZE];

void load_test_labels(uint8_t* test_label_array){
	FILE* stream;
	stream = fopen( "data/t10k-labels-idx1-ubyte", "rb");
	uint32_t magic;
	uint32_t label_size;
	fread(&magic, sizeof(uint32_t), 1, stream);
	fread(&label_size, sizeof(uint32_t), 1, stream);

	fread(test_label_array, 1, label_size, stream);

	fclose(stream);
}
void load_test_data(uint8_t** test_data_array){
	FILE* stream;
	stream = fopen( "data/t10k-images-idx3-ubyte", "rb");
	uint32_t magic;
	uint32_t data_size;
	uint32_t rows;
	uint32_t columns;
	fread(&magic, sizeof(uint32_t), 1, stream);
	fread(&data_size, sizeof(uint32_t), 1, stream);
	fread(&rows, sizeof(uint32_t), 1, stream);
	fread(&columns, sizeof(uint32_t), 1, stream);

	for(int i = 0; i < TEST_SIZE; i++) {
		test_data_array[i] = malloc(sizeof(uint8_t) * IMAGE_SIZE * IMAGE_SIZE);
		fread(test_data_array[i], 1, IMAGE_SIZE * IMAGE_SIZE, stream);
	}

	fclose(stream);
}

void load_train_labels(uint8_t* train_label_array){
	FILE* stream;
	stream = fopen( "data/train-labels-idx1-ubyte", "rb");
	uint32_t magic;
	uint32_t label_size;
	fread(&magic, sizeof(uint32_t), 1, stream);
	fread(&label_size, sizeof(uint32_t), 1, stream);

	fread(train_label_array, 1, label_size, stream);

	fclose(stream);
}
void load_train_data(uint8_t** train_data_array){
	FILE* stream;
	stream = fopen( "data/train-images-idx3-ubyte", "rb");
	uint32_t magic;
	uint32_t data_size;
	uint32_t rows;
	uint32_t columns;
	fread(&magic, sizeof(uint32_t), 1, stream);
	fread(&data_size, sizeof(uint32_t), 1, stream);
	fread(&rows, sizeof(uint32_t), 1, stream);
	fread(&columns, sizeof(uint32_t), 1, stream);

	for(int i = 0; i < TRAIN_SIZE; i++) {
		train_data_array[i] = malloc(sizeof(uint8_t) * IMAGE_SIZE * IMAGE_SIZE);
		fread(train_data_array[i], 1, IMAGE_SIZE * IMAGE_SIZE, stream);
	}

	fclose(stream);
}

void load_mnist_data() {
   load_train_data(train_image);
   load_train_labels(train_label);
   load_test_data(test_image);
   load_test_labels(test_label);
}

#endif