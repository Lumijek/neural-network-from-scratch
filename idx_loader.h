#ifndef IDX_LOADER_H_
#define IDX_LOADER_H_

#include <stdint.h>

typedef struct {
  uint8_t* data;
  uint32_t count;
  uint32_t rows;
  uint32_t cols;
} idx_u8_images;

typedef struct {
  uint8_t* data;
  uint32_t count;
} idx_u8_labels;

/**
 * Read an IDX file containing unsigned-byte images (3D tensor: count x rows x cols).
 * On success, allocates a contiguous buffer into out->data (caller must free with idx_free()).
 * Returns 0 on success, nonzero on failure.
 */
int idx_read_u8_images(const char* path, idx_u8_images* out);

/**
 * Read an IDX file containing unsigned-byte labels (1D tensor: count).
 * On success, allocates a contiguous buffer into out->data (caller must free with idx_free()).
 * Returns 0 on success, nonzero on failure.
 */
int idx_read_u8_labels(const char* path, idx_u8_labels* out);

/**
 * Free memory returned by idx_read_u8_images / idx_read_u8_labels.
 * Safe to call with NULL.
 */
void idx_free(void* p);

#endif