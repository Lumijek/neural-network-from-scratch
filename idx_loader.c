#include "idx_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32_be(FILE* f, uint32_t* out) {
  uint8_t b[4];
  if (fread(b, 1, 4, f) != 4) return 1;
  *out = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
  return 0;
}

void idx_free(void* p) {
  free(p);
}

int idx_read_u8_images(const char* path, idx_u8_images* out) {
  if (!out) return 1;
  memset(out, 0, sizeof(*out));

  FILE* f = fopen(path, "rb");
  if (!f) return 2;

  uint32_t magic = 0, count = 0, rows = 0, cols = 0;
  if (read_u32_be(f, &magic) || read_u32_be(f, &count) || read_u32_be(f, &rows) || read_u32_be(f, &cols)) {
    fclose(f);
    return 3;
  }

  /* 0x00000803 = unsigned byte, 3 dimensions */
  if (magic != 0x00000803) {
    fclose(f);
    return 4;
  }

  uint64_t n = (uint64_t)count * (uint64_t)rows * (uint64_t)cols;
  uint8_t* data = (uint8_t*)malloc((size_t)n);
  if (!data) {
    fclose(f);
    return 5;
  }

  if (fread(data, 1, (size_t)n, f) != (size_t)n) {
    free(data);
    fclose(f);
    return 6;
  }

  fclose(f);

  out->data = data;
  out->count = count;
  out->rows = rows;
  out->cols = cols;
  return 0;
}

int idx_read_u8_labels(const char* path, idx_u8_labels* out) {
  if (!out) return 1;
  memset(out, 0, sizeof(*out));

  FILE* f = fopen(path, "rb");
  if (!f) return 2;

  uint32_t magic = 0, count = 0;
  if (read_u32_be(f, &magic) || read_u32_be(f, &count)) {
    fclose(f);
    return 3;
  }

  /* 0x00000801 = unsigned byte, 1 dimension */
  if (magic != 0x00000801) {
    fclose(f);
    return 4;
  }

  uint8_t* data = (uint8_t*)malloc((size_t)count);
  if (!data) {
    fclose(f);
    return 5;
  }

  if (fread(data, 1, (size_t)count, f) != (size_t)count) {
    free(data);
    fclose(f);
    return 6;
  }

  fclose(f);

  out->data = data;
  out->count = count;
  return 0;
}
