#ifndef MUL_CUH
#define MUL_CUH

#include <cstddef>

const int multiProcessorCount = []() {
  int device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  return deviceProp.multiProcessorCount;
}();

void matmul(float&, const float*, const float*, std::size_t, std::size_t, std::size_t, float*);

#endif