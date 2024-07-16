#ifndef DEBUBBLE_CUH
#define DEBUBBLE_CUH

#include <vector>
#include <cstddef>

const int multiProcessorCount = []() {
  int device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  return deviceProp.multiProcessorCount;
}();

std::size_t debubble(float&, std::vector <int> &data);

#endif