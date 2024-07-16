#ifndef SORT_CUH
#define SORT_CUH

#include <vector>
#include <cstddef>

const int multiProcessorCount = []() {
  int device = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  return deviceProp.multiProcessorCount;
}();

void sort(float&, std::vector <int> &nums);

#endif