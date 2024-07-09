#ifndef ADD_CUH
#define ADD_CUH

#include <cstddef>

__global__ void reduce_sum(const float*, std::size_t, std::size_t, float*);

void reduce(float&, const float*, std::size_t, std::size_t, float*);

#endif