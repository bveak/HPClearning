#include "add.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

__global__ void reduce_sum(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dim) {
        output_vec[i] = 0;
        for (int j = 0; j < n; ++j) 
            output_vec[i] += input_vecs[j * dim + i];
    }
}

void reduce(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    float* input = NULL;
    assert(cudaMalloc((void**)&input, n * dim * sizeof(float)) == cudaSuccess);
    float* output = NULL;
    assert(cudaMalloc((void**)&output, dim * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(input, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    int threadsPerBlock = 1024;
    reduce_sum<<<(dim - 1) / threadsPerBlock + 1, threadsPerBlock>>>(input, n, dim, output);
    assert(cudaGetLastError() == cudaSuccess);
    auto err = cudaMemcpy(output_vec, output, dim * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        assert(err != cudaSuccess);
        fprintf(stderr, "Failed to reduce sum (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // assert(cudaMemcpy(output_vec, output, dim, cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(input) == cudaSuccess);
    assert(cudaFree(output) == cudaSuccess);
}