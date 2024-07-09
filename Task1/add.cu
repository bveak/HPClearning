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

void reduce(float& total_time, const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    float* input = NULL;
    assert(cudaMalloc((void**)&input, n * dim * sizeof(float)) == cudaSuccess);
    float* output = NULL;
    assert(cudaMalloc((void**)&output, dim * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(input, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    int threadsPerBlock = 1024;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_sum<<<(dim - 1) / threadsPerBlock + 1, threadsPerBlock>>>(input, n, dim, output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(output_vec, output, dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(input) == cudaSuccess);
    assert(cudaFree(output) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
}