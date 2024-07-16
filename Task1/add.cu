#include "add.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

#define Float4(val) *(float4*)(&(val))

__global__ void reduce_sum(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    int i = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (i < dim) {
        float4 sum, a;
        sum.x = sum.y = sum.z = sum.w = 0;
        for (int j = 0; j < n; ++j) {
            a = Float4(input_vecs[j * dim + i]);
            sum.x += a.x;
            sum.y += a.y;
            sum.z += a.z;
            sum.w += a.w;
        }
        Float4(output_vec[i]) = sum;
    }
}

void reduce(float& total_time, const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    float* input = NULL;
    assert(cudaMalloc((void**)&input, n * dim * sizeof(float)) == cudaSuccess);
    float* output = NULL;
    assert(cudaMalloc((void**)&output, dim * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(input, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    int threadsPerBlock = 32;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_sum<<<(dim - 1) / (threadsPerBlock / 4) + 1, threadsPerBlock>>>(input, n, dim, output);
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