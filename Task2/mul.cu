#include "mul.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

__global__ void mul(const float* A, const float* B, std::size_t N, std::size_t M, std::size_t K, float* C) {
    __shared__ float tmp[256];
    float res[8][8];
    int x = threadIdx.x, y = threadIdx.y;
    int X = blockIdx.x * (blockDim.x * 8), Y = blockIdx.y * (blockDim.y * 8);
    int id = x * 16 + y;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) 
            res[i][j] = 0;
    for (int i = 0; i < M; ++i) {
        tmp[id] = (id < 128? A[(X + id) * M + i]: B[i * K + Y + id - 128]);
        __syncthreads();
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 8; ++k)
                res[j][k] += tmp[x * 8 + j] * tmp[y * 8 + k + 128];
        __syncthreads();
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            C[(X + x * 8 + i) * K + Y + y * 8 + j] = res[i][j];
}

void matmul(float& total_time, const float* A, const float* B, std::size_t n, std::size_t m, std::size_t k, float* C) {
    float* a = NULL;
    assert(cudaMalloc((void**)&a, n * m * sizeof(float)) == cudaSuccess);
    float* b = NULL;
    assert(cudaMalloc((void**)&b, m * k * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(a, A, n * m * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(b, B, m * k * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    float* c = NULL;
    assert(cudaMalloc((void**)&c, n * k * sizeof(float)) == cudaSuccess);
    dim3 threadsPerBlock(16, 16);
    dim3 numblock((n - 1) / 128 + 1, (k - 1) / 128 + 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    mul<<<numblock, threadsPerBlock>>>(a, b, n, m, k, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    auto err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));
    assert(err == cudaSuccess);
    assert(cudaMemcpy(C, c, n * k * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(a) == cudaSuccess);
    assert(cudaFree(b) == cudaSuccess);
    assert(cudaFree(c) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
}