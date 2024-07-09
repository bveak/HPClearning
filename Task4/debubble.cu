#include "debubble.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

__global__ void count(int* output, int* input, std::size_t n) {
    __shared__ int count[1024], arr[1024];
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        arr[tid] = input[blockIdx.x * (blockDim.x / 2) + tid / 2];
        count[tid] = arr[tid] != 0;
    }
    __syncthreads();
    for (int S = 2; S < blockDim.x; S *= 2) {
        int fr1 = tid & (S - 1);
        if ((tid & S) == S && fr1 < count[tid - fr1]) {
            int fr = tid & ~(S * 2 - 1);
            arr[fr + count[fr] + fr1] = arr[tid];
        }
        __syncthreads();
        if ((tid & (S * 2 - 1)) == 0)
            count[tid] += count[tid + S];
        __syncthreads();
    }
    if (tid < count[0])
        input[blockIdx.x * (blockDim.x / 2) + tid] = arr[tid];
    if (tid == 0) 
        output[blockIdx.x + 1] = count[0];
}

__global__ void bubbleCopy(int* siz, int* output, int* input, std::size_t n, std::size_t blocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < siz[blockIdx.x + 1] - siz[blockIdx.x])
        output[siz[blockIdx.x] + threadIdx.x] = input[i];
    if (i >= siz[blocks])
        output[i] = 0;
}

__global__ void prefix(int* siz, std::size_t n) {
    for (int i = 1; i < n; ++i)
        siz[i] += siz[i - 1];
}

std::size_t debubble(float& total_time, std::vector <int> &data) {
    int n = data.size();
    int* input = NULL;
    assert(cudaMalloc((void**)&input, n * sizeof(int)) == cudaSuccess);
    assert(cudaMemcpy(input, &data[0], n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    const int threadSize = 512;
    int blocks = (n - 1) / threadSize + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int* siz = NULL;
    assert(cudaMalloc((void**)&siz, (blocks + 1) * sizeof(int)) == cudaSuccess);
    count<<<blocks, 2 * threadSize>>>(siz, input, n);
    prefix<<<1, 1>>>(siz, blocks + 1);
    int ret = 0;
    assert(cudaMemcpy(&ret, siz + blocks, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    int* output = NULL;
    assert(cudaMalloc((void**)&output, n * sizeof(int)) == cudaSuccess);
    bubbleCopy<<<blocks, threadSize>>>(siz, output, input, n, blocks);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    assert(cudaGetLastError() == cudaSuccess);

    assert(cudaMemcpy(&data[0], output, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(input) == cudaSuccess);
    assert(cudaFree(output) == cudaSuccess);
    assert(cudaFree(siz) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
    return ret;
}