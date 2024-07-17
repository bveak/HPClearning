#include "debubble.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void count(int* output, int* input, std::size_t n) {
    unsigned mask = __activemask();
    int tid = threadIdx.x;
    int rank = tid % warpSize;
    for (int pos = blockIdx.x * blockDim.x + tid; pos < n; pos += blockDim.x * gridDim.x) {
        bool nonzero = input[pos] != 0;
        unsigned state = __ballot_sync(mask, nonzero);
        if (rank == 0)
            output[pos / warpSize] = __popc(state);
    }
}

__global__ void bubbleCopy(int* siz, int* output, int* input, std::size_t n, std::size_t blocks) {
    unsigned mask = __activemask();
    int tid = threadIdx.x;
    int rank = tid % warpSize;
    for (int pos = blockIdx.x * blockDim.x + tid; pos < n; pos += blockDim.x * gridDim.x) {
        bool nonzero = input[pos] != 0;
        unsigned state = __ballot_sync(mask, nonzero);
        if (nonzero)
            output[siz[pos / warpSize] - __popc(state >> rank)] = input[pos];
    }
}

__global__ void prefix(int* siz, int* tmp, std::size_t n) {
    __shared__ int arr[8192];
    int tid = threadIdx.x;
    for (int pos = blockIdx.x * (blockDim.x * 8); pos < n; pos += (blockDim.x * 8) * gridDim.x) {
        #pragma unroll
        for (int i = 0; i < 8192; i += 1024)
            if (pos + i + tid < n)
                arr[i + tid] = siz[pos + i + tid];
            else arr[i + tid] = 0;
        __syncthreads();
        for (int j = 1; j <= 4096; j *= 2) {
            #pragma unroll
            for (int p = tid; p < 4096; p += 1024) {
                int lb = p & (j - 1);
                int u = (p ^ lb) << 1 | (j - 1), v = (p ^ lb) << 1 | j | lb;
                arr[v] += arr[u];
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < 8192; i += 1024)
            if (pos + i + tid < n)
                siz[pos + i + tid] = arr[i + tid];
        if (tid == 0) tmp[pos / 8192] = arr[8191];
    }
}

__global__ void shift(int* siz, int* tmp, std::size_t n) {
    int tid = threadIdx.x;
    for (int pos = (blockIdx.x + 1) * (blockDim.x * 8); pos < n; pos += (blockDim.x * 8) * gridDim.x) {
        #pragma unroll
        for (int i = 0; i < 8192; i += 1024)
            siz[pos + i + tid] += tmp[pos / 8192 - 1];
    }
}

void getprefix(int* arr, std::size_t n) {
    int totsiz = 1;
    for (int i = n / 8192; i; i /= 8192) totsiz += i;
    int* tmp = NULL;
    assert(cudaMalloc((void**)&tmp, totsiz * sizeof(int)) == cudaSuccess);
    auto get = [](auto &&get, int* arr, int* tmp, std::size_t n) -> void {
        prefix<<<multiProcessorCount, 1024>>>(arr, tmp, n);
        if (n > 8192) {
            get(get, tmp, tmp + n / 8192, n / 8192);
            shift<<<multiProcessorCount, 1024>>>(arr, tmp, n);
        }
    };
    get(get, arr, tmp, n);
    assert(cudaFree(tmp) == cudaSuccess);
}

std::size_t debubble(float& total_time, std::vector <int> &data) {
    int n = data.size();
    int* input = NULL;
    assert(cudaMalloc((void**)&input, n * sizeof(int)) == cudaSuccess);
    assert(cudaMemcpy(input, &data[0], n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    const int threadSize = 1024;
    int blocks = (n - 1) / 32 + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int* output = NULL;
    assert(cudaMalloc((void**)&output, n * sizeof(int)) == cudaSuccess);
    int* siz = NULL;
    assert(cudaMalloc((void**)&siz, (blocks + 1) * sizeof(int)) == cudaSuccess);
    count<<<multiProcessorCount, threadSize>>>(siz, input, n);
    getprefix(siz, blocks);
    bubbleCopy<<<multiProcessorCount, threadSize>>>(siz, output, input, n, blocks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(&data[0], output, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    int ret = 0;
    for (int i = std::__lg(n - 1); i >= 0; --i)
        if (data[ret + (1 << i)])
            ret += 1 << i;
    ++ret;

    assert(cudaFree(input) == cudaSuccess);
    assert(cudaFree(output) == cudaSuccess);
    assert(cudaFree(siz) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
    return ret;
}