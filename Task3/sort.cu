#include "sort.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

__device__ void swap(int& a, int& b) {
    // int t = a;
    // a = b;
    // b = t;
    a ^= b ^= a ^= b;
}

__global__ void bitonic_sort_simple(int* arr, int i, int j, int n) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < n; id += blockDim.x * gridDim.x) {
        int id_comp = id ^ j;
        if (id > id_comp)
            if ((arr[id] < arr[id_comp]) == !(id & i))
                swap(arr[id], arr[id_comp]);
    }
}

__global__ void bitonic_sort_small(int* arr, int n) {
    __shared__ int a[8192];
    int id = threadIdx.x;
    for (int pos = blockIdx.x * (blockDim.x * 8); pos < n; pos += (blockDim.x * 8) * gridDim.x) {
        #pragma unroll
        for (int j = 0; j < 8192; j += 1024)
            a[id + j] = arr[pos + id + j];
        __syncthreads();
        for (int i = 2; i <= 8192; i *= 2) {
            #pragma unroll
            for (int j = i / 2; j; j /= 2) {
                #pragma unroll
                for (int p = id; p < 4096; p += 1024) {
                    int lb = p & (j - 1);
                    int u = (p ^ lb) << 1 | lb, v = u | j;
                    if ((a[v] < a[u]) == !((u | pos) & i)) 
                        swap(a[v], a[u]);
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for (int j = 0; j < 8192; j += 1024)
            arr[pos + id + j] = a[id + j];
    }
}

__global__ void bitonic_sort_large(int* arr, int i, int n) {
    __shared__ int a[8192];
    int id = threadIdx.x;
    for (int pos = blockIdx.x * (blockDim.x * 8); pos < n; pos += (blockDim.x * 8) * gridDim.x) {
        #pragma unroll
        for (int j = 0; j < 8192; j += 1024)
            a[id + j] = arr[pos + id + j];
        __syncthreads();
        #pragma unroll
        for (int j = 4096; j; j /= 2) {
            #pragma unroll
            for (int p = id; p < 4096; p += 1024) {
                int lb = p & (j - 1);
                int u = (p ^ lb) << 1 | lb, v = u | j;
                if ((a[v] < a[u]) == !((u | pos) & i)) 
                    swap(a[v], a[u]);
            }
            __syncthreads();
        }
        #pragma unroll
        for (int j = 0; j < 8192; j += 1024)
            arr[pos + id + j] = a[id + j];
    }
}

void sort(float& total_time, std::vector <int> &nums) {
    int n = nums.size();
    int* input = NULL;
    assert(cudaMalloc((void**)&input, n * sizeof(int)) == cudaSuccess);
    assert(cudaMemcpy(input, &nums[0], n * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    const int threadSize = 1024;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    bitonic_sort_small<<<multiProcessorCount, threadSize>>>(input, n);
    for (int i = 16384; i <= n; i *= 2) {
        for (int j = i / 2; j >= 8192; j /= 2) 
            bitonic_sort_simple<<<multiProcessorCount, threadSize>>>(input, i, j, n);
        bitonic_sort_large<<<multiProcessorCount, threadSize>>>(input, i, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(&nums[0], input, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(input) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
}