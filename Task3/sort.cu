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

__global__ void bitonic_sort(int* arr, int i, int j) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int id_comp = id ^ j;
    if (id > id_comp)
        if ((arr[id] < arr[id_comp]) == !(id & i))
            swap(arr[id], arr[id_comp]);
}

__global__ void bitonic_sort_small(int* arr, int n) {
    __shared__ int a[1024];
    int id = threadIdx.x;
    a[id] = arr[blockIdx.x * blockDim.x + id];
    __syncthreads();
    for (int i = 2; i <= 1024; i *= 2) {
        bool chk = id & i;
        if (i == 1024) chk = blockIdx.x & 1;
        for (int j = i / 2; j; j /= 2) {
            int id_comp = id ^ j;
            if (id > id_comp)
                if ((a[id] < a[id_comp]) != chk)
                    swap(a[id], a[id_comp]);
            __syncthreads();
        }
    }
    arr[blockIdx.x * blockDim.x + id] = a[id];
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
    bitonic_sort_small<<<(n - 1) / threadSize + 1, threadSize>>>(input, n);
    for (int i = 2048; i <= n; i *= 2)
        for (int j = i / 2; j; j /= 2) 
            bitonic_sort<<<(n - 1) / threadSize + 1, threadSize>>>(input, i, j);
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