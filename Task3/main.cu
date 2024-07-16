#include "sort.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void warmup_kernel(int* test) {
    int i = 1, j = 1;
    i += j;
    if (blockIdx.x == threadIdx.x)
        ++test[blockIdx.x];
}

void warmup() {
    int* test = NULL;
    assert(cudaMalloc((void**)&test, 1024 * sizeof(int)) == cudaSuccess);
    int* rnd = (int*)malloc(1024 * sizeof(int));
    unsigned rndval = 1;
    for (int i = 0; i < 1024; ++i) {
        rnd[i] = rndval;
        rndval ^= rndval << 3;
        rndval ^= rndval >> 5;
        rndval ^= rndval << 17;
    }
    assert(cudaMemcpy(test, rnd, 1024 * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 0; i < 8; ++i)
        warmup_kernel<<<1024, 1024>>>(test);
    assert(cudaMemcpy(rnd, test, 1024 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
}
int main() {
    std::size_t n = 1 << 28;
    std::vector <int> a(n);
    std::mt19937 rnd(0x3ac2ed7b);
    for (int i = 0; i < n; ++i) 
        a[i] = rnd();
    warmup();
    float total_time = 0, max_time = 0, min_time = 1e5;
    int TestCount = 10;
    std::vector <int> b;
    for (int i = 0; i < TestCount; ++i) {
        b = a;
        float current_time = 0;
        sort(current_time, b);
        max_time = std::max(max_time, current_time);
        min_time = std::min(min_time, current_time);
        total_time += current_time;
    }
    printf("GPU average time(ms): %f\n", (total_time - max_time - min_time) / (TestCount - 2));
    total_time = 0, max_time = 0, min_time = 1e5;
    std::vector <int> c = a;
    for (int i = 0; i < TestCount; ++i) {
        // thrust::host_vector <int> arr = a;
        thrust::device_vector <int> arr = a;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        thrust::sort(arr.begin(), arr.end());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        assert(cudaGetLastError() == cudaSuccess);
        thrust::copy(arr.begin(), arr.end(), c.begin());
        float current_time = 0;
        cudaEventElapsedTime(&current_time, start, stop);
        max_time = std::max(max_time, current_time);
        min_time = std::min(min_time, current_time);
        total_time += current_time;
    }
    printf("thrust average time(ms): %f\n", (total_time - max_time - min_time) / (TestCount - 2));
    // time_t start = clock();
    // std::sort(a.begin(), a.end());
    // time_t end = clock();
    // printf("CPU time(ms): %f\n", 1.0 * (end - start) / CLOCKS_PER_SEC * 1e3);
    printf("CPU ommited\n");
    assert(b == c);
    puts("Accepted!!");
    return 0;
}