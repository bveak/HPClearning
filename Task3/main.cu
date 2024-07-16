#include "sort.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>

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
    std::size_t n = 16 * 1024 * 1024;
    std::vector <int> a(n);
    std::mt19937 rnd(0x3ac2ed7b);
    for (int i = 0; i < n; ++i) 
        a[i] = rnd();
    warmup();
    float total_time = 0;
    int TestCount = 10;
    std::vector <int> b;
    for (int i = 0; i < TestCount; ++i) {
        b = a;
        sort(total_time, b);
    }
    printf("GPU average time(ms): %f\n", total_time / TestCount);
    time_t start = clock();
    std::sort(a.begin(), a.end());
    time_t end = clock();
    printf("CPU time(ms): %f\n", 1.0 * (end - start) / CLOCKS_PER_SEC * 1e3);
    assert(a == b);
    puts("Accepted!!");
    return 0;
}