#include "debubble.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>

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
    for (int i = rnd() % n; i; --i)
        a[rnd() % n] = 0;
    warmup();
    float total_time = 0, max_time = 0, min_time = 1e5;
    int TestCount = 10;
    std::vector <int> b;
    int gpucount = 0;
    for (int i = 0; i < TestCount; ++i) {
        b = a;
        float current_time = 0;
        gpucount = debubble(current_time, b);
        max_time = std::max(max_time, current_time);
        min_time = std::min(min_time, current_time);
        total_time += current_time;
    }
    printf("GPU average time(ms): %f\n", (total_time - max_time - min_time) / (TestCount - 2));
    time_t start = clock();
    int count = 0;
    for (int i = 0; i < n; ++i)
        if (a[i] != 0)
            a[count++] = a[i];
    for (int i = count; i < n; ++i)
        a[i] = 0;
    time_t end = clock();
    printf("CPU time(ms): %f\n", 1.0 * (end - start) / CLOCKS_PER_SEC * 1e3);
    assert(count == gpucount);
    // for (int i = 0; i < n; ++i) printf("%d ", a[i]);
    // puts("");
    // for (int i = 0; i < n; ++i) printf("%d ", b[i]);
    // puts("");
    for (int i = 0; i < n; ++i)
        assert(a[i] == b[i]);
    puts("Accepted!!");
    return 0;
}