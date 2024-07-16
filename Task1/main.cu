#include "add.cuh"
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
    for (int i = 0; i < 8; ++i) {
        warmup_kernel<<<1024, 1024>>>(test);
        assert(cudaGetLastError() == cudaSuccess);
    }
    assert(cudaMemcpy(rnd, test, 1024 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
}

int main() {
    std::size_t n = 50000 << 9;
    std::size_t dim = 32;
    std::mt19937 rnd((0x3ac2ed7b));
    float* input = (float*)malloc(n * dim * sizeof(float));
    for (int i = 0; i < n * dim; ++i)
        input[i] = rnd() / 1e4;
    float* output = (float*)malloc(dim * sizeof(float));
    float total_time = 0, max_time = 0, min_time = 1e5;
    warmup();
    int TestCount = 10;
    for (int i = 0; i < TestCount; ++i) {
        float current_time = 0;
        reduce(current_time, input, n, dim, output);
        max_time = std::max(max_time, current_time);
        min_time = std::min(min_time, current_time);
        total_time += current_time;
    }
    printf("GPU average time(ms): %f\n", (total_time - max_time - min_time) / (TestCount - 2));
    float* ansput = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; ++i)
        ansput[i] = 0;
    time_t start = clock();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j)
            ansput[j] += input[i * dim + j];
    }
    time_t end = clock();
    printf("CPU time(ms): %f\n", 1.0 * (end - start) / CLOCKS_PER_SEC * 1e3);
    // printf("%f %f\n", ansput[0], output[0]);
    for (int i = 0; i < dim; ++i)
        assert(std::fabs((ansput[i] - output[i]) / ansput[i]) < 1e-3);
    puts("Accepted!!");
    free(input);
    free(output);
    return 0;
}