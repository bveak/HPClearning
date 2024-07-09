#include "add.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>

__global__ void warmup_kernel() {
    int i = 1, j = 1;
    i += j;
}

void warmup() {
    for (int i = 0; i < 8; ++i)
        warmup_kernel<<<1, 256>>>();
}

int main() {
    warmup();
    std::size_t n = 5000;
    std::size_t dim = 16384;
    std::mt19937 rnd((0x3ac2ed7b));
    float* input = (float*)malloc(n * dim * sizeof(float));
    for (int i = 0; i < n * dim; ++i)
        input[i] = rnd() / 1e4;
    float* output = (float*)malloc(dim * sizeof(float));
    float total_time = 0;
    int TestCount = 10;
    for (int i = 0; i < TestCount; ++i) {
        reduce(total_time, input, n, dim, output);
    }
    printf("GPU average time(ms): %f\n", total_time / TestCount);
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
    for (int i = 0; i < dim; ++i)
        assert(std::fabs(ansput[i] - output[i]) < 1e-5);
    puts("Accepted!!");
    free(input);
    free(output);
    return 0;
}