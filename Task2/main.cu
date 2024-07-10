#include "mul.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>
#include <algorithm>

__global__ void warmup_kernel() {
    int i = 1, j = 1;
    i += j;
}

void warmup() {
    for (int i = 0; i < 8; ++i)
        warmup_kernel<<<1024, 1024>>>();
}

int main() {
    warmup();
    std::size_t n = 1024, m = 1024, k = 1024;
    std::mt19937 rnd((0x3ac2ed7b));
    float* A = (float*)malloc(n * m * sizeof(float));
    for (int i = 0; i < n * n; ++i)
        A[i] = rnd() / 1e7;
    float* B = (float*)malloc(m * k * sizeof(float));
    for (int i = 0; i < n * n; ++i)
        B[i] = rnd() / 1e7;
    float* output = (float*)malloc(n * k * sizeof(float));
    float total_time = 0;
    int TestCount = 10;
    for (int i = 0; i < TestCount; ++i) {
        matmul(total_time, A, B, n, m, k, output);
    }
    printf("GPU average time(ms): %f\n", total_time / TestCount);
    float* ansput = (float*)malloc(n * k * sizeof(float));
    for (int i = 0; i < n * k; ++i)
        ansput[i] = 0;
    time_t start = clock();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j)
            for (int l = 0; l < k; ++l)
                ansput[i * k + l] += A[i * m + j] * B[j * k + l];
    }
    time_t end = clock();
    printf("CPU time(ms): %f\n", 1.0 * (end - start) / CLOCKS_PER_SEC * 1e3);
    for (int i = 0; i < n * k; ++i)
        assert(std::fabs((ansput[i] - output[i]) / ansput[i]) < 1e-5);
    puts("Accepted!!");
    free(A);
    free(B);
    free(ansput);
    free(output);
    return 0;
}