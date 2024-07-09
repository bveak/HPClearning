#include "sort.cuh"
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
    std::size_t n = 16 * 1024 * 1024;
    std::vector <int> a(n);
    std::mt19937 rnd(0x3ac2ed7b);
    for (int i = 0; i < n; ++i) 
        a[i] = rnd();
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