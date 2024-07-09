#include "debubble.cuh"
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
    std::size_t n = 64 * 1024 * 1024;
    std::vector <int> a(n);
    std::mt19937 rnd(0x3ac2ed7b);
    for (int i = 0; i < n; ++i) 
        a[i] = rnd();
    for (int i = rnd() % n; i; --i)
        a[rnd() % n] = 0;
    float total_time = 0;
    int TestCount = 10;
    std::vector <int> b;
    int gpucount = 0;
    for (int i = 0; i < TestCount; ++i) {
        b = a;
        gpucount = debubble(total_time, b);
    }
    printf("GPU average time(ms): %f\n", total_time / TestCount);
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