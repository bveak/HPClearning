#include "add.cuh"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>

int main() {
    std::size_t n = 5000;
    std::size_t dim = 16384;
    std::mt19937 rnd((0x3ac2ed7b));
    float* input = (float*)malloc(n * dim * sizeof(float));
    for (int i = 0; i < n * dim; ++i)
        input[i] = rnd() / 1e4;
    float* output = (float*)malloc(dim * sizeof(float));
    reduce(input, n, dim, output);
    for (int i = 0; i < dim; ++i) {
        float ans = 0;
        for (int j = 0; j < n; ++j)
            ans += input[j * dim + i];
        assert(std::fabs(ans - output[i]) <= 1e-5);
    }
    puts("Accepted!!");
    free(input);
    free(output);
    return 0;
}