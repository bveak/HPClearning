#include "add.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>

#define Float4(val) *(float4*)(&(val))

__global__ void reduce_sum(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    int i = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (i < dim) {
        float4 sum, a;
        sum.x = sum.y = sum.z = sum.w = 0;
        #pragma unroll
        for (int j = 0; j < n; ++j) {
            a = *((float4*)(input_vecs + j * dim + i));
            sum.x += a.x;
            sum.y += a.y;
            sum.z += a.z;
            sum.w += a.w;
        }
        *((float4*)(output_vec + i)) = sum;
    }
}

__inline__ __device__ float branch_warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void reduce_sum_dim1(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    __shared__ float sum;
    if (threadIdx.x == 0) sum = 0;
    __syncthreads();
    float local_sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        local_sum += __ldg(input_vecs + i);
    }
    local_sum = branch_warp_reduce_sum(local_sum);
    if (threadIdx.x % warpSize == 0) 
        atomicAdd(&sum, local_sum);
    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(output_vec, sum);
}

__global__ void reduce_sum_small(const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    const int TILE_SIZE = 32;
    __shared__ float sum[TILE_SIZE];
    if (threadIdx.x < TILE_SIZE)
        sum[threadIdx.x] = 0;
    __syncthreads();
    float local_sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * dim; i += blockDim.x * gridDim.x) {
        local_sum += __ldg(input_vecs + i);
    }
    atomicAdd(sum + threadIdx.x % TILE_SIZE, local_sum);
    __syncthreads();
    if (threadIdx.x < TILE_SIZE && dim <= threadIdx.x)
        atomicAdd(sum + threadIdx.x % dim, sum[threadIdx.x]);
    __syncthreads();
    if (threadIdx.x < dim)
        atomicAdd(output_vec + threadIdx.x, sum[threadIdx.x]);
}

void reduce(float& total_time, const float* input_vecs, std::size_t n, std::size_t dim, float* output_vec) {
    float* input = NULL;
    assert(cudaMalloc((void**)&input, n * dim * sizeof(float)) == cudaSuccess);
    float* output = NULL;
    assert(cudaMalloc((void**)&output, dim * sizeof(float)) == cudaSuccess);
    assert(cudaMemcpy(input, input_vecs, n * dim * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    int threadsPerBlock = 1024;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    assert(cudaMemset(output, 0, dim * sizeof(float)) == cudaSuccess);
    if (dim >= threadsPerBlock * 4) {
        reduce_sum<<<(dim - 1) / (threadsPerBlock * 4) + 1, threadsPerBlock>>>(input, n, dim, output);
    } else if (dim == 1) {
        reduce_sum_dim1<<<multiProcessorCount, threadsPerBlock>>>(input, n, dim, output);
    } else {
        if (dim > 32 || (dim & -dim) != dim) throw std::invalid_argument("Not impl");
        reduce_sum_small<<<multiProcessorCount, threadsPerBlock>>>(input, n, dim, output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaMemcpy(output_vec, output, dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(input) == cudaSuccess);
    assert(cudaFree(output) == cudaSuccess);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time(ms): %f\n", milliseconds);
    total_time += milliseconds;
}