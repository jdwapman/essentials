

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../tiled_spmv/spmv_utils.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda/annotated_ptr>
#include <cooperative_groups.h>
#include <thrust/sequence.h>
#include <nvbench/nvbench.cuh>

// Set clock64_t as long long int
typedef long long int clock64_t;

template <typename T>
__device__ __forceinline__ auto cache_read(T ptr, uint32_t addr) {
  clock64_t start_time;
  clock64_t end_time;
  uint32_t data;
  uint32_t accum = 0;

  // Start the clock
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_time)::"memory");

  data = ptr[addr];

  __threadfence();

  accum += data;

  // Insert an asm volatile instruction that gets the clock64 register
  // value and stores it in the global variable
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_time)::"memory");

  // Subtract 2 cycles for the add
  printf("%d,%d,%d,%ld\n", addr, data, accum, end_time - start_time - 2);
}

#define AMPERE_L2_CACHE_LINE_SIZE 128
constexpr int NUM_LINES = 1 << 14;

__global__ void cache_sweep_kernel(volatile int* d_in,  int* d_out) {
  int sum = 0;
  // cuda::annotated_ptr<int, cuda::access_property::persisting> d_in_ptr(d_in);
  cuda::annotated_ptr<int, cuda::access_property::streaming> d_out_ptr(d_out);

  // Global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Global thread stride
  int stride = blockDim.x * gridDim.x;

  for (auto i = tid; i < NUM_LINES * AMPERE_L2_CACHE_LINE_SIZE / sizeof(int);
       i += stride) {
    // printf("d_in_ptr[%d] = %d", i, d_in_ptr[i]);
    // cache_read(d_in, i);
    sum += d_in[0];
  }

  d_out_ptr[0] = sum;
}

void cachebench(nvbench::state& state) {
  printf("Running benchmark with %d lines\n", NUM_LINES);
  state.collect_l2_hit_rates();

  // Get the current CUDA device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  thrust::device_vector<int> d_in(NUM_LINES * AMPERE_L2_CACHE_LINE_SIZE /
                                  sizeof(int));
  thrust::device_vector<int> d_out(NUM_LINES * AMPERE_L2_CACHE_LINE_SIZE /
                                   sizeof(int));

  // Fill the vector with increasing numbers
  thrust::sequence(d_in.begin(), d_in.end());
  thrust::sequence(d_out.begin(), d_out.end());

  CHECK_CUDA(cudaCtxResetPersistingL2Cache());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cache_sweep_kernel<<<1, 1, 0>>>(d_in.data().get(), d_out.data().get());
  });
}

int main(int argc, char** argv) {
  NVBENCH_BENCH(cachebench);
  NVBENCH_MAIN_BODY(argc, argv);
}
