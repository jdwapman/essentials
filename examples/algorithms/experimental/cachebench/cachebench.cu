

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

#define AMPERE_L2_CACHE_LINE_SIZE 128
constexpr int NUM_LINES = 1 << 14;

__global__ void cache_sweep_kernel(volatile int* d_in, int* d_out) {
  int sum = 0;
  // cuda::annotated_ptr<int, cuda::access_property::persisting> d_in_ptr(d_in);
  cuda::annotated_ptr<int, cuda::access_property::streaming> d_out_ptr(d_out);

  // Global thread index
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Global thread stride
  int stride = blockDim.x * gridDim.x;

  for (auto i = tid; i < NUM_LINES * AMPERE_L2_CACHE_LINE_SIZE / sizeof(int);
       i += stride) {
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
  thrust::device_vector<int> d_out(1);

  // Fill the vector with increasing numbers
  thrust::sequence(d_in.begin(), d_in.end());

  CHECK_CUDA(cudaCtxResetPersistingL2Cache());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cache_sweep_kernel<<<1, 1, 0>>>(d_in.data().get(), d_out.data().get());
  });

  thrust::host_vector<int> h_out = d_out;
  printf("Final Value: %d\n", h_out[0]);
}

int main(int argc, char** argv) {
  NVBENCH_BENCH(cachebench);
  NVBENCH_MAIN_BODY(argc, argv);
}
