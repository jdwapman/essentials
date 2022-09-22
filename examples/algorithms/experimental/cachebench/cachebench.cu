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

__global__ void cache_sweep_kernel(int* d_in, int* d_out) {
  for (auto i = 0; i < AMPERE_L2_CACHE_LINE_SIZE / sizeof(int); i++) {
    cuda::annotated_ptr<int, cuda::access_property::normal> d_in_ptr(d_in);
    cuda::annotated_ptr<int, cuda::access_property::streaming> d_out_ptr(d_out);

    // Write without passing through L2 cache
    d_out_ptr[i] = d_in_ptr[i];
  }
}

void run(nvbench::state& state) {
  printf("Running benchmark\n");
  state.collect_l2_hit_rates();

  // Get the current CUDA device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  thrust::device_vector<int> d_in(AMPERE_L2_CACHE_LINE_SIZE / sizeof(int));
  thrust::device_vector<int> d_out(AMPERE_L2_CACHE_LINE_SIZE / sizeof(int));

  // Fill the vector with increasing numbers
  thrust::sequence(d_in.begin(), d_in.end());
  thrust::sequence(d_out.begin(), d_out.end());

  CHECK_CUDA(cudaCtxResetPersistingL2Cache());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cache_sweep_kernel<<<1, 1, 0>>>(d_in.data().get(), d_out.data().get());
  });
}

int main(int argc, char** argv) {
  NVBENCH_BENCH(run);

  return 0;
}
