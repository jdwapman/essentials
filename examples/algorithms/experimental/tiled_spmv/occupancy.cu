#include <cuda_runtime_api.h>
#include <iostream>

#define CHECK_CUDA(func)                                                  \
  {                                                                       \
    cudaError_t status = (func);                                          \
    if (status != cudaSuccess) {                                          \
      printf("CUDA API failed at file %s, line %d with error: %s (%d)\n", \
             __FILE__, __LINE__, cudaGetErrorString(status), status);     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

__global__ void foo() {
  int temp[33];
}

int main(int argc, char** argv) {
  auto device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));

  // Setup grid and block properties
  auto numBlocksPerSm = 0;
  auto numThreadsPerBlock = 0;
  int shmemPerBlock = 0;  // bytes

  // Use the max number of threads per block to maximize parallelism over
  // shmem
  auto target_occupancy = 2;
  numThreadsPerBlock = deviceProp.maxThreadsPerBlock / target_occupancy;
  shmemPerBlock =
      (deviceProp.sharedMemPerBlockOptin - 1024 * target_occupancy) /
      target_occupancy;

  int carveout = 100;
  CHECK_CUDA(cudaFuncSetAttribute(
      foo, cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

  std::cout << "Threads Per Block: " << numThreadsPerBlock << std::endl;
  std::cout << "Shmem Per Block (bytes): " << shmemPerBlock << std::endl;

  CHECK_CUDA(cudaFuncSetAttribute(
      foo, cudaFuncAttributeMaxDynamicSharedMemorySize, shmemPerBlock));

  // Need to know the max occupancy to determine how many blocks to launch
  // for the cooperative kernel. All blocks must be resident on SMs
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, foo, numThreadsPerBlock, shmemPerBlock))

  // See how many registers the kernel uses
  cudaFuncAttributes attr;
  CHECK_CUDA(cudaFuncGetAttributes(&attr, foo));

  std::cout << "Registers: " << attr.numRegs << std::endl;

  std::cout << "Max Active Blocks Per SM: " << numBlocksPerSm << std::endl;

  foo<<<1024, numThreadsPerBlock>>>();

  CHECK_CUDA(cudaDeviceSynchronize());
}