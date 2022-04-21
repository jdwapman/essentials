#include <cuda_runtime_api.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CHECK_CUDA(func)                                                  \
  {                                                                       \
    cudaError_t status = (func);                                          \
    if (status != cudaSuccess) {                                          \
      printf("CUDA API failed at file %s, line %d with error: %s (%d)\n", \
             __FILE__, __LINE__, cudaGetErrorString(status), status);     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

// Create an enum for L1_HIT, L2_HIT, and MISS
enum Cache_Hit { L1_HIT, L2_HIT, MISS, ERROR };

template <typename T>
__device__ __forceinline__ auto cache_read(T* ptr, size_t addr) {
  long long int start_time = clock64();
  T val = ptr[addr];
  __threadfence();
  long long int end_time = clock64();

  if (val < 0) {
    return ERROR;
  }

  double time = (end_time - start_time) / (double)CLOCKS_PER_SEC;

  if (time > 0.001) {
    return MISS;
  } else if (time < 0.00035) {
    return L1_HIT;
  } else {
    return L2_HIT;
  }
}

template <typename T>
__global__ void foo(T* cache_data, T* out, int size) {
  out[0] = 0;

  for (int i = 0; i < size && i < 100; i++) {
    auto hit_type = cache_read(cache_data, i);
    if (hit_type == L1_HIT) {
      printf("L1 HIT\n");
    } else if (hit_type == L2_HIT) {
      printf("L2 HIT\n");
    } else if (hit_type == MISS) {
      printf("MISS\n");
    } else if (hit_type == ERROR) {
      printf("ERROR\n");
    }
  }
}

int main(int argc, char** argv) {
  // Get the current CUDA device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Create a thrust vector equal to the size of the device's L2 cache
  thrust::device_vector<int> v(deviceProp.l2CacheSize);

  // Fill the vector with increasing numbers
  thrust::sequence(v.begin(), v.end());

  thrust::device_vector<int> out(1);

  foo<<<1, 1>>>(v.data().get(), out.data().get(), v.size());

  CHECK_CUDA(cudaDeviceSynchronize());

  thrust::host_vector<int> h_out(out.size());
  h_out = out;

  printf("%d\n", h_out[0]);
}