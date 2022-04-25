#include <cuda_runtime_api.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "philox.cuh"

#define CHECK_CUDA(func)                                                  \
  {                                                                       \
    cudaError_t status = (func);                                          \
    if (status != cudaSuccess) {                                          \
      printf("CUDA API failed at file %s, line %d with error: %s (%d)\n", \
             __FILE__, __LINE__, cudaGetErrorString(status), status);     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

template <typename T>
__device__ __forceinline__ auto cache_read(volatile T* ptr, size_t addr) {
  long long int start_time = clock64();
  __threadfence();
  T val = ptr[addr];
  __threadfence();
  long long int end_time = clock64();

  if (val < 0) {
    return -1;
  }

  double time = (end_time - start_time) / (double)CLOCKS_PER_SEC;

  printf("%ld,%f\n", addr, time);
}

template <typename T>
__global__ void foo(T* cache_data,
                    size_t cache_data_size,
                    T* thrash_data,
                    size_t thrash_data_size,
                    T* out) {
  // What does this mean?
  // The NVIDIA Ampere architecture adds Compute Data Compression to accelerate
  // unstructured sparsity and other compressible data patterns. Compression in
  // L2 provides up to 4x improvement to DRAM read/write bandwidth, up to 4x
  // improvement in L2 read bandwidth, and up to 2x improvement in L2 capacity.
  // https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/

  printf("STEP1\n");
  // 1. Prime the cache by reading in all of the Ampere-pinned data.
  //    Since this is a miss in both L1 and L2, this should get serviced in
  //    32-byte chunks
  for (int i = 0; i < cache_data_size; i++) {
    cache_read(cache_data, i);
  }

  // 2. Read in a LOT of random values, such that the cache should be
  //    completely thrashed. Again, lots of misses in L2 so this should be
  //    serviced in 32-byte chunks
  // Philox_2x32 sampler;
  printf("STEP2\n");
  for (int i = 0; i < thrash_data_size; i++) {
    // auto data_addr = sampler.rand_int(i, 0, 0, thrash_data_size);
    // cache_read(thrash_data, data_addr);
    cache_read(thrash_data, i);
  }

  // 3. Read in the cache-pinned values again. These _should_ still be hits
  //    if everything is working correctly. Still 32-byte chunks, but with
  //    lower latencies since the cache is primed.
  printf("STEP3\n");
  for (int i = 0; i < cache_data_size; i++) {
    cache_read(cache_data, i);
  }
}

int main(int argc, char** argv) {
  // Get the current CUDA device
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Create a thrust vector equal to the size of the device's L2 cache
  // Divide because l2CacheSize is in bytes and we need to convert this to
  // elements
  thrust::device_vector<int> cache_data(deviceProp.l2CacheSize / sizeof(int));
  thrust::device_vector<int> thrash_data(cache_data.size() * 1);

  // Fill the vector with increasing numbers
  thrust::sequence(cache_data.begin(), cache_data.end());
  thrust::sequence(thrash_data.begin(), thrash_data.end());

  thrust::device_vector<int> out(1);

  // Set up ampere cache pinning
  cudaStream_t stream;
  cudaStreamCreate(&stream);  // Create CUDA stream

  size_t size =
      min(int(deviceProp.l2CacheSize), deviceProp.persistingL2CacheMaxSize);
  CHECK_CUDA(
      cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                         size));  // set-aside 100% of L2 cache for
                                  // persisting accesses or the max allowed

  size_t data_size_bytes = cache_data.size() * sizeof(cache_data[0]);

  size_t window_size = min((size_t)deviceProp.accessPolicyMaxWindowSize,
                           data_size_bytes);  // Select minimum of user defined
                                              // num_bytes and max window size.

  cudaStreamAttrValue
      stream_attribute;  // Stream level attributes data structure
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(
      cache_data.data().get());  // Global Memory data pointer
  stream_attribute.accessPolicyWindow.num_bytes =
      window_size;  // Number of bytes for persistence access
  stream_attribute.accessPolicyWindow.hitRatio =
      1.0;  // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitProp =
      cudaAccessPropertyPersisting;  // Persistence Property
  stream_attribute.accessPolicyWindow.missProp =
      cudaAccessPropertyStreaming;  // Type of access property on cache miss

  CHECK_CUDA(cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow,
      &stream_attribute));  // Set the attributes to a CUDA Stream

  CHECK_CUDA(cudaCtxResetPersistingL2Cache());

  foo<<<1, 1, 0, stream>>>(cache_data.data().get(), cache_data.size(),
                           thrash_data.data().get(), thrash_data.size(),
                           out.data().get());

  CHECK_CUDA(cudaDeviceSynchronize());

  thrust::host_vector<int> h_out(out.size());
  h_out = out;
}