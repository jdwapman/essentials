#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <tuple>

#include <nlohmann/json.hpp>

// for convenience
using json = nlohmann::json;

namespace cg = cooperative_groups;

#define CHECK_CUDA(func)                                                  \
  {                                                                       \
    cudaError_t status = (func);                                          \
    if (status != cudaSuccess) {                                          \
      printf("CUDA API failed at file %s, line %d with error: %s (%d)\n", \
             __FILE__, __LINE__, cudaGetErrorString(status), status);     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define print_device(fmt, ...)                   \
  {                                              \
    if (true) {                                  \
      if (blockIdx.x == 0 && threadIdx.x == 0) { \
        printf(fmt, __VA_ARGS__);                \
      }                                          \
      cg::grid_group grid = cg::this_grid();     \
      grid.sync();                               \
    }                                            \
  }

#define print_block(fmt, ...)     \
  {                               \
    if (true) {                   \
      if (threadIdx.x == 0) {     \
        printf(fmt, __VA_ARGS__); \
      }                           \
      __syncthreads();            \
    }                             \
  }

template <typename vector_t>
void display(vector_t v, std::string name, int count, bool verbose = true) {
  if (verbose) {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < v.size() && (i < count); i++)
      std::cout << v[i] << " ";

    if (v.size() >= count) {
      std::cout << "...";
    }
    std::cout << "]" << std::endl;
  }
}

// Helper function to concatenate tuples of tuples. For example:
// < <1, 2> > + < < 3, 4> > = < <1, 2>, <3, 4> >
// Notice how this keeps the tuples at the same level of nesting
template <typename T, typename... Ts>
auto concat(T t, Ts... ts) {
  return std::tuple_cat(std::make_tuple(t), concat(ts...));
}

// Apply an operation to a given tuple element at with the index selected at
// runtime Note the need for a tuple element, which is necessary if the lambda
// function is to change an internal member of the tuple when the function exits
template <typename func, size_t Idx = 0, typename... Ts>
__host__ __device__ __forceinline__ constexpr void
TupleRuntimeApply(func foo, size_t target_idx, std::tuple<Ts...>& tup) {
  if constexpr (Idx == sizeof...(Ts)) {
    return;
  } else {
    if (Idx == target_idx) {
      auto& x = std::get<Idx>(tup);
      foo(Idx, x);
    }

    // Recurse again
    TupleRuntimeApply<func, Idx + 1>(foo, target_idx, tup);
  }

  return;
}

// Return a value from the given tuple at an index selected at runtime
// Can also modify the tuple
template <size_t Idx = 0, typename... Ts>
__host__ __device__ __forceinline__ constexpr auto TupleReturnValue(
    size_t target_idx,
    const std::tuple<Ts...>& tup) {
  if constexpr (Idx == sizeof...(Ts)) {
    // Base case. Should never get here but just return the last element anyways
    auto& x = std::get<Idx - 1>(tup);
    auto retval_tup = x;
    return retval_tup;
  } else {
    // Extract the tuple index and evaluate
    auto x = std::get<Idx>(tup);
    auto retval_tup = x;

    // Recurse to get the next element.
    auto retval_recurse = TupleReturnValue<Idx + 1>(target_idx, tup);

    // Pick which one to return
    if (target_idx == Idx) {
      return retval_tup;
    } else {
      return retval_recurse;
    }
  }
}

// Iterate over all elements in a tuple and apply an operation to each
template <typename func, size_t Idx = 0, typename... Ts>
__host__ __device__ __forceinline__ constexpr void TupleForEach(
    func foo,
    std::tuple<Ts...>& tup) {
  if constexpr (Idx == sizeof...(Ts)) {
    // base case
    return;
  } else {
    // Extract the tuple index and evaluate
    auto& x = std::get<Idx>(tup);
    foo(Idx, x);

    // Going for next element.
    TupleForEach<func, Idx + 1>(foo, tup);
  }

  return;
}

// Performs reduction on tuples.
// Need two lambdas
// 1) Perform the reduction c = op(a, b)
// 2) Extract the value to reduce from the tuple
template <typename IdentityT,
          typename ExtractionFunc,
          typename ReductionFunc,
          size_t Idx = 0,
          typename... Ts>
__host__ __device__ __forceinline__ constexpr IdentityT TupleReduction(
    IdentityT identity,
    ExtractionFunc f_extract,
    ReductionFunc f_reduce,
    std::tuple<Ts...> tup) {
  if constexpr (Idx == sizeof...(Ts)) {
    // base case
    return identity;
  } else {
    // Extract the tuple index and evaluate
    auto x = std::get<Idx>(tup);
    IdentityT curr = f_extract(x);

    // Going for next element.
    IdentityT next =
        TupleReduction<IdentityT, ExtractionFunc, ReductionFunc, Idx + 1>(
            identity, f_extract, f_reduce, tup);

    return f_reduce(curr, next);
  }

  return identity;
}

void print_gpu_stats(json& _results) {
  // Get the current CUDA device
  int device;
  cudaGetDevice(&device);

  _results["gpustats"]["device"] = device;

  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Print the type of device
  printf("CUDA Device [%d]: \"%s\"\n", device, deviceProp.name);

  _results["gpustats"]["name"] = deviceProp.name;

  // Print the compute capability
  printf("- Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);

  _results["gpustats"]["compute_capability"] =
      std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);

  // Print the number of multiprocessors
  printf("- Total number of SMs: %d\n", deviceProp.multiProcessorCount);

  _results["gpustats"]["total_sms"] = deviceProp.multiProcessorCount;

  // Print the amount of memory in GB
  printf("- Total amount of global memory: %.0f GB\n",
         deviceProp.totalGlobalMem * 1e-9f);

  _results["gpustats"]["total_global_memory"] = deviceProp.totalGlobalMem;

  // Print the amount of L2 cache
  printf("- L2 cache size: %.0f KB\n", deviceProp.l2CacheSize * 1e-3f);

  _results["gpustats"]["l2_cache_size"] = deviceProp.l2CacheSize;

  // Print the amount of persisting L2 cache
  printf("- Persisting L2 cache size: %.0f KB\n",
         deviceProp.persistingL2CacheMaxSize * 1e-3f);

  _results["gpustats"]["persisting_l2_cache_size"] =
      deviceProp.persistingL2CacheMaxSize;

  // Print the max policy window size
  printf("- Access policy max window size: %.0f KB\n",
         deviceProp.accessPolicyMaxWindowSize * 1e-3f);
  printf("  - %d float32s\n", deviceProp.accessPolicyMaxWindowSize / 4);
  printf("  - %d float64s\n", deviceProp.accessPolicyMaxWindowSize / 8);

  _results["gpustats"]["access_policy_max_window_size"] =
      deviceProp.accessPolicyMaxWindowSize;

  // Print the amount of shared memory available per block
  printf("- Shared memory available per block (default): %.0f KB\n",
         deviceProp.sharedMemPerBlock * 1e-3f);

  _results["gpustats"]["shared_memory_per_block"] =
      deviceProp.sharedMemPerBlock;

  printf("- Shared memory available per block (extended): %.0f KB\n",
         deviceProp.sharedMemPerBlockOptin * 1e-3f);

  _results["gpustats"]["shared_memory_per_block_extended"] =
      deviceProp.sharedMemPerBlockOptin;

  _results["gpustats"]["shared_memory_per_multiprocessor"] =
      deviceProp.sharedMemPerMultiprocessor;

  // Print the max number of threads per SM and block
  printf("- Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("- Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

  _results["gpustats"]["max_threads_per_sm"] =
      deviceProp.maxThreadsPerMultiProcessor;
  _results["gpustats"]["max_threads_per_block"] = deviceProp.maxThreadsPerBlock;
}

template <typename vector_t>
cudaStream_t setup_ampere_cache(vector_t data, json& _results) {
  // The hitRatio parameter can be used to specify the fraction of accesses that
  // receive the hitProp property. In both of the examples above, 60% of the
  // memory accesses in the global memory region [ptr..ptr+num_bytes) have the
  // persisting property and 40% of the memory accesses have the streaming
  // property. Which specific memory accesses are classified as persisting (the
  // hitProp) is random with a probability of approximately hitRatio; the
  // probability distribution depends upon the hardware architecture and the
  // memory extent.

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#L2_access_intro

  // --
  // Set up cache configuration

  cudaStream_t stream;
  cudaStreamCreate(&stream);  // Create CUDA stream

  cudaDeviceProp prop;                // CUDA device properties variable
  cudaGetDeviceProperties(&prop, 0);  // Query GPU properties

  if (prop.major >= 8) {
    // Using Ampere

    /* Old way of doing this

    size_t size = min(int(prop.l2CacheSize), prop.persistingL2CacheMaxSize);
    CHECK_CUDA(
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           size));  // set-aside 100% of L2 cache for
                                    // persisting accesses or the max allowed

    size_t data_size_bytes = data.size() * sizeof(data[0]);

    size_t window_size =
        min((size_t)prop.accessPolicyMaxWindowSize,
            data_size_bytes);  // Select minimum of user defined
                               // num_bytes and max window size.

    cudaStreamAttrValue
        stream_attribute;  // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(
        data.data().get());  // Global Memory data pointer
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

    _results["ampere"]["persisting_access_ratio"] = 1.0;
    _results["ampere"]["streaming_access_ratio"] = 0.0;
    _results["ampere"]["persisting_access_size(bytes)"] = window_size;
    _results["ampere"]["data_size(bytes)"] = data_size_bytes;

    */

    // New way of doing this using annotated pointers

    } else {
    // Using Volta or below
    printf(
        "WARNING: L2 Cache Management available only for compute capabilities "
        ">= 8\n");
  }

  return stream;
}

template <typename stream_t>
void reset_ampere_cache(stream_t stream) {
  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;

  CHECK_CUDA(cudaStreamGetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow,
      &stream_attribute));  // Get the attributes from a CUDA Stream

  // Setting the window size to 0 disable it
  stream_attribute.accessPolicyWindow.num_bytes = 0;

  // Overwrite the access policy attribute to a CUDA Stream
  CHECK_CUDA(cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

  // Remove any persistent lines in L2
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
}

template <typename kernel_t>
int occupancy_shmem_bst(int min, int max, kernel_t kernel, int occupancy) {
  // Min is the minimum amount of shmem
  // Max is the maximum amount of shmem
  // Occupancy is the target occupancy

  int mid = (min + max) / 2;

  // Do a binary search
  if (min == max) {
    return min;
  }

  int occupancy_mid = 0;
  int occupancy_min = 0;
  int occupancy_max = 0;

  // Get occupancy for mid
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_mid,
                                                           kernel, 1, mid));

  // Get occupancy for min
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_min,
                                                           kernel, 1, min));

  // Get occupancy for max
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_max,
                                                           kernel, 1, max));

  if (occupancy_mid == occupancy) {
    return mid;
    // If the occupancy is too low, use a smaller amount of shmem
  } else if (occupancy_mid < occupancy) {
    return occupancy_shmem_bst(min, mid, kernel, occupancy);
  } else {
    // If the occupancy is too high, use more shmem
    return occupancy_shmem_bst(mid, max, kernel, occupancy);
  }
}

template <typename kernel_t>
int occupancy_threads_bst(int min,
                          int max,
                          int shmem,
                          kernel_t kernel,
                          int occupancy) {
  // Min is the minimum amount of threads
  // Max is the maximum amount of threads
  // Occupancy is the target occupancy

  int mid = (min + max) / 2;

  // Do a binary search
  if (min == max) {
    return min;
  }

  int occupancy_mid = 0;
  int occupancy_min = 0;
  int occupancy_max = 0;

  // Get occupancy for mid
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_mid,
                                                           kernel, mid, shmem));

  // Get occupancy for min
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_min,
                                                           kernel, min, shmem));

  // Get occupancy for max
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_max,
                                                           kernel, max, shmem));

  if (occupancy_mid == occupancy) {
    return mid;
    // If the occupancy is too low, use a smaller amount of shmem
  } else if (occupancy_mid < occupancy) {
    return occupancy_threads_bst(min, mid, shmem, kernel, occupancy);
  } else {
    // If the occupancy is too high, use more shmem
    return occupancy_threads_bst(mid, max, shmem, kernel, occupancy);
  }
}