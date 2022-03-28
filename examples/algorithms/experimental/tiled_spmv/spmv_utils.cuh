#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>
#include <tuple>

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
void display(vector_t v, std::string name, bool verbose = true) {
  if (verbose) {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < v.size() && (i < 40); i++)
      std::cout << v[i] << " ";

    if (v.size() >= 40) {
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

void print_gpu_stats() {
  int device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  // Print the type of device
  printf("CUDA Device [%d]: \"%s\"\n", device, deviceProp.name);

  // Print the compute capability
  printf("- Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);

  // Print the number of multiprocessors
  printf("- Total number of SMs: %d\n", deviceProp.multiProcessorCount);

  // Print the amount of memory in GB
  printf("- Total amount of global memory: %.0f GB\n",
         deviceProp.totalGlobalMem * 1e-9f);

  // Print the amount of L2 cache
  printf("- L2 cache size: %.0f KB\n", deviceProp.l2CacheSize * 1e-3f);

  // Print the amount of persisting L2 cache
  printf("- Persisting L2 cache size: %.0f KB\n",
         deviceProp.persistingL2CacheMaxSize * 1e-3f);

  // Print the max policy window size
  printf("- Access policy max window size: %.0f KB\n",
         deviceProp.accessPolicyMaxWindowSize * 1e-3f);
  printf("  - %d float32s\n", deviceProp.accessPolicyMaxWindowSize / 4);
  printf("  - %d float64s\n", deviceProp.accessPolicyMaxWindowSize / 8);

  // Print the amount of shared memory available per block
  printf("- Shared memory available per block (default): %.0f KB\n",
         deviceProp.sharedMemPerBlock * 1e-3f);
  printf("- Shared memory available per block (extended): %.0f KB\n",
         deviceProp.sharedMemPerBlockOptin * 1e-3f);

  // Print the max number of threads per SM and block
  printf("- Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("- Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
}

template <typename vector_t>
cudaStream_t setup_ampere_cache(vector_t data) {
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

  // Setting the window size to 0 disable it
  stream_attribute.accessPolicyWindow.num_bytes = 0;

  // Overwrite the access policy attribute to a CUDA Stream
  CHECK_CUDA(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                         &stream_attribute));
  // Remove any persistent lines in L2
  CHECK_CUDA(cudaCtxResetPersistingL2Cache());
}