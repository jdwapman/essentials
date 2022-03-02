#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "spmv_utils.cuh"

namespace cg = cooperative_groups;

using namespace gunrock;
using namespace memory;

// template <typename index_t>
// __device__ __forceinline__ index_t global2tile(const index_t &global_idx,
//                                                const index_t &tile_size)
// {
//   // Note that this function assumes a valid global index and does not
//   attempt
//   // to perform bounds checking for partial tiles

//   // Tile_index + offset_within_tile
//   index_t local_idx = (global_idx / tile_size) + (global_idx % tile_size);

//   return local_idx;
// }

// template <typename index_t>
// __device__ __forceinline__ index_t tile2global(const index_t &local_idx,
//                                                const index_t &tile_idx,
//                                                const index_t &tile_size)
// {
//   // Note that this function does not attempt to perform bounds checking for
//   the
//   // final tile

//   index_t global_idx = local_idx + (tile_idx * tile_size);

//   return global_idx;
// }

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

template <typename graph_t, typename vector_t, typename shmem_t>
class TileIterator {
 public:
  __device__ TileIterator(const graph_t _graph,
                          const vector_t* _input,
                          vector_t* _output,
                          int* _queue_counter,
                          const size_t _tile_row_size,
                          const size_t _tile_col_size,
                          shmem_t* _shmem)
      : graph(_graph),
        input(_input),
        output(_output),
        queue_counter(_queue_counter),
        tile_row_size(_tile_row_size),
        tile_col_size(_tile_col_size),
        shmem(_shmem) {
    cur_tile_col_idx = 0;
    cur_tile_row_idx = blockIdx.x;

    num_row_tiles = graph.get_number_of_rows() / tile_row_size;

    // Handle the remainder
    if (graph.get_number_of_rows() % tile_row_size != 0) {
      num_row_tiles++;
    }

    num_col_tiles = graph.get_number_of_columns() / tile_col_size;

    if (graph.get_number_of_columns() % tile_col_size != 0) {
      num_col_tiles++;
    }

    // Reset the row tile queue counter
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      queue_counter[0] = blockDim.x;
    }

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();
  }

  __device__ __forceinline__ bool all_columns_finished() {
    if (cur_tile_col_idx >= num_col_tiles) {
      return true;
    }

    return false;
  }

  __device__ __forceinline__ void process_block_row_tile() {
    print_block(" - Processing block row tile %d\n", (int)cur_tile_row_idx);
  }

  __device__ __forceinline__ void get_next_block_row_tile() {
    // Atomically increment the current tile row index
    // Note that this needs to be to a GLOBAL variable so that all blocks
    // can see it.

    // TODO will need to change this to use the externally-added shmem
    __shared__ int shared_cur_tile_row_idx;

    if (threadIdx.x == 0) {
      cur_tile_row_idx = atomicAdd(&queue_counter[0], 1);
      shared_cur_tile_row_idx = cur_tile_row_idx;
    }

    __syncthreads();

    cur_tile_row_idx = shared_cur_tile_row_idx;
  }

  __device__ __forceinline__ void process_gpu_col_tile() {
    // Iterate over the row tiles as long as it's in bounds.

    print_device("Processing GPU col tile %d\n", (int)cur_tile_col_idx);

    // The GPU has its tile row index and col index. Need to first load in
    // metadata, then do the load balancing, then do the computation, then
    // unload any metadata we need to save to global mem for the next time we
    // come back to this row. Finally, need to increment an atomic and go on to
    // the next row tile if there are more to process.

    // Then do a grid-wide synchronization.

    // All blocks iterate over the row tiles
    while (cur_tile_row_idx < num_row_tiles) {
      process_block_row_tile();
      get_next_block_row_tile();
    }

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();
  }

  __device__ __forceinline__ void get_next_gpu_col_tile() {
    // Reset the tile metadata
    cur_tile_row_idx = blockIdx.x;
    cur_tile_col_idx += 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      queue_counter[0] = gridDim.x;
    }

    print_device("Starting column tile %d\n", (int)cur_tile_col_idx);

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // TODO do something to reset the caching here?
  }

  __device__ __forceinline__ void process_all_tiles() {
    while (!all_columns_finished()) {
      process_gpu_col_tile();
      get_next_gpu_col_tile();
    }
  }

 private:
  // Store the inputs and outputs
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  int* queue_counter;

  // Tiling metadata
  const size_t tile_row_size;
  const size_t tile_col_size;

  size_t cur_tile_row_idx;
  size_t cur_tile_col_idx;
  size_t num_row_tiles;
  size_t num_col_tiles;
};

template <typename graph_t, typename vector_t>
__global__ void spmv_tiled_kernel(graph_t graph,
                                  vector_t* input,
                                  vector_t* output,
                                  int* queue_counter,
                                  size_t tile_row_size,
                                  size_t tile_col_size) {
  // Store the output in shared memory
  using row_t = decltype(graph.get_row_offsets());
  extern __shared__ row_t shmem[];

  TileIterator<graph_t, vector_t, row_t> iterator(
      graph, input, output, queue_counter, tile_row_size, tile_col_size, shmem);

  iterator.process_all_tiles();

  // Simple, single-threaded implementation
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (auto i = 0; i < graph.get_number_of_rows(); i++) {
      vector_t y = 0;
      for (auto k = graph.get_row_offsets()[i];
           k < graph.get_row_offsets()[i + 1]; k++) {
        y = y + (graph.get_nonzero_values()[k] *
                 input[graph.get_column_indices()[k]]);
      }
      output[i] = y;
    }
  }
}

template <typename csr_t, typename vector_t>
double spmv_tiled(csr_t& csr, vector_t& input, vector_t& output) {
  auto debug = false;

  // --
  // Build graph

  // Convert the dataset you loaded into an `essentials` graph.
  // `memory_space_t::device` -> the graph will be created on the GPU.
  // `graph::view_t::csr`     -> your input data is in `csr` format.
  //
  // Note that `graph::build::from_csr` expects pointers, but the `csr` data
  // arrays are `thrust` vectors, so we need to unwrap them w/ `.data().get()`.
  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  // Need the types of the csr matrix for kernel setup
  using row_t = decltype(csr.row_offsets.data().get()[0]);
  using nonzero_t = decltype(csr.nonzero_values.data().get()[0]);

  /* ========== Setup Device Properties ========== */
  auto device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));

  // Setup grid and block properties
  auto numBlocksPerSm = 0;
  auto numThreadsPerBlock = 0;
  auto shmemPerBlock = 0;  // bytes

  auto target_occupancy = 2;

  // Number of coordinates. TODO calculate this
  // based on architecture L2 properties
  auto tile_size = 0;

  // Use the max number of threads per block to maximize parallelism over
  // shmem

  numThreadsPerBlock = deviceProp.maxThreadsPerBlock / target_occupancy;
  shmemPerBlock = (deviceProp.sharedMemPerBlockOptin / target_occupancy);

  auto store_end_offsets_in_shmem = true;

  auto data_elems_per_row = 1;
  if (store_end_offsets_in_shmem) {
    data_elems_per_row = 2;
  }
  auto rows_per_block = (shmemPerBlock / (sizeof(row_t) * data_elems_per_row));

  std::cout << "Threads Per Block: " << numThreadsPerBlock << std::endl;
  std::cout << "Rows Per Block: " << rows_per_block << std::endl;
  std::cout << "Shmem Per Block (bytes): " << shmemPerBlock << std::endl;

  CHECK_CUDA(cudaFuncSetAttribute(spmv_tiled_kernel<decltype(G), float>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  shmemPerBlock));

  // Need to know the max occupancy to determine how many blocks to launch
  // for the cooperative kernel. All blocks must be resident on SMs
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_tiled_kernel<decltype(G), float>,
      numThreadsPerBlock, shmemPerBlock))

  std::cout << "Max Active Blocks Per SM: " << numBlocksPerSm << std::endl;

  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  /* ========== Setup Kernel Call ========== */
  void* input_ptr = thrust::raw_pointer_cast(input.data());
  void* output_ptr = thrust::raw_pointer_cast(output.data());

  auto queue_counter = thrust::device_vector<int>(1);
  void* queue_counter_ptr = thrust::raw_pointer_cast(queue_counter.data());

  void* kernelArgs[] = {
      &G,        &input_ptr, &output_ptr, &queue_counter_ptr, &rows_per_block,
      &tile_size};

  /* ========== SETUP TILE SIZE ========== */
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));  // Create CUDA stream

  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;

  if (deviceProp.major >= 8) {
    // Using Ampere

    size_t size =
        min(int(deviceProp.l2CacheSize), deviceProp.persistingL2CacheMaxSize);

    // size is in bytes. Need to convert to elements
    tile_size = size / sizeof(row_t);

    printf("Device has cache size of %d bytes\n", (int)size);
    printf("Data elems per row: %d\n", (int)data_elems_per_row);
    printf("Data size: %ld\n", sizeof(row_t));

  } else {
    // Using Volta or below
    printf(
        "WARNING: L2 Cache Management available only for compute capabilities "
        "> 8\n");

    printf("Device has cache size of %d bytes\n", deviceProp.l2CacheSize);
    printf("Data elems per row: %d\n", data_elems_per_row);
    printf("Data size: %ld\n", sizeof(row_t));

    tile_size = (deviceProp.l2CacheSize / data_elems_per_row) / sizeof(row_t);
  }

  printf("Tile Size (elements): %d * %d, %d\n", (int)rows_per_block, (int)dimGrid.x,
         (int)tile_size);

  /* ========== Execute SPMV ========== */
  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUDA(cudaLaunchCooperativeKernel(
      (void*)spmv_tiled_kernel<decltype(G), float>, dimGrid, dimBlock,
      kernelArgs, shmemPerBlock, stream));

  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  return timer.milliseconds();
}
