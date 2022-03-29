#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "spmv_utils.cuh"
#include "tile_iterator.cuh"

namespace cg = cooperative_groups;

using namespace gunrock;
using namespace memory;

// The parent class for all tiled iteration

template <typename graph_t, typename vector_t>
__global__ void spmv_tiled_kernel(graph_t graph,
                                  vector_t* input,
                                  vector_t* output,
                                  size_t tile_row_size,
                                  size_t tile_col_size,
                                  size_t shmem_size) {
  // Store the output in shared memory
  using row_t = typename graph_t::vertex_type;
  extern __shared__ row_t shmem[];

  // Set up the tiles

  auto base_layout =
      make_layout(graph.get_number_of_rows(), graph.get_number_of_columns());
  auto device_batch_tiled_layout = base_layout.tile(
      tile_row_size * blockDim.x, graph.get_number_of_columns());
  auto device_tiled_layout =
      device_batch_tiled_layout.tile(tile_row_size * blockDim.x, tile_col_size);
  auto block_tiled_layout =
      device_tiled_layout.tile(tile_row_size, tile_col_size);

  auto dims = block_tiled_layout.tiledims;

  TileIterator<graph_t, vector_t, row_t, decltype(block_tiled_layout)>
      matrix_tile_iterator(graph, input, output, shmem, shmem_size,
                           block_tiled_layout);

  matrix_tile_iterator.process_all_tiles();

  // Simple, single-threaded implementation
  // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //   for (auto i = 0; i < graph.get_number_of_rows(); i++) {
  //     vector_t y = 0;
  //     for (auto k = graph.get_row_offsets()[i];
  //          k < graph.get_row_offsets()[i + 1]; k++) {
  //       y = y + (graph.get_nonzero_values()[k] *
  //                input[graph.get_column_indices()[k]]);
  //     }
  //     output[i] = y;
  //   }
  // }
}

template <typename csr_t, typename vector_t, typename args_t>
double spmv_tiled(csr_t& csr, vector_t& input, vector_t& output, args_t pargs) {
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
  int shmemPerBlock = 0;  // bytes

  // Use the max number of threads per block to maximize parallelism over
  // shmem

  numThreadsPerBlock = deviceProp.maxThreadsPerBlock / 4;
  shmemPerBlock = deviceProp.sharedMemPerBlockOptin / 4;

  auto bytes_per_row = 2 * sizeof(row_t) + sizeof(nonzero_t);
  auto rows_per_block = (shmemPerBlock / bytes_per_row) - 1;

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

  // See how many registers the kernel uses
  cudaFuncAttributes attr;
  CHECK_CUDA(
      cudaFuncGetAttributes(&attr, spmv_tiled_kernel<decltype(G), float>));

  std::cout << "Registers: " << attr.numRegs << std::endl;

  std::cout << "Max Active Blocks Per SM: " << numBlocksPerSm << std::endl;

  assert(numBlocksPerSm > 0);

  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  /* ========== SETUP TILE SIZE ========== */
  // TODO need to set this up so I actually do cache pinning. I think this is
  // already done in the main function? Just need to pass the stream in

  auto cols_per_block = 0;

  if (deviceProp.major >= 8) {
    // Using Ampere

    auto pinned_cache_size =
        min(int(deviceProp.l2CacheSize), deviceProp.persistingL2CacheMaxSize);

    // size is in bytes. Need to convert to elements
    cols_per_block = pinned_cache_size / sizeof(nonzero_t);

    printf("Device has cache size of %d bytes\n", (int)pinned_cache_size);

  } else {
    // Using Volta or below
    printf(
        "WARNING: L2 Cache Management available only for compute capabilities "
        "> 8\n");

    printf("Device has cache size of %d bytes\n", deviceProp.l2CacheSize);
    printf("Data bytes per row: %ld\n", bytes_per_row);
    printf("Data size: %ld\n", sizeof(row_t));

    cols_per_block = (deviceProp.l2CacheSize / sizeof(nonzero_t));
  }

  /* ========== Setup Kernel Call ========== */
  void* input_ptr = thrust::raw_pointer_cast(input.data());
  void* output_ptr = thrust::raw_pointer_cast(output.data());

  void* kernelArgs[] = {&G,
                        &input_ptr,
                        &output_ptr,
                        &rows_per_block,
                        &cols_per_block,
                        &shmemPerBlock};

  printf("Tile Size (elements): %d * %d, %d\n", (int)rows_per_block,
         (int)dimGrid.x, (int)cols_per_block);

  /* ========== Execute SPMV ========== */

  // Create a cuda stream
  auto stream = setup_ampere_cache(input);

  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUDA(cudaLaunchCooperativeKernel(
      (void*)spmv_tiled_kernel<decltype(G), float>, dimGrid, dimBlock,
      kernelArgs, shmemPerBlock, stream));

  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  return timer.milliseconds();
}
