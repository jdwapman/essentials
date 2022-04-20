#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

#include "spmv_utils.cuh"
#include "tile_iterator.cuh"

namespace cg = cooperative_groups;

using namespace gunrock;
using namespace memory;

struct tiledims {
  int matrix_tile_rows;
  int matrix_tile_cols;
  int spatial_tile_rows;
  int spatial_tile_cols;
  int temporal_tile_rows;
  int temporal_tile_cols;
  int block_tile_rows;
  int block_tile_cols;
};

// The parent class for all tiled iteration

template <typename graph_t, typename vector_t>
__global__ void __launch_bounds__(1024, 3)
    spmv_tiled_kernel(graph_t graph,
                      vector_t* input,
                      vector_t* output,
                      tiledims* dims,  // Used to extract for reporting
                      int tile_row_size,
                      int tile_col_size,
                      size_t shmem_size) {
  // Store the output in shared memory
  using row_t = typename graph_t::vertex_type;
  extern __shared__ row_t shmem[];

  // Just the matrix dimensions
  auto matrix_layout =
      make_layout(graph.get_number_of_rows(), graph.get_number_of_columns());

  // Use the actual row tile size
  auto spatial_layout = matrix_layout.tile(
      min(tile_row_size * gridDim.x, graph.get_number_of_rows()),
      graph.get_number_of_columns());

  // Use the actual row tile size
  auto temporal_layout = spatial_layout.tile(
      min(tile_row_size * gridDim.x, graph.get_number_of_rows()),
      graph.get_number_of_columns());

  // BUT if the entire matrix will fit in one block, we want to redistribute
  // the rows among other blocks

  // Remap the tile row size so that all blocks have work
  tile_row_size =
      min(tile_row_size,
          (int)ceil((float)graph.get_number_of_rows() / (float)gridDim.x));

  auto block_temporal_layout =
      temporal_layout.tile(min(tile_row_size, graph.get_number_of_rows()),
                           graph.get_number_of_columns());

  dims->matrix_tile_rows = matrix_layout.rows_in_tile(0);
  dims->matrix_tile_cols = matrix_layout.cols_in_tile(0);
  dims->spatial_tile_rows = spatial_layout.rows_in_tile(1);
  dims->spatial_tile_cols = spatial_layout.cols_in_tile(1);
  dims->temporal_tile_rows = temporal_layout.rows_in_tile(2);
  dims->temporal_tile_cols = temporal_layout.cols_in_tile(2);
  dims->block_tile_rows = block_temporal_layout.rows_in_tile(3);
  dims->block_tile_cols = block_temporal_layout.cols_in_tile(3);

  //   if (threadIdx.x == 0 && blockIdx.x == 0) {
  //     printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n",
  //            (int)block_temporal_layout.num_child_row_tiles(0),
  //            (int)block_temporal_layout.num_child_col_tiles(0),
  //            (int)block_temporal_layout.num_child_row_tiles(1),
  //            (int)block_temporal_layout.num_child_col_tiles(1),
  //            (int)block_temporal_layout.num_child_row_tiles(2),
  //            (int)block_temporal_layout.num_child_col_tiles(2),
  //            (int)block_temporal_layout.num_child_row_tiles(3),
  //            (int)block_temporal_layout.num_child_col_tiles(3));

  //     // Now print the tile sizes
  //     printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n",
  //            (int)block_temporal_layout.rows_in_tile(0),
  //            (int)block_temporal_layout.cols_in_tile(0),
  //            (int)block_temporal_layout.rows_in_tile(1),
  //            (int)block_temporal_layout.cols_in_tile(1),
  //            (int)block_temporal_layout.rows_in_tile(2),
  //            (int)block_temporal_layout.cols_in_tile(2),
  //            (int)block_temporal_layout.rows_in_tile(3),
  //            (int)block_temporal_layout.cols_in_tile(3));
  //   }

  TileIterator<graph_t, vector_t, row_t, decltype(block_temporal_layout)>
      matrix_tile_iterator(graph, input, output, shmem, shmem_size,
                           block_temporal_layout);

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
double spmv_tiled(cudaStream_t stream,
                  csr_t& csr,
                  vector_t& input,
                  vector_t& output,
                  args_t pargs,
                  json& _results) {
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
  auto target_occupancy = 3;

  _results["tiled_spmv"]["target_occupancy"] = target_occupancy;

  numThreadsPerBlock =
      min(deviceProp.maxThreadsPerBlock,
          deviceProp.maxThreadsPerMultiProcessor / target_occupancy);
  shmemPerBlock =
      (deviceProp.sharedMemPerBlockOptin - target_occupancy * 1024) /
      target_occupancy;

  auto bytes_per_row = 2 * sizeof(row_t) + sizeof(nonzero_t);
  auto rows_per_block = (shmemPerBlock / bytes_per_row) - 1;

  std::cout << "Threads Per Block: " << numThreadsPerBlock << std::endl;
  std::cout << "Rows Per Block: " << rows_per_block << std::endl;
  std::cout << "Shmem Per Block (bytes): " << shmemPerBlock << std::endl;

  CHECK_CUDA(cudaFuncSetAttribute(spmv_tiled_kernel<decltype(G), float>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  deviceProp.sharedMemPerBlockOptin));

  //   int carveout = 100;
  //   CHECK_CUDA(cudaFuncSetAttribute(
  //       spmv_tiled_kernel<decltype(G), float>,
  //       cudaFuncAttributePreferredSharedMemoryCarveout, carveout));

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

  _results["tiled_spmv"]["registers"] = attr.numRegs;
  _results["tiled_spmv"]["threads_per_block"] = numThreadsPerBlock;
  _results["tiled_spmv"]["shmem_per_block"] = shmemPerBlock;
  _results["tiled_spmv"]["max_active_blocks_per_sm"] = numBlocksPerSm;

  assert(numBlocksPerSm == target_occupancy);

  dim3 dimBlock(numThreadsPerBlock, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

  _results["tiled_spmv"]["blocks"] = dimGrid.x;

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

  thrust::device_vector<tiledims> tiledims_vec(1);
  void* tiledims_ptr = thrust::raw_pointer_cast(tiledims_vec.data());

  _results["tiled_spmv"]["max_rows_per_block"] = rows_per_block;
  _results["tiled_spmv"]["max_cols_per_block"] = cols_per_block;

  void* kernelArgs[] = {&G,
                        &input_ptr,
                        &output_ptr,
                        &tiledims_ptr,
                        &rows_per_block,
                        &cols_per_block,
                        &shmemPerBlock};

  printf("Tile Size (elements): %d * %d, %d\n", (int)rows_per_block,
         (int)dimGrid.x, (int)cols_per_block);

  /* ========== Execute SPMV ========== */

  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUDA(cudaLaunchCooperativeKernel(
      (void*)spmv_tiled_kernel<decltype(G), float>, dimGrid, dimBlock,
      kernelArgs, shmemPerBlock, stream));

  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  thrust::host_vector<tiledims> tiledims_vec_host(1);
  tiledims_vec_host = tiledims_vec;

  // Save the tile size results
  _results["tiled_spmv"]["tile_size"]["matrix_tile_rows"] =
      tiledims_vec_host[0].matrix_tile_rows;
  _results["tiled_spmv"]["tile_size"]["matrix_tile_cols"] =
      tiledims_vec_host[0].matrix_tile_cols;
  _results["tiled_spmv"]["tile_size"]["spatial_tile_rows"] =
      tiledims_vec_host[0].spatial_tile_rows;
  _results["tiled_spmv"]["tile_size"]["spatial_tile_cols"] =
      tiledims_vec_host[0].spatial_tile_cols;
  _results["tiled_spmv"]["tile_size"]["temporal_tile_rows"] =
      tiledims_vec_host[0].temporal_tile_rows;
  _results["tiled_spmv"]["tile_size"]["temporal_tile_cols"] =
      tiledims_vec_host[0].temporal_tile_cols;
  _results["tiled_spmv"]["tile_size"]["block_tile_rows"] =
      tiledims_vec_host[0].block_tile_rows;
  _results["tiled_spmv"]["tile_size"]["block_tile_cols"] =
      tiledims_vec_host[0].block_tile_cols;

  return timer.milliseconds();
}
