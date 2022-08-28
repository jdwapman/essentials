#pragma once

#include "spmv_utils.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <gunrock/util/timer.hxx>
#include <thrust/device_vector.h>

namespace cg = cooperative_groups;

template <typename graph_t, typename vector_t>
__global__ void spmv_queue_kernel(graph_t graph,
                                  vector_t* input,
                                  vector_t* output,
                                  int* queue) {
  // Use cooperative groups to get a warp tile
  auto tile32 = cg::tiled_partition<32>(cg::this_thread_block());

  const auto global_thread_id = gcuda::thread::global::id::x();
  const auto global_warp_id = global_thread_id / 32;

  auto row = global_warp_id;

  // Iterate over all rows in the matrix.
  while (row < graph.get_number_of_rows()) {
    // Do the SPMV here
    vector_t thread_value = 0;

    // 1. Get the row start and end offsets given the row ID
    const auto row_start_offset = graph.get_row_offsets()[row];
    const auto row_end_offset = graph.get_row_offsets()[row + 1];

    // 2. Each thread loads and accumulates its matrix and vector values in a
    // loop.
    for (auto col_idx = row_start_offset + tile32.thread_rank();
         col_idx < row_end_offset; col_idx += 32) {
      const auto col = graph.get_column_indices()[col_idx];
      thread_value += input[col] * graph.get_nonzero_values()[col_idx];
    }

    // 3. Do a reduction to the final value
    const auto row_reduced_value =
        cg::reduce(tile32, thread_value, cg::plus<vector_t>());

    // 4. Store the result in the output vector (don't need an atomic since
    // we're doing warp-per-row)
    if (tile32.thread_rank() == 0) {
      output[row] = row_reduced_value;
    }

    // 5. Increment the queue value to get a new row
    if (tile32.thread_rank() == 0) {
      row = atomicAdd(&queue[0], 1);
    }

    // Broadcase the new row to all threads in the warp
    row = __shfl_sync(0xffffffff, row, 0);
  }
}

template <typename csr_t, typename vector_t, typename args_t>
double spmv_queue(cudaStream_t stream,
                  csr_t& csr,
                  vector_t& input,
                  vector_t& output,
                  args_t pargs) {
  int device = pargs["device"].template as<int>();
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device));

  thrust::device_vector<int> queue(1);

  auto G = graph::build::from_csr<memory_space_t::device, graph::view_t::csr>(
      csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
      csr.row_offsets.data().get(), csr.column_indices.data().get(),
      csr.nonzero_values.data().get());

  // TODO tune these values, make sure I'm getting full occupancy.
  int numBlocksPerSm = 0;
  const int threads_per_block = 256;

  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, spmv_queue_kernel<decltype(G), float>, threads_per_block,
      0))

  const auto numBlocks = numBlocksPerSm * deviceProp.multiProcessorCount;

  // Since we start off with every warp having a sincle row, initialize the
  // queue to num_warps.
  // TODO need to update this for grid size
  queue[0] = numBlocks * threads_per_block / 32;

  // execute SpMV
  gunrock::util::timer_t timer;
  timer.begin();

  // Launch the kernel here
  spmv_queue_kernel<<<numBlocks, threads_per_block, 0, stream>>>(
      G, input.data().get(), output.data().get(), queue.data().get());

  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  return timer.milliseconds();
}