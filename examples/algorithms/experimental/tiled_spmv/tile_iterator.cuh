#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>
#include "spmv_utils.cuh"
#include <tuple>
#include "hierarchy_tools.cuh"
#include <cuda/annotated_ptr>

namespace cg = cooperative_groups;

template <typename graph_t,
          typename vector_t,
          typename shmem_t,
          typename layout_t>
class TileIterator {
 public:
  __device__ __forceinline__ TileIterator(const graph_t _graph,
                                          vector_t* _input,
                                          vector_t* _output,
                                          shmem_t* _shmem,
                                          size_t _shmem_size,
                                          layout_t _tile_layout)
      : graph(_graph),
        input(_input),
        output(_output),
        shmem(_shmem),
        shmem_size(_shmem_size),
        tile_layout(_tile_layout) {
    using row_t = typename graph_t::vertex_type;

    // TODO make sure shmem is aligned
    shmem_row_offsets_start = (row_t*)shmem;

    shmem_row_offsets_end =
        (row_t*)(&shmem_row_offsets_start[tile_layout.rows_in_tile(
            TILE_TEMPORAL_BLOCK)]);

    shmem_output = (vector_t*)(&shmem_row_offsets_end[tile_layout.rows_in_tile(
        TILE_TEMPORAL_BLOCK)]);

    // Check if shared memory is aligned to 4, 8, 16, 32, 64, or 128 bytes
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      // Print the addresses of start and end
      printf("Shared memory start: %p\n", shmem_row_offsets_start);
      printf("Shared memory end: %p\n", shmem_row_offsets_end);
    }

    row_queue =
        (int*)(&shmem_output[tile_layout.rows_in_tile(TILE_TEMPORAL_BLOCK)]);
  }

  template <typename tile_index_t, typename RowT>
  __device__ __forceinline__ void load_tile(tile_index_t parent_tile_idx,
                                            RowT row_offsets) {
    // If we're loading the first tile in a batch, the device tile is by
    // default  (0, 0) relative to the parent
    auto device_tile_idx = make_tile_index(0, 0, parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    auto block_tile_idx = make_tile_index((int)blockIdx.x, 0, device_tile_idx);
    auto rows_in_block = tile_layout.rows_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.

    auto matrix_coord = tile_layout.remap_point(
        Point<int, int>(0, 0), block_tile_idx, (size_t)TILE_MATRIX);

    // Get the boundaries of the tile: the minimum of the number of rows in
    // the block and the remainder of the number of rows in graph

    // Ampere async copy
    auto block_group = cg::this_thread_block();

    auto size =
        min(rows_in_block, graph.get_number_of_rows() - matrix_coord.row) *
        sizeof(shmem_t);

    if (matrix_coord.row < graph.get_number_of_rows()) {
      cg::memcpy_async(block_group,                                         //
                       shmem_row_offsets_start,                             //
                       &(this->graph.get_row_offsets()[matrix_coord.row]),  //
                       size);
      cg::memcpy_async(
          block_group,                                             // CG
          shmem_row_offsets_end,                                   // dst
          &(this->graph.get_row_offsets()[matrix_coord.row + 1]),  // src
          size);
      cg::wait(block_group);
    }

// Iterate and copy to shared
// TODO don't need to read in starts and stops separately. Combine for
//      efficiency
#pragma unroll
    for (auto row_idx = threadIdx.x;
         row_idx < rows_in_block &&
         matrix_coord.row + row_idx < graph.get_number_of_rows();
         row_idx += blockDim.x) {
      this->shmem_output[row_idx] = 0;
    }

    __syncthreads();
  }

  template <typename tile_index_t,
            typename RowT,
            typename ColT,
            typename ValT,
            typename InputT,
            typename OutputT>
  __device__ __forceinline__ void process_tile_thread_per_row(
      tile_index_t parent_tile_idx,
      RowT row_offsets_ptr,
      ColT col_indices_ptr,
      ValT values_ptr,
      InputT input_ptr,
      OutputT output_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("Processing tile\n");
    }

    // If we're loading the first tile in a batch, the device tile is by
    // default  (0, 0) relative to the parent
    auto device_tile_idx = make_tile_index(0, 0, parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    auto block_tile_idx = make_tile_index((int)blockIdx.x, 0, device_tile_idx);
    auto rows_in_block = tile_layout.rows_in_tile(block_tile_idx);
    auto cols_in_block = tile_layout.cols_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.

    auto matrix_coord = tile_layout.remap_point(
        Point<int, int>(0, 0), block_tile_idx, (size_t)TILE_MATRIX);

    // Do the SPMV
    // 1. Iterate over the rows in the block
#pragma unroll
    for (auto row_idx = threadIdx.x;
         row_idx < rows_in_block &&
         matrix_coord.row + row_idx < graph.get_number_of_rows();
         row_idx += blockDim.x) {
      // printf("Block %d, Thread %d processing row %d\n", (int)blockIdx.x,
      //        (int)threadIdx.x, (int)(matrix_coord.row + row_idx));

      // 2. Within a row, iterate over the rows in the block until we reach
      //    either the end of the tile or the end of the matrix
      vector_t accum = this->shmem_output[row_idx];

      auto tile_boundary =
          min(graph.get_number_of_columns(),
              (parent_tile_idx.col[TILE_TEMPORAL] + 1) * cols_in_block);

      auto offset = this->shmem_row_offsets_start[row_idx];

      while (true) {
        auto col = __ldcs(&(this->graph.get_column_indices()[offset]));

        printf("Thread %d accessing address %p\n", threadIdx.x,
               &(this->graph.get_column_indices()[offset]));

        // Check if we've crossed a tile boundary...
        if ((int)col >= (int)tile_boundary) {
          break;
        }

        // ... OR reached the end of the row
        if ((int)offset >= this->shmem_row_offsets_end[row_idx]) {
          break;
        }

        accum += __ldcs(&(this->graph.get_nonzero_values()[offset])) *
                 this->input[col];

        offset++;
      }

      // Save the offset and values for the next iterations
      this->shmem_row_offsets_start[row_idx] = offset;
      this->shmem_output[row_idx] = accum;
    }
  }

  template <typename tile_index_t,
            typename RowT,
            typename ColT,
            typename ValT,
            typename InputT,
            typename OutputT>
  __device__ __forceinline__ void process_tile_warp_per_row(
      tile_index_t parent_tile_idx,
      RowT row_offsets_ptr,
      ColT col_indices_ptr,
      ValT values_ptr,
      InputT input_ptr,
      OutputT output_ptr) {
    // If we're loading the first tile in a batch, the device tile is by
    // default (0, 0) relative to the parent
    auto device_tile_idx = make_tile_index(0, 0, parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    auto block_tile_idx = make_tile_index((int)blockIdx.x, 0, device_tile_idx);
    auto rows_in_block = tile_layout.rows_in_tile(block_tile_idx);
    auto cols_in_block = tile_layout.cols_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.

    auto matrix_coord = tile_layout.remap_point(
        Point<int, int>(0, 0), block_tile_idx, (size_t)TILE_MATRIX);

    // Do the SPMV
    // 1. Rows iterate over the rows in the block
    auto stride = blockDim.x / 32;
#pragma unroll
    for (auto row_idx = (threadIdx.x / 32);
         row_idx < rows_in_block &&
         matrix_coord.row + row_idx < graph.get_number_of_rows();
         row_idx += stride) {
      // if (threadIdx.x % 32 == 0) {
      //   printf("Block %d, Warp %d processing row %d\n", (int)blockIdx.x,
      //          (int)(threadIdx.x / 32), (int)(matrix_coord.row + row_idx));
      // }

      // 2. Within a row, iterate over the rows in the block until we reach
      //    either the end of the tile or the end of the matrix

      vector_t accum = this->shmem_output[row_idx];

      auto tile_boundary =
          min(graph.get_number_of_columns(),
              (parent_tile_idx.col[TILE_TEMPORAL] + 1) * cols_in_block);

      auto offset = this->shmem_row_offsets_start[row_idx];

      // TODO hardcode this to 10000 for debugging and remove boundary checks.
      // Want to get performance parity for dense datasets

      while (true) {
        // Each thread gets a column. Nice and coalesced
        auto col = __ldcs(
            &(this->graph.get_column_indices()[offset + (threadIdx.x % 32)]));

        // Check if we've crossed a tile boundary...
        if ((int)col >= (int)tile_boundary) {
          break;
        }

        // ... OR reached the end of the row
        if ((int)(offset + threadIdx.x % 32) >=
            this->shmem_row_offsets_end[row_idx]) {
          break;
        }

        auto active = cg::coalesced_threads();

        vector_t warp_val =
            __ldcs(&(this->graph
                         .get_nonzero_values()[offset + (threadIdx.x % 32)])) *
            this->input[col];

        auto warp_reduce_val =
            cg::reduce(active, warp_val, cg::plus<vector_t>());

        accum += warp_reduce_val;

        offset += active.size();
      }

      // Save the offset and values for the next iterations.
      // When using warp-per-row, only one thread needs to do this.
      if (threadIdx.x % 32 == 0) {
        this->shmem_row_offsets_start[row_idx] = offset;
        this->shmem_output[row_idx] = accum;
      }
    }
  }

  template <typename tile_index_t,
            typename RowT,
            typename ColT,
            typename ValT,
            typename InputT,
            typename OutputT>
  __device__ __forceinline__ void process_tile_warp_per_row_queue(
      tile_index_t parent_tile_idx,
      RowT row_offsets_ptr,
      ColT col_indices_ptr,
      ValT values_ptr,
      InputT input_ptr,
      OutputT output_ptr) {
    // If we're loading the first tile in a batch, the device tile is by
    // default (0, 0) relative to the parent
    auto device_tile_idx = make_tile_index(0, 0, parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    auto block_tile_idx = make_tile_index((int)blockIdx.x, 0, device_tile_idx);
    auto rows_in_block = tile_layout.rows_in_tile(block_tile_idx);
    auto cols_in_block = tile_layout.cols_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.

    auto matrix_coord = tile_layout.remap_point(
        Point<int, int>(0, 0), block_tile_idx, (size_t)TILE_MATRIX);

    // Init the warp rows
    auto row_idx = threadIdx.x / 32;

    // Reset the queue counter
    if (threadIdx.x == 0) {
      this->row_queue[0] = (int)(blockDim.x / 32);
    }
    __syncthreads();

    while (row_idx < rows_in_block &&
           matrix_coord.row + row_idx < graph.get_number_of_rows()) {
      // 2. Within a row, iterate over the rows in the block until we reach
      //    either the end of the tile or the end of the matrix

      // Check if the row is completely empty
      if (this->shmem_row_offsets_start[row_idx] ==
          this->shmem_row_offsets_end[row_idx]) {
        // Get the next row from the queue
        if (threadIdx.x % 32 == 0) {
          row_idx = atomicAdd(&(this->row_queue[0]), 1);
        }

        // Broadcast the row idx to all other threads in the warp
        row_idx = __shfl_sync(0xffffffff, row_idx, 0);
        continue;
      }

      vector_t accum = this->shmem_output[row_idx];

      auto tile_boundary =
          min(graph.get_number_of_columns(),
              (parent_tile_idx.col[TILE_TEMPORAL] + 1) * cols_in_block);

      auto offset = this->shmem_row_offsets_start[row_idx];

      while (true) {
        // Each thread gets a column. Nice and coalesced
        // auto col = __ldcs(
        //     &(this->graph.get_column_indices()[offset + (threadIdx.x %
        //     32)]));

        const auto col = col_indices_ptr[offset + (threadIdx.x % 32)];

        // Check if we've crossed a tile boundary...
        if ((int)col >= (int)tile_boundary) {
          break;
        }

        // ... OR reached the end of the row
        if ((int)(offset + threadIdx.x % 32) >=
            this->shmem_row_offsets_end[row_idx]) {
          break;
        }

        auto active = cg::coalesced_threads();

        // vector_t warp_val =
        //     __ldcs(&(this->graph
        //                  .get_nonzero_values()[offset + (threadIdx.x % 32)]))
        //                  *
        //     this->input[col];

        auto addr = offset + (threadIdx.x % 32);

        printf("Thread %d Loading Address %d for row %d\n", threadIdx.x, addr,
               row_idx);

        vector_t warp_val = values_ptr[addr] * input_ptr[col];

        auto warp_reduce_val =
            cg::reduce(active, warp_val, cg::plus<vector_t>());

        accum += warp_reduce_val;

        offset += active.size();
      }

      // Save the offset and values for the next iterations.
      // When using warp-per-row, only one thread needs to do this.
      if (threadIdx.x % 32 == 0) {
        this->shmem_row_offsets_start[row_idx] = offset;
        this->shmem_output[row_idx] = accum;
      }

      // Get the next row from the queue
      if (threadIdx.x % 32 == 0) {
        row_idx = atomicAdd(&(this->row_queue[0]), 1);
      }

      // Broadcast the row idx to all other threads in the warp
      row_idx = __shfl_sync(0xffffffff, row_idx, 0);
    }
  }

  template <typename tile_index_t, typename OutputT>
  __device__ __forceinline__ void store_tile(tile_index_t parent_tile_idx,
                                             OutputT output_ptr) {
    // Write the outputs to the output vector
    // Unload data from shared memory

    // If we're loading the first tile in a batch, the device tile is by
    // default  (0, 0) relative to the parent
    auto device_tile_idx = make_tile_index(0, 0, parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    auto block_tile_idx = make_tile_index((int)blockIdx.x, 0, device_tile_idx);
    auto rows_in_block = tile_layout.rows_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.

    auto matrix_coord = tile_layout.remap_point(
        Point<int, int>(0, 0), block_tile_idx, (size_t)TILE_MATRIX);

// Iterate and copy to global
// TODO make this vectorized?
#pragma unroll
    for (auto row_idx = threadIdx.x;
         row_idx < rows_in_block &&
         matrix_coord.row + row_idx < graph.get_number_of_rows();
         row_idx += blockDim.x) {
      // Write the outputs to the output vector. Store streaming since
      // we won't access this again
      // output[matrix_coord.row + row_idx] = this->shmem_output[row_idx];

      output_ptr[matrix_coord.row + row_idx] = this->shmem_output[row_idx];

      // __stcs(&(output[matrix_coord.row + row_idx]),
      //        this->shmem_output[row_idx]);
    }
  }

  template <typename tile_index_t,
            typename RowT,
            typename ColT,
            typename ValT,
            typename InputT,
            typename OutputT>
  __device__ __forceinline__ void process_all_tiles_at_hierarchy(
      tile_index_t parent_tile_idx,
      RowT row_offsets_ptr,
      ColT col_indices_ptr,
      ValT values_ptr,
      InputT input_ptr,
      OutputT output_ptr) {
    // print_device("Processing tile (%d, %d) at %d\n",
    //              (int)parent_tile_idx.row[parent_tile_idx.getHierarchy()],
    //              (int)parent_tile_idx.col[parent_tile_idx.getHierarchy()],
    //              (int)parent_tile_idx.getHierarchy());

    // ===== SETUP TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_SPATIAL) {
      load_tile(parent_tile_idx, row_offsets_ptr);
    }

    // ===== TILE PROCESSING TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_TEMPORAL) {
      // We aren't iterating over the tile anymore, we're now processing it
      // and diving into parallel work
      process_tile_thread_per_row(parent_tile_idx, row_offsets_ptr,
                                  col_indices_ptr, values_ptr, input_ptr,
                                  output_ptr);
      // process_tile_warp_per_row(parent_tile_idx, row_offsets_ptr,
      //                                 col_indices_ptr, values_ptr, input_ptr,
      //                                 output_ptr);
      // process_tile_warp_per_row_queue(parent_tile_idx, row_offsets_ptr,
      //                                 col_indices_ptr, values_ptr, input_ptr,
      //                                 output_ptr);

      // Get the current grid and sync
      auto grid = cg::this_grid();
      grid.sync();
    } else {
      // Tile indexer to the child tiles if the parent tile
      auto child_tile_idx = make_tile_index(0, 0, parent_tile_idx);

      auto h_idx = child_tile_idx.getHierarchy();

      // Iterate over the child tiles
#pragma unroll
      for (child_tile_idx.row[h_idx] = 0;
           child_tile_idx.row[h_idx] <
           tile_layout.num_child_row_tiles(parent_tile_idx);
           child_tile_idx.row[h_idx]++) {
#pragma unroll
        for (child_tile_idx.col[h_idx] = 0;
             child_tile_idx.col[h_idx] <
             tile_layout.num_child_col_tiles(parent_tile_idx);
             child_tile_idx.col[h_idx]++) {
          process_all_tiles_at_hierarchy(child_tile_idx, row_offsets_ptr,
                                         col_indices_ptr, values_ptr, input_ptr,
                                         output_ptr);
        }
      }
    }

    // ===== TEARDOWN TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_SPATIAL) {
      store_tile(parent_tile_idx, output_ptr);
    }
  }

  // Iterate all tiles within level of the hierarchy (h0 is the Matrix)
  __device__ __forceinline__ void process_all_tiles(bool pin) {
    auto matrix_tile_index = make_tile_index(0, 0);

    if (pin) {
      // Set up the annotated pointers
      cuda::annotated_ptr<int, cuda::access_property::streaming>
          row_offsets_ptr(this->graph.get_row_offsets());
      cuda::annotated_ptr<int, cuda::access_property::streaming>
          column_indices_ptr(this->graph.get_column_indices());
      cuda::annotated_ptr<float, cuda::access_property::streaming>
          nonzero_values_ptr(this->graph.get_nonzero_values());
      cuda::annotated_ptr<float, cuda::access_property::streaming> output_ptr(
          this->output);
      cuda::annotated_ptr<float, cuda::access_property::persisting> input_ptr(
          this->input);

      process_all_tiles_at_hierarchy(matrix_tile_index, row_offsets_ptr,
                                     column_indices_ptr, nonzero_values_ptr,
                                     input_ptr, output_ptr);
    } else {
      // Set up the annotated pointers
      cuda::annotated_ptr<int, cuda::access_property::normal> row_offsets_ptr(
          this->graph.get_row_offsets());
      cuda::annotated_ptr<int, cuda::access_property::normal>
          column_indices_ptr(this->graph.get_column_indices());
      cuda::annotated_ptr<float, cuda::access_property::normal>
          nonzero_values_ptr(this->graph.get_nonzero_values());
      cuda::annotated_ptr<float, cuda::access_property::normal> output_ptr(
          this->output);
      cuda::annotated_ptr<float, cuda::access_property::normal> input_ptr(
          this->input);

      process_all_tiles_at_hierarchy(matrix_tile_index, row_offsets_ptr,
                                     column_indices_ptr, nonzero_values_ptr,
                                     input_ptr, output_ptr);
    }
  }

 public:
  const graph_t graph;
  vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  size_t shmem_size;

  shmem_t* shmem_row_offsets_start;
  shmem_t* shmem_row_offsets_end;
  vector_t* shmem_output;

  shmem_t* row_queue;

  layout_t tile_layout;
};
