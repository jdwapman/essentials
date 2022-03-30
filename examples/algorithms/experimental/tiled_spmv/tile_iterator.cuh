#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "spmv_utils.cuh"
#include <tuple>

namespace cg = cooperative_groups;

#define TILE_MATRIX 0
#define TILE_SPATIAL 1
#define TILE_TEMPORAL 2
#define TILE_TEMPORAL_BLOCK 3

// Forward declare
template <typename rowdim_t, typename coldim_t>
__host__ __device__ __forceinline__ constexpr auto make_layout(rowdim_t rowdim,
                                                               coldim_t coldim);

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
__host__ __device__ __forceinline__ constexpr auto
make_layout(rowdim_t rowdim, coldim_t coldim, parentlayout_t parentlayout);

template <typename row_t, typename col_t>
struct Point {
  row_t row;
  col_t col;
  __device__ __forceinline__ Point(row_t _row, col_t _col)
      : row(_row), col(_col) {}
};

// TODO need to rewrite this to be runtime-modifiable (no tuples)
template <typename row_t, typename col_t, int hierarchy>
struct TileIdx {
  row_t row[hierarchy + 1];
  col_t col[hierarchy + 1];

  __host__ __device__ __forceinline__ TileIdx(row_t _row, col_t _col) {
    row[0] = _row;
    col[0] = _col;
  }

  __host__ __device__ __forceinline__ TileIdx(row_t _row,
                                              col_t _col,
                                              row_t* _parent_rows,
                                              col_t* _parent_cols) {
    row[hierarchy] = _row;
    col[hierarchy] = _col;
    for (int i = hierarchy - 1; i >= 0; i--) {
      row[i] = _parent_rows[i];
      col[i] = _parent_cols[i];
    }
  }

  __host__ __device__ __forceinline__ constexpr auto getHierarchy() const {
    return hierarchy;
  }
};

template <typename row_t, typename col_t>
__host__ __device__ __forceinline__ constexpr auto make_tile_index(row_t row,
                                                                   col_t col) {
  return TileIdx<row_t, col_t, 0>(row, col);
}

template <typename row_t, typename col_t, typename parenttile_t>
__host__ __device__ __forceinline__ constexpr auto
make_tile_index(row_t row, col_t col, parenttile_t parenttile) {
  return TileIdx<row_t, col_t, parenttile.getHierarchy() + 1>(
      row, col, parenttile.row, parenttile.col);
}

// NOTE: Need to store layout data as a tuple.
// Format: < <row0, col0>, <row1, col1>, ... >

template <typename tiledim_t>
class Layout {
 public:
  __host__ __device__
      __forceinline__ constexpr Layout(const tiledim_t& _tiledims)
      : tiledims(_tiledims) {}

  template <typename rowdim_t, typename coldim_t>
  __host__ __device__ __forceinline__ constexpr auto tile(
      const rowdim_t rowdim,
      const coldim_t coldim) {
    return make_layout(rowdim, coldim, *this);
  }

  __host__ __device__ __forceinline__ constexpr bool has_parent() const {
    return std::tuple_size<tiledim_t>::value > 1;
  }

  __host__ __device__ __forceinline__ constexpr auto get_hierarchy_level()
      const {
    return std::tuple_size<tiledim_t>::value - 1;
  }

  // ===== TILE INFO FUNCTIONS ===== //

  // Get the dimensions of a tile
  __device__ __forceinline__ constexpr auto rows_in_tile(
      const int hierarchy) const {
    auto tiledim = TupleReturnValue(hierarchy, tiledims);
    return std::get<0>(tiledim);
  }

  __device__ __forceinline__ constexpr auto cols_in_tile(
      const int hierarchy) const {
    auto tiledim = TupleReturnValue(hierarchy, tiledims);
    return std::get<1>(tiledim);
  }

  template <typename tile_index_t>
  __device__ __forceinline__ constexpr auto rows_in_tile(
      const tile_index_t tile_index) const {
    return rows_in_tile(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __device__ __forceinline__ constexpr auto cols_in_tile(
      const tile_index_t tile_index) const {
    return cols_in_tile(tile_index.getHierarchy());
  }

  // Get the number of child tiles
  // TODO need to handle the remainders
  __device__ __forceinline__ constexpr auto num_child_row_tiles(
      const int& hierarchy) const {
    if (hierarchy == get_hierarchy_level()) {
      return 1;
    } else {
      auto num_even_tiles =
          rows_in_tile(hierarchy) / rows_in_tile(hierarchy + 1);

      if (rows_in_tile(hierarchy) % rows_in_tile(hierarchy + 1) == 0) {
        return num_even_tiles;
      } else {
        return num_even_tiles + 1;
      }
    }
  }

  __device__ __forceinline__ constexpr auto num_child_col_tiles(
      const int hierarchy) const {
    if (hierarchy == get_hierarchy_level()) {
      return 1;
    } else {
      auto num_even_tiles =
          cols_in_tile(hierarchy) / cols_in_tile(hierarchy + 1);

      if (cols_in_tile(hierarchy) % cols_in_tile(hierarchy + 1) == 0) {
        return num_even_tiles;
      } else {
        return num_even_tiles + 1;
      }
    }
  }

  template <typename tile_index_t>
  __device__ __forceinline__ auto num_child_row_tiles(
      const tile_index_t& tile_index) const {
    return num_child_row_tiles(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __device__ __forceinline__ auto num_child_col_tiles(
      const tile_index_t& tile_index) const {
    return num_child_col_tiles(tile_index.getHierarchy());
  }

  // Get the number of tiles at the level of the given tile
  __device__ __forceinline__ constexpr auto num_row_tiles_at_level(
      const int& hierarchy) const {
    if (hierarchy == 0) {
      return 1;
    }

    return num_child_row_tiles(hierarchy - 1);
  }

  __device__ __forceinline__ constexpr auto num_col_tiles_at_level(
      const int& hierarchy) const {
    if (hierarchy == 0) {
      return 1;
    }

    return num_child_col_tiles(hierarchy - 1);
  }

  template <typename tile_index_t>
  __device__ __forceinline__ constexpr auto num_row_tiles_at_level(
      const tile_index_t& tile_index) const {
    return num_row_tiles_at_level(tile_index.getHierarchy());
  }

  template <typename tile_index_t>
  __device__ __forceinline__ constexpr auto num_col_tiles_at_level(
      const tile_index_t& tile_index) const {
    return num_col_tiles_at_level(tile_index.getHierarchy());
  }

  // Not constexpr since the point changes at runtime
  template <typename point_t, typename tile_index_t, typename hierarchy_t>
  __device__ __forceinline__ auto remap_point(point_t point,
                                              tile_index_t tile_index,
                                              hierarchy_t goal_hierarchy) {
    if (tile_index.getHierarchy() < goal_hierarchy) {
      auto new_point = point;

#pragma unroll
      for (auto h_idx = tile_index.getHierarchy(); h_idx < goal_hierarchy;
           h_idx++) {
        new_point.row %= rows_in_tile(h_idx + 1);
        new_point.col %= cols_in_tile(h_idx + 1);
      }

      return new_point;

    } else if (tile_index.getHierarchy() > goal_hierarchy) {
      auto new_point = point;
      // Going from a small to a big tile
#pragma unroll
      for (auto h_idx = tile_index.getHierarchy(); h_idx > goal_hierarchy;
           h_idx--) {
        new_point.row += tile_index.row[h_idx] * rows_in_tile(h_idx);
        new_point.col += tile_index.col[h_idx] * cols_in_tile(h_idx);
      }

      return new_point;
    } else {
      return point;
    }
  }

  tiledim_t tiledims;
};

template <typename rowdim_t, typename coldim_t>
__host__ __device__ __forceinline__ constexpr auto make_layout(
    rowdim_t rowdim,
    coldim_t coldim) {
  std::tuple<rowdim_t, coldim_t> tiledim{rowdim, coldim};
  std::tuple<decltype(tiledim)> tiledim_wrapper{tiledim};
  return Layout<decltype(tiledim_wrapper)>(tiledim_wrapper);
}

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
__host__ __device__ __forceinline__ constexpr auto
make_layout(rowdim_t rowdim, coldim_t coldim, parentlayout_t parentlayout) {
  std::tuple<rowdim_t, coldim_t> tiledim{rowdim, coldim};
  std::tuple<decltype(tiledim)> tiledim_wrapper{tiledim};

  // concatenate parentlayout and tiledim tuples
  auto tiledim_wrapper_nested =
      std::tuple_cat(parentlayout.tiledims, tiledim_wrapper);

  return Layout<decltype(tiledim_wrapper_nested)>(tiledim_wrapper_nested);
}

template <typename graph_t,
          typename vector_t,
          typename shmem_t,
          typename layout_t>
class TileIterator {
 public:
  __device__ __forceinline__ TileIterator(const graph_t _graph,
                                          const vector_t* _input,
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

    row_queue =
        (int*)(&shmem_output[tile_layout.rows_in_tile(TILE_TEMPORAL_BLOCK)]);
  }

  template <typename tile_index_t>
  __device__ __forceinline__ void load_tile(tile_index_t parent_tile_idx) {
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("Loading data into shared memory\n");
    // }

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

// Iterate and copy to shared
// TODO convert to ampere async copy
// TODO use vector loads?
// TODO don't need to read in starts and stops separately. Combine for
//      efficiency
// TODO use streaming loads? (This might conflict with async copy)
#pragma unroll
    for (auto row_idx = threadIdx.x;
         row_idx < rows_in_block &&
         matrix_coord.row + row_idx < graph.get_number_of_rows();
         row_idx += blockDim.x) {
      // Copy the row offset to shared memory. Load streaming since we won't
      // reuse this
      this->shmem_row_offsets_start[row_idx] =
          __ldcs(&(this->graph.get_row_offsets()[matrix_coord.row + row_idx]));
      this->shmem_row_offsets_end[row_idx] = __ldcs(
          &(this->graph.get_row_offsets()[matrix_coord.row + row_idx + 1]));
      this->shmem_output[row_idx] = 0;
    }

    __syncthreads();
  }

  template <typename tile_index_t>
  __device__ __forceinline__ void process_tile_thread_per_row(
      tile_index_t parent_tile_idx) {
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("Processing tile\n");
    // }

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

  template <typename tile_index_t>
  __device__ __forceinline__ void process_tile_warp_per_row(
      tile_index_t parent_tile_idx) {
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

  template <typename tile_index_t>
  __device__ __forceinline__ void process_tile_warp_per_row_queue(
      tile_index_t parent_tile_idx) {
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

      // Get the next row from the queue
      if (threadIdx.x % 32 == 0) {
        row_idx = atomicAdd(&(this->row_queue[0]), 1);
      }

      // Broadcast the row idx to all other threads in the warp
      row_idx = __shfl_sync(0xffffffff, row_idx, 0);
    }
  }

  template <typename tile_index_t>
  __device__ __forceinline__ void store_tile(tile_index_t parent_tile_idx) {
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

      __stcs(&(output[matrix_coord.row + row_idx]),
             this->shmem_output[row_idx]);
    }
  }

  template <typename tile_index_t>
  __device__ __forceinline__ void process_all_tiles_at_hierarchy(
      tile_index_t parent_tile_idx) {
    // print_device("Processing tile (%d, %d) at %d\n",
    //              (int)parent_tile_idx.row[parent_tile_idx.getHierarchy()],
    //              (int)parent_tile_idx.col[parent_tile_idx.getHierarchy()],
    //              (int)parent_tile_idx.getHierarchy());

    // ===== SETUP TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_SPATIAL) {
      load_tile(parent_tile_idx);
    }

    // ===== TILE PROCESSING TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_TEMPORAL) {
      // We aren't iterating over the tile anymore, we're now processing it
      // and diving into parallel work
      // process_tile_thread_per_row(parent_tile_idx);
      // process_tile_warp_per_row(parent_tile_idx);
      process_tile_warp_per_row_queue(parent_tile_idx);

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
          process_all_tiles_at_hierarchy(child_tile_idx);
        }
      }
    }

    // ===== TEARDOWN TASKS ===== //
    if constexpr (parent_tile_idx.getHierarchy() == TILE_SPATIAL) {
      store_tile(parent_tile_idx);
    }
  }

  // Iterate all tiles within level of the hierarchy (h0 is the Matrix)
  __device__ __forceinline__ void process_all_tiles() {
    auto matrix_tile_index = make_tile_index(0, 0);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("Number of child row tiles: %d\n",
             tile_layout.num_child_row_tiles(matrix_tile_index));
    }

    process_all_tiles_at_hierarchy(matrix_tile_index);
  }

 public:
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  size_t shmem_size;

  shmem_t* shmem_row_offsets_start;
  shmem_t* shmem_row_offsets_end;
  vector_t* shmem_output;

  shmem_t* row_queue;

  layout_t tile_layout;
};
