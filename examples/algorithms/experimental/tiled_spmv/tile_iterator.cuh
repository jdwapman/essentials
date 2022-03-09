#pragma once

#include "spmv_utils.cuh"
#include <tuple>

#define TILE_MATRIX 0
#define TILE_DEVICE_BATCH 1
#define TILE_DEVICE 2
#define TILE_BLOCK 343

template <typename row_t, typename col_t>
struct Point {
  row_t row;
  col_t col;
  __device__ __forceinline__ Point(row_t _row, col_t _col)
      : row(_row), col(_col) {}
};

template <typename row_t, typename col_t, typename parent_t, int hierarchy>
struct TileIdx {
  row_t row;
  col_t col;
  parent_t parent;

  __host__ __device__ __forceinline__ TileIdx(row_t _row,
                                              col_t _col,
                                              parent_t _parent)
      : row(_row), col(_col), parent(_parent) {}

  __host__ __device__ __forceinline__ constexpr bool operator==(
      const int comparare_hierarchy) const {
    return hierarchy == comparare_hierarchy;
  }

  __host__ __device__ __forceinline__ constexpr auto getHierarchy() const {
    return hierarchy;
  }
};

template <typename row_t, typename col_t>
__host__ __device__ __forceinline__ constexpr auto make_tile_index(row_t row,
                                                                   col_t col) {
  std::tuple<> blank_tuple;
  return TileIdx<row_t, col_t, std::tuple<>, 0>(row, col, blank_tuple);
}

template <typename row_t, typename col_t, typename parent_t>
__host__ __device__ __forceinline__ constexpr auto
make_tile_index(row_t row, col_t col, parent_t parent) {
  return TileIdx<row_t, col_t, parent_t, parent.getHierarchy() + 1>(row, col,
                                                                    parent);
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
    return make_layout(rowdim, coldim, this->tiledims);
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

  template <typename layout_t, typename tile_index_t>
  __device__ __forceinline__ constexpr auto rows_in_tile(
      const layout_t layout,
      const tile_index_t tile_index) const {
    return rows_in_tile(layout, tile_index.getHierarchy());
  }

  template <typename layout_t, typename tile_index_t>
  __device__ __forceinline__ constexpr auto cols_in_tile(
      const layout_t layout,
      const tile_index_t tile_index) const {
    return cols_in_tile(layout, tile_index.getHierarchy());
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

  // // Not constexpr since the point changes at runtime
  // template <typename point_t, typename tile_index_t, typename hierarchy_t>
  // __device__ __forceinline__ auto remap_point(point_t point,
  //                                             tile_index_t tile_index,
  //                                             hierarchy_t goal_hierarchy) {
  //   if (tile_index.getHierarchy() < goal_hierarchy) {
  //     // Going from a big tile to a small tile

  //     Point smaller_point(
  //         point.row % rows_in_tile(tile_index.getHierarchy() + 1),
  //         point.col % cols_in_tile(tile_index.getHierarchy()) + 1);

  //     auto smaller_tile_idx = make_tile_idx(
  //         point.row / rows_in_tile(tile_index.getHierarchy() + 1),
  //         point.col / cols_in_tile(tile_index.getHierarchy() + 1),
  //         tile_index);

  //     return remap_point(smaller_point, smaller_tile_idx, goal_hierarchy);

  //   } else if (tile_index.getHierarchy() > goal_hierarchy) {
  //     // Going from a small to a big tile
  //     Point larger_point(
  //         point.row + tile_index.row *
  //         rows_in_tile(tile_index.getHierarchy()), point.col + tile_index.col
  //         * cols_in_tile(tile_index.getHierarchy()));

  //     auto larger_tile_idx = tile_index.parent;

  //     return remap_point(larger_point, larger_tile_idx, goal_hierarchy);
  //   } else {
  //     return point;
  //   }
  // }

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
/*
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
                                          const size_t _shmem_size,
                                          layout_t _tile_layout)
      : graph(_graph),
        input(_input),
        output(_output),
        shmem(_shmem),
        shmem_size(_shmem_size),
        tile_layout(_tile_layout) {
    // TODO make sure shmem is aligned
    shmem_row_offsets = shmem;
    // shmem_output = &shmem[tile_row_size];

    // print_device("shmem_size: %d\n", (int)shmem_size);
    // print_device("tile_row_size: %d\n", (int)tile_row_size);
  }

  // __device__ __forceinline__ void load_tile(TileIdx parent_tile_idx) {
  //   // Save the row offsets to shared memory and initialize the outputs to
  //   0
  //   // Load data into shared memory

  //   // If we're loading the first tile in a batch, the device tile is by
  //   default
  //       // (0, 0) relative to the parent
  //       TileIdx device_tile_idx(0, 0, &parent_tile_idx);

  //   // Then, we need to get the block idx relative to the device tile
  //   TileIdx block_tile_idx(blockIdx.x, 0, &device_tile_idx);

  //   if (threadIdx.x == 0 && blockIdx.x == 0) {
  //     printf("Loading data into shared memory\n");
  //   }

  //   // Then iterate over the number of rows in the block
  //   // NOTE: Use ampere async copy
  //   size_t rows_in_block = this->tile_indexer.rows_in_tile(block_tile_idx);

  //   // Convert coordinates relative to one tile to coordinates relative to
  //   // another tile. Note that we need to use block_tile_idx since it has
  //   // references to its parents.
  //   Point matrix_idx = this->tile_indexer.convert_index(
  //       Point(0, 0), &block_tile_idx, (size_t)TILE_MATRIX);

  //   // Iterate and copy to shared
  //   // TODO convert to ampere async copy
  //   // TODO unroll this
  //   for (size_t row_idx = threadIdx.x; row_idx < rows_in_block;
  //        row_idx += blockDim.x) {
  //     // Copy the row offset to shared memory
  //     this->shmem_row_offsets[row_idx] =
  //         this->graph.get_row_offsets()[matrix_idx.row + row_idx];
  //     this->shmem_output[row_idx] = 0;
  //   }
  // }

  // __device__ __forceinline__ void process_tile(TileIdx parent_tile_idx) {
  //   // Iterate over sub tiles
  //   // Basically, here we need another tile iterator. This time for
  //   "device"
  //   // tiles. These _should_ be really easy to implement
  // }

  // __device__ __forceinline__ void store_tile(TileIdx parent_tile_idx) {
  //   // Write the outputs to the output vector
  //   // Unload data from shared memory
  //   if (threadIdx.x == 0 && blockIdx.x == 0) {
  //     printf("Unloading data from shared memory\n");
  //   }

  //   // If we're unloading the last tile in a batch, the device tile is by
  //   // default (0, N-1) relative to the parent
  //   TileIdx device_tile_idx(
  //       0, this->tile_indexer.num_child_tiles_col(TILE_MATRIX) - 1,
  //       &parent_tile_idx);

  //   // Then, we need to get the block idx relative to the device tile
  //   TileIdx block_tile_idx(blockIdx.x, 0, &device_tile_idx);

  //   // Then iterate over the number of rows in the block
  //   size_t rows_in_block = this->tile_indexer.rows_in_tile(block_tile_idx);

  //   // Convert coordinates relative to one tile to coordinates relative to
  //   // another tile. Note that we need to use block_tile_idx since it has
  //   // references to its parents.
  //   Point matrix_idx = this->tile_indexer.convert_index(
  //       Point(0, 0), &block_tile_idx, (size_t)TILE_MATRIX);

  //   // Iterate and copy from shared to global
  //   for (size_t row_idx = threadIdx.x; row_idx < rows_in_block;
  //        row_idx += blockDim.x) {
  //     this->graph.get_row_offsets()[matrix_idx.row + row_idx] =
  //         this->shmem_row_offsets[row_idx];

  //     // this->output[matrix_idx.row + row_idx] =
  //     this->shmem_output[row_idx];
  //   }
  // }

  template <typename tile_index_t>
  __device__ __forceinline__ void process_all_tiles_at_hierarchy(
      tile_index_t parent_tile_idx) {
    print_device("Processing tile (%d, %d)\n", (int)parent_tile_idx.row,
                 (int)parent_tile_idx.col);

    // ===== SETUP TASKS ===== //
    if constexpr (parent_tile_idx == TILE_DEVICE_BATCH) {
      // load_tile(parent_tile_idx);
    }

    // ===== TILE PROCESSING TASKS ===== //
    if constexpr (parent_tile_idx == TILE_DEVICE) {
      // We aren't iterating over the tile anymore, we're now processing it
      // and diving into parallel work
      // process_tile(parent_tile_idx);
    } else {
      // Tile indexer to the child tiles if the parent tile
      auto child_tile_idx = make_tile_index(0, 0, parent_tile_idx);

// Iterate over the child tiles
#pragma unroll
      for (child_tile_idx.row == 0;
           child_tile_idx.row <
           tile_layout.num_child_row_tiles(parent_tile_idx);
           child_tile_idx.row++) {
#pragma unroll
        for (child_tile_idx.col = 0;
             child_tile_idx.col <
             tile_layout.num_child_col_tiles(parent_tile_idx);
             child_tile_idx.col++) {
          process_all_tiles_at_hierarchy(child_tile_idx);
        }
      }
    }

    // ===== TEARDOWN TASKS ===== //
    if constexpr (parent_tile_idx == TILE_DEVICE_BATCH) {
      // store_tile(parent_tile_idx);
    }
  }

  // Iterate all tiles within level of the hierarchy (h0 is the Matrix)
  __device__ __forceinline__ void process_all_tiles() {
    auto matrix_tile_index = make_tile_index(0, 0);
    process_all_tiles_at_hierarchy(matrix_tile_index);
  }

 public:
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  const size_t shmem_size;

  shmem_t* shmem_row_offsets;
  shmem_t* shmem_output;

  layout_t tile_layout;
};
*/