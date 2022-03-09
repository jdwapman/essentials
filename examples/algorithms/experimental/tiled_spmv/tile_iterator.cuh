#pragma once

#include "spmv_utils.cuh"
#include <tuple>

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
class Layout {
 public:
  __host__ __device__
      __forceinline__ constexpr Layout(const rowdim_t& rowdim,
                                       const coldim_t& coldim,
                                       const parentlayout_t& parentlayout)
      : rowdim(rowdim), coldim(coldim), parentlayout(parentlayout) {}

  template <typename _rowdim_t, typename _coldim_t>
  __host__ __device__ __forceinline__ constexpr auto tile(
      const _rowdim_t rowdim,
      const _coldim_t coldim) {
    return make_layout(rowdim, coldim, *this);
  }

  __host__ __device__ __forceinline__ constexpr bool has_parent() const {
    constexpr bool is_same = std::is_same<parentlayout_t, std::tuple<>>::value;

    return !is_same;
  }

  rowdim_t rowdim;
  coldim_t coldim;
  parentlayout_t parentlayout;
};

template <typename rowdim_t, typename coldim_t>
__host__ __device__ __forceinline__ constexpr auto make_layout(
    rowdim_t rowdim,
    coldim_t coldim) {
  std::tuple<> blank_tuple;
  return Layout<rowdim_t, coldim_t, std::tuple<>>(rowdim, coldim, blank_tuple);
}

template <typename rowdim_t, typename coldim_t, typename parentlayout_t>
__host__ __device__ __forceinline__ constexpr auto
make_layout(rowdim_t rowdim, coldim_t coldim, parentlayout_t parentlayout) {
  return Layout<rowdim_t, coldim_t, parentlayout_t>(rowdim, coldim,
                                                    parentlayout);
}

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

  __host__ __device__ __forceinline__ constexpr bool getHierarchy() const {
    return hierarchy;
  }

  __host__ __device__ __forceinline__ constexpr auto num_row_children() const {
    return 0;
  }

  __host__ __device__ __forceinline__ constexpr auto num_col_children() const {
    return 0;
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

#define ROWMAJOR 0
#define COLMAJOR 1

#define TILE_MATRIX 0
#define TILE_DEVICE_BATCH 1
#define TILE_DEVICE 2
#define TILE_BLOCK 3

// // Helper class used to convert between layers of a tile hierarchy.
// // For example, if I'm at block tile 3 within a device tile, what
// // is the overall block tile IDX?
// // Note that 0 is the largest level of the hierarchy
// // Included functions:
// // 1. Given row,col coordinates with respect to one tile, get them with
// respect
// // to another tile
// // 2. Given row,col coordinates with respect to one tile, get the
// //    offset within the array
// // 3. Given the offset within an array, get the row,col coordates with
// respect
// // to another tile
// template <size_t HIERARCHY_N>
// class TileIndexer {
//  public:
//   __device__ TileIndexer(const size_t _format) {
// // Init arrays to 0
// #pragma unroll
//     for (size_t i = 0; i < HIERARCHY_N; i++) {
//       tile_row_dim[i] = 0;
//       tile_col_dim[i] = 0;
//     }
//     format = _format;
//   }

//   __device__ __forceinline__ void add_tile_info(const size_t hierarchy,
//                                                 const size_t tile_rows,
//                                                 const size_t tile_cols) {
//     tile_row_dim[hierarchy] = tile_rows;
//     tile_col_dim[hierarchy] = tile_cols;

//     if (hierarchy >= 1) {
//       tile_row_dim[hierarchy] =
//           min(tile_row_dim[hierarchy], tile_row_dim[hierarchy - 1]);
//       tile_col_dim[hierarchy] =
//           min(tile_col_dim[hierarchy], tile_col_dim[hierarchy - 1]);
//     }
//   }

//   // Number of rows in the tile given by idx
//   __device__ __forceinline__ size_t rows_in_tile(const TileIdx idx) {
//     return tile_row_dim[idx.h];
//   }

//   // Number of cols in the tile given by idx
//   __device__ __forceinline__ size_t cols_in_tile(const TileIdx idx) {
//     return tile_col_dim[idx.h];
//   }

//   // Number of row tiles at the given level of the hierarchy, where idx is a
//   // pointer to a tile in that hierarchy
//   // TODO is this correct? Am I going the wrong way in the hierarchy?
//   __device__ __forceinline__ size_t num_row_tiles(const size_t h) {
//     if (h == 0) {
//       return 1;
//     }

//     size_t num = tile_row_dim[h - 1] / tile_row_dim[h];

//     if (tile_row_dim[h - 1] % tile_row_dim[h] != 0) {
//       num++;
//     }

//     return num;
//   }

//   __device__ __forceinline__ size_t num_row_tiles(const TileIdx idx) {
//     return num_row_tiles(idx.h);
//   }

//   // Number of col tiles at the given level of the hierarchy, where idx
//   // is a pointer to a tile in that hierarchy
//   __device__ __forceinline__ size_t num_col_tiles(const size_t h) {
//     if (h == 0) {
//       return 1;
//     }

//     size_t num = tile_col_dim[h - 1] / tile_col_dim[h];

//     if (tile_col_dim[h - 1] % tile_col_dim[h] != 0) {
//       num++;
//     }

//     return num;
//   }

//   __device__ __forceinline__ size_t num_col_tiles(const TileIdx idx) {
//     return num_col_tiles(idx.h);
//   }

//   // Number of child tiles for a tile at the given level of the hierarchy,
//   where
//   // idx is a pointer to a tile in that hierarchy
//   __device__ __forceinline__ size_t num_child_tiles_row(const size_t h) {
//     if (h == HIERARCHY_N - 1) {
//       return 1;
//     }

//     return tile_row_dim[h] / tile_row_dim[h + 1];
//   }

//   __device__ __forceinline__ size_t num_child_tiles_row(const TileIdx idx) {
//     return num_child_tiles_row(idx.h);
//   }

//   // Number of child tiles for a tile at the given level of the hierarchy,
//   where
//   // idx is a pointer to a tile in that hierarchy
//   __device__ __forceinline__ size_t num_child_tiles_col(const size_t h) {
//     if (h == HIERARCHY_N - 1) {
//       return 1;
//     }

//     return tile_col_dim[h] / tile_col_dim[h + 1];
//   }

//   __device__ __forceinline__ size_t num_child_tiles_col(const TileIdx idx) {
//     return num_child_tiles_col(idx.h);
//   }

//   // Given a TileIdx struct with the tile's row, column, and level in the
//   // hierarchy, convert that to the row, column coordinates of the tile in
//   the
//   // goal hierarchy
//   __device__ __forceinline__ Point convert_index(Point _point,
//                                                  TileIdx* _idx,
//                                                  size_t _goal_h) {
//     // CONVERT TO ITERATIVE
//     Point point = _point;
//     TileIdx* idx = _idx;
//     size_t goal_h = _goal_h;

//     TileIdx temp_idx;

//     while (idx->h != goal_h) {
//       // Recursive implementation.
//       if (goal_h < idx->h) {
//         // Go from a small tile to a large tile
//         Point larger_point(point.row + idx->row * tile_row_dim[idx->h],
//                            point.col + idx->col * tile_col_dim[idx->h]);

//         point = larger_point;
//         idx = idx->parent;

//       } else if (goal_h > idx->h) {
//         // Go from a large tile to a small tile
//         // Calculate the point in the immediately smaller tile
//         Point smaller_point(point.row % tile_row_dim[idx->h + 1],
//                             point.col % tile_col_dim[idx->h + 1]);

//         temp_idx.row = point.row / tile_row_dim[idx->h + 1];
//         temp_idx.col = point.col / tile_col_dim[idx->h + 1];
//         temp_idx.h = idx->h + 1;
//         temp_idx.parent = NULL;

//         point = smaller_point;
//         idx = &temp_idx;
//       } else {
//         printf("ERROR\n");
//       }
//     }

//     return point;
//   }

//   __device__ __forceinline__ size_t point2address(Point _point) {
//     return _point.row * tile_row_dim[0] + _point.col;
//   }

//   // Create arrays here to hold information about tile
//   size_t tile_row_dim[HIERARCHY_N + 1];
//   size_t tile_col_dim[HIERARCHY_N + 1];
//   size_t format;
// };

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
  //   // Save the row offsets to shared memory and initialize the outputs to 0
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
  //   // Basically, here we need another tile iterator. This time for "device"
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
           child_tile_idx.row < parent_tile_idx.num_row_children();
           child_tile_idx.row++) {
#pragma unroll
        for (child_tile_idx.col = 0;
             child_tile_idx.col < parent_tile_idx.num_col_children();
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
