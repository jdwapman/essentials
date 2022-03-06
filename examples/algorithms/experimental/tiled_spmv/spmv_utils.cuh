#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
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

#define ROWMAJOR 0
#define COLMAJOR 1

// Helper class used to convert between layers of a tile hierarchy.
// For example, if I'm at block tile 3 within a device tile, what
// is the overall block tile IDX?
// Note that 0 is the largest level of the hierarchy
// Included functions:
// 1. Given row,col coordinates with respect to one tile, get them with respect
// to another tile
// 2. Given row,col coordinates with respect to one tile, get the
//    offset within the array
// 3. Given the offset within an array, get the row,col coordates with respect
// to another tile
template <size_t HIERARCHY_N>
class TileIndexer {
 public:
  __device__ TileIndexer(const size_t _format) {
    // Init arrays to 0
    for (size_t i = 0; i < HIERARCHY_N; i++) {
      tile_row_dim[i] = 0;
      tile_col_dim[i] = 0;
    }
    format = _format;
  }

  __device__ __forceinline__ void add_tile_info(const size_t hierarchy,
                                                const size_t tile_rows,
                                                const size_t tile_cols) {
    tile_row_dim[hierarchy] = tile_rows;
    tile_col_dim[hierarchy] = tile_cols;

    if (hierarchy >= 1) {
      tile_row_dim[hierarchy] =
          min(tile_row_dim[hierarchy], tile_row_dim[hierarchy - 1]);
      tile_col_dim[hierarchy] =
          min(tile_col_dim[hierarchy], tile_col_dim[hierarchy - 1]);
    }

    print_device("Tile (%d x %d) added to hierarchy %d\n",
                 (int)tile_row_dim[hierarchy], (int)tile_col_dim[hierarchy],
                 (int)hierarchy);
  }

  //   __device__ __forceinline__ size_t
  //   num_sub_tiles(const size_t hierarchy_upper, const size_t hierarchy_lower)
  //   {
  //     // Do some recursive calls to get the number of sub tiles
  //     size_t num_sub_tiles = 0;
  //     if (hierarchy_upper > hierarchy_lower) {
  //       num_sub_tiles = num_sub_tiles(hierarchy_upper - 1, hierarchy_lower) *
  //                       tile_row_dim[hierarchy_upper - 1] *
  //                       tile_col_dim[hierarchy_upper - 1];
  //     } else if (hierarchy_upper == hierarchy_lower) {
  //       num_sub_tiles = 1;
  //     }
  //   }

  __device__ __forceinline__ size_t rows_in_tile(const size_t hierarchy) {
    return tile_row_dim[hierarchy];
  }

  __device__ __forceinline__ size_t cols_in_tile(const size_t hierarchy) {
    return tile_col_dim[hierarchy];
  }

  //   // Tile-to-tile (x,y)
  //   __device__ __forceinline__ void t2t_rowcol(const size_t& row1,
  //                                              const size_t& row22,
  //                                              const size_t& col1,
  //                                              const size_t& col2,
  //                                              const size_t hierarchy_1,
  //                                              const size_t hierarchy_2) {}

  //   // Tile (x,y) to array offset
  //   __device__ __forceinline__ size_t t2a_offset(const size_t& row,
  //                                                const size_t& col,
  //                                                const size_t hierarchy) {
  //     // Before we can convert to an array offset, we first need to know
  //     // the global row,col coordinates at the last level of the hierarchy
  //     size_t global_row = 0;
  //     size_t global_col = 0;

  //     t2t_rowcol(row, col, global_row, global_col, hierarchy, 0);

  //     size_t offset = 0;

  //     if (format == ROWMAJOR) {
  //       offset = global_row * tile_col_dim[0] + global_col;
  //     } else if (format == COLMAJOR) {
  //       offset = global_col * tile_row_dim[0] + global_row;
  //     } else {
  //       offset = -1;
  //       printf("ERROR: Unknown format\n");
  //       assert(0);
  //     }

  //     return offset;
  //   }

  // Number of row tiles at the given level of the hierarchy
  __device__ __forceinline__ size_t num_row_tiles(const size_t hierarchy) {
    size_t num = 0;
    num = tile_row_dim[hierarchy] / tile_row_dim[hierarchy + 1];

    if (tile_row_dim[hierarchy] % tile_row_dim[hierarchy + 1] != 0) {
      num++;
    }

    return num;
  }

  // Number of col tiles at the given level of the hierarchy
  __device__ __forceinline__ size_t num_col_tiles(const size_t hierarchy) {
    size_t num = 0;
    num = tile_col_dim[hierarchy] / tile_col_dim[hierarchy + 1];

    if (tile_col_dim[hierarchy] % tile_col_dim[hierarchy + 1] != 0) {
      num++;
    }

    return num;
  }

 private:
  // Create arrays here to hold information about tile
  size_t tile_row_dim[HIERARCHY_N + 1];
  size_t tile_col_dim[HIERARCHY_N + 1];
  size_t format;
};

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
//   // Note that this function does not attempt to perform bounds checking
//   for the
//   // final tile

//   index_t global_idx = local_idx + (tile_idx * tile_size);

//   return global_idx;
// }
