#pragma once

#include "spmv_utils.cuh"

struct Point {
  size_t row;
  size_t col;
  __device__ Point(size_t row, size_t col) : row(row), col(col) {}
};

struct TileIdx {
  size_t row;
  size_t col;
  size_t h;
  TileIdx* parent;

  __device__ TileIdx() : row(0), col(0), h(0), parent(NULL) {}
  __device__ TileIdx(size_t _row, size_t _col, TileIdx* _parent)
      : row(_row), col(_col), h(_parent->h + 1), parent(_parent) {}

  __device__ TileIdx(size_t _row, size_t _col, size_t _h)
      : row(_row), col(_col), h(_h), parent(NULL) {}
};

#define ROWMAJOR 0
#define COLMAJOR 1

#define TILE_MATRIX 0
#define TILE_DEVICE_BATCH 1
#define TILE_DEVICE 2
#define TILE_BLOCK 3
#define TILE_ROW 4

template <typename graph_t,
          typename vector_t,
          typename shmem_t,
          typename indexer_t,
          size_t HIERARCHY_N>
class TileIterator {
 public:
  __device__ TileIterator() {}

  __device__ TileIterator(const graph_t _graph,
                          const vector_t* _input,
                          vector_t* _output,
                          const size_t _tile_row_size,
                          const size_t _tile_col_size,
                          shmem_t* _shmem,
                          const size_t _shmem_size,
                          const indexer_t _tile_indexer)
      : graph(_graph),
        input(_input),
        output(_output),
        tile_row_size(_tile_row_size),
        tile_col_size(_tile_col_size),
        shmem(_shmem),
        shmem_size(_shmem_size),
        tile_indexer(_tile_indexer) {
    shmem_row_offsets = shmem;
    shmem_output = shmem + shmem_size / 2;
  }

 public:
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  const size_t tile_row_size;
  const size_t tile_col_size;
  shmem_t* shmem;
  const size_t shmem_size;

  shmem_t* shmem_row_offsets;
  shmem_t* shmem_output;

  indexer_t tile_indexer;
};

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
#pragma unroll
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
  }

  // Number of rows in the tile given by idx
  __device__ __forceinline__ size_t rows_in_tile(const TileIdx idx) {
    return tile_row_dim[idx.h];
  }

  // Number of cols in the tile given by idx
  __device__ __forceinline__ size_t cols_in_tile(const TileIdx idx) {
    return tile_col_dim[idx.h];
  }

  // Number of row tiles at the given level of the hierarchy, where idx is a
  // pointer to a tile in that hierarchy
  // TODO is this correct? Am I going the wrong way in the hierarchy?
  __device__ __forceinline__ size_t num_row_tiles(const size_t h) {
    if (h == 0) {
      return 1;
    }

    size_t num = tile_row_dim[h - 1] / tile_row_dim[h];

    if (tile_row_dim[h - 1] % tile_row_dim[h] != 0) {
      num++;
    }

    return num;
  }

  __device__ __forceinline__ size_t num_row_tiles(const TileIdx idx) {
    return num_row_tiles(idx.h);
  }

  // Number of col tiles at the given level of the hierarchy, where idx
  // is a pointer to a tile in that hierarchy
  __device__ __forceinline__ size_t num_col_tiles(const size_t h) {
    if (h == 0) {
      return 1;
    }

    size_t num = tile_col_dim[h - 1] / tile_col_dim[h];

    if (tile_col_dim[h - 1] % tile_col_dim[h] != 0) {
      num++;
    }

    return num;
  }

  __device__ __forceinline__ size_t num_col_tiles(const TileIdx idx) {
    return num_col_tiles(idx);
  }

  // Number of child tiles for a tile at the given level of the hierarchy, where
  // idx is a pointer to a tile in that hierarchy
  __device__ __forceinline__ size_t num_child_tiles_row(const size_t h) {
    if (h == HIERARCHY_N - 1) {
      return 1;
    }

    return tile_row_dim[h] / tile_row_dim[h + 1];
  }

  __device__ __forceinline__ size_t num_child_tiles_row(const TileIdx idx) {
    return num_child_tiles_row(idx.h);
  }

  // Number of child tiles for a tile at the given level of the hierarchy, where
  // idx is a pointer to a tile in that hierarchy
  __device__ __forceinline__ size_t num_child_tiles_col(const size_t h) {
    if (h == HIERARCHY_N - 1) {
      return 1;
    }

    return tile_col_dim[h] / tile_col_dim[h + 1];
  }

  __device__ __forceinline__ size_t num_child_tiles_col(const TileIdx idx) {
    return num_child_tiles_col(idx.h);
  }

  // Given a TileIdx struct with the tile's row, column, and level in the
  // hierarchy, convert that to the row, column coordinates of the tile in the
  // goal hierarchy
  __device__ __forceinline__ Point convert_index(Point _point,
                                                 TileIdx* _idx,
                                                 size_t _goal_h) {
    // CONVERT TO ITERATIVE
    Point point = _point;
    TileIdx* idx = _idx;
    size_t goal_h = _goal_h;

    TileIdx temp_idx;

    while (idx->h != goal_h) {
      // Recursive implementation.
      if (goal_h < idx->h) {
        // Go from a small tile to a large tile
        Point larger_point(point.row + idx->row * tile_row_dim[idx->h],
                           point.col + idx->col * tile_col_dim[idx->h]);

        point = larger_point;
        idx = idx->parent;

      } else if (goal_h > idx->h) {
        // Go from a large tile to a small tile
        // Calculate the point in the immediately smaller tile
        Point smaller_point(point.row % tile_row_dim[idx->h + 1],
                            point.col % tile_col_dim[idx->h + 1]);

        temp_idx.row = point.row / tile_row_dim[idx->h + 1];
        temp_idx.col = point.col / tile_col_dim[idx->h + 1];
        temp_idx.h = idx->h + 1;
        temp_idx.parent = NULL;

        point = smaller_point;
        idx = &temp_idx;
      } else {
        printf("ERROR\n");
      }
    }

    return point;
  }

  // Create arrays here to hold information about tile
  size_t tile_row_dim[HIERARCHY_N + 1];
  size_t tile_col_dim[HIERARCHY_N + 1];
  size_t format;
};

// Make a class that inherits from
template <typename graph_t,
          typename vector_t,
          typename shmem_t,
          typename indexer_t,
          size_t HIERARCHY_N>
class MatrixTileIterator
    : public TileIterator<graph_t, vector_t, shmem_t, indexer_t, HIERARCHY_N> {
 public:
  __device__ __forceinline__ MatrixTileIterator(const graph_t _graph,
                                                const vector_t* _input,
                                                vector_t* _output,
                                                const size_t _tile_row_size,
                                                const size_t _tile_col_size,
                                                shmem_t* _shmem,
                                                const size_t _shmem_size,
                                                indexer_t _tile_indexer)
      : TileIterator<graph_t, vector_t, shmem_t, indexer_t, HIERARCHY_N>(
            _graph,
            _input,
            _output,
            _tile_row_size,
            _tile_col_size,
            _shmem,
            _shmem_size,
            _tile_indexer) {}

  __device__ __forceinline__ void load_tile(TileIdx parent_tile_idx) {
    // Save the row offsets to shared memory and initialize the outputs to 0
    // Load data into shared memory

    // If we're loading the first tile in a batch, the device tile is by default
    // (0, 0) relative to the parent

    TileIdx device_tile_idx(0, 0, &parent_tile_idx);

    // Then, we need to get the block idx relative to the device tile
    TileIdx block_tile_idx(blockIdx.x, 0, &device_tile_idx);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("Loading data into shared memory\n");
    }

    // Then iterate over the number of rows in the block
    // NOTE: Use ampere async copy
    size_t rows_in_block = this->tile_indexer.rows_in_tile(block_tile_idx);

    // Convert coordinates relative to one tile to coordinates relative to
    // another tile. Note that we need to use block_tile_idx since it has
    // references to its parents.
    Point tile_point(0, 0);
    Point matrix_idx = this->tile_indexer.convert_index(
        tile_point, &block_tile_idx, (size_t)TILE_MATRIX);

    // size_t global_mem_idx = tile_indexer.t2g(matrix_idx);

    // 1. Get this block's tile index relative to the parent tile

    // 2. Iterate over the rows assigned to this block and store them in shared
    //    memory. Note that we need to do an address conversion to global

    // 3. Initialize the output to 0
  }

  __device__ __forceinline__ void process_tile(TileIdx parent_tile_idx) {
    // Iterate over sub tiles
    // Basically, here we need another tile iterator. This time for "device"
    // tiles. These _should_ be really easy to implement
  }

  __device__ __forceinline__ void store_tile(TileIdx parent_tile_idx) {
    // Write the outputs to the output vector
    // Unload data from shared memory
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("Unloading data from shared memory\n");
    }
  }

  __device__ __forceinline__ void process_all_tiles_at_hierarchy(
      TileIdx parent_tile_idx) {
    print_device("Processing tile (%d, %d) at %d\n", (int)parent_tile_idx.row,
                 (int)parent_tile_idx.col, (int)parent_tile_idx.h);

    // ===== SETUP TASKS ===== //
    if (parent_tile_idx.h == TILE_DEVICE_BATCH) {
      load_tile(parent_tile_idx);
    }

    // ===== TILE PROCESSING TASKS ===== //
    if (parent_tile_idx.h == TILE_DEVICE) {
      // We aren't iterating over the tile anymore, we're now processing it
      // and diving into parallel work
      process_tile(parent_tile_idx);
    } else {
      // Tile indexer to the child tiles if the parent tile
      TileIdx child_tile_idx(0, 0, &parent_tile_idx);

      // Iterate over the child tiles
      for (child_tile_idx.row == 0;
           child_tile_idx.row <
           this->tile_indexer.num_row_tiles(child_tile_idx);
           child_tile_idx.row++) {
        for (child_tile_idx.col = 0;
             child_tile_idx.col <
             this->tile_indexer.num_col_tiles(child_tile_idx);
             child_tile_idx.col++) {
          process_all_tiles_at_hierarchy(child_tile_idx);
        }
      }
    }

    // ===== TEARDOWN TASKS ===== //
    if (parent_tile_idx.h == TILE_DEVICE_BATCH) {
      store_tile(parent_tile_idx);
    }
  }

  // Iterate all tiles within level of the hierarchy (h0 is the Matrix)
  __device__ __forceinline__ void process_all_tiles() {
    TileIdx tile_idx(0, 0, (size_t)TILE_MATRIX);
    process_all_tiles_at_hierarchy(tile_idx);
  }
};