#pragma once

#include "spmv_utils.cuh"

struct TileIdx {
  size_t row;
  size_t col;
  size_t h;

 public:
  __device__ TileIdx() : row(0), col(0), h(0) {}
  __device__ TileIdx(size_t _row, size_t _col, size_t _h)
      : row(_row), col(_col), h(_h) {}
};

#define ROWMAJOR 0
#define COLMAJOR 1

#define TILE_MATRIX 0
#define TILE_DEVICE 1
#define TILE_BLOCK 2
#define TILE_THREAD 3

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
        tile_indexer(_tile_indexer) {}

  __device__ __forceinline__ bool all_row_tiles_finished() {}

  __device__ __forceinline__ bool all_col_tiles_finished() {}

  __device__ __forceinline__ bool all_tiles_finished() {}

  __device__ __forceinline__ size_t nonzeros_in_tile() {}

 public:
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  const size_t tile_row_size;
  const size_t tile_col_size;
  shmem_t* shmem;
  const size_t shmem_size;

  indexer_t tile_indexer;
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

  __device__ __forceinline__ void load_tile() {
    // Save the row offsets to shared memory and initialize the outputs to 0
  }

  __device__ __forceinline__ void process_tile() {
    // Iterate over sub tiles
    // Basically, here we need another tile iterator. This time for "device"
    // tiles. These _should_ be really easy to implement
  }

  __device__ __forceinline__ void store_tile() {
    // Write the outputs to the output vector
  }

  __device__ __forceinline__ void process_all_tiles_at_hierarchy(
      size_t h,
      TileIdx parent_tile_idx) {
    if (h == HIERARCHY_N - 1) {
      // We aren't iterating over the tile anymore, we're now processing it
      print_device("At block tile %d\n", (int)h);
    } else {
      // Need some "if"s here for loading and unloading the tiles

      // TODO convert to an index relative to the parent
      TileIdx tile_idx(0, 0, h);
      for (tile_idx.row == 0;
           tile_idx.row < this->tile_indexer.num_row_tiles(h); tile_idx.row++) {
        for (tile_idx.col = 0;
             tile_idx.col < this->tile_indexer.num_col_tiles(h);
             tile_idx.col++) {
          print_device("Processing tile (%d, %d) at %d\n", (int)tile_idx.row,
                       (int)tile_idx.col, (int)h);
          process_all_tiles_at_hierarchy(h + 1, tile_idx);
        }
      }
    }
  }

  // Iterate all tiles within level of the hierarchy (h0 is the Matrix)
  __device__ __forceinline__ void process_all_tiles() {
    TileIdx tile_idx(0, 0, 0);
    process_all_tiles_at_hierarchy(0, tile_idx);
  }
};