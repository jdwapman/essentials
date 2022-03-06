template <typename graph_t, typename vector_t, typename shmem_t>
class TileIterator {
 public:
  __device__ TileIterator(const graph_t _graph,
                          const vector_t* _input,
                          vector_t* _output,
                          const size_t _tile_row_size,
                          const size_t _tile_col_size,
                          shmem_t* _shmem,
                          const size_t _shmem_size)
      : graph(_graph),
        input(_input),
        output(_output),
        tile_row_size(_tile_row_size),
        tile_col_size(_tile_col_size),
        shmem(_shmem),
        shmem_size(_shmem_size) {
    rows_per_block_tile = _rows_per_block_tile;
    rows_per_gpu_tile = _rows_per_block_tile * gridDim.x;

    cur_row_tile_idx = 0;
    cur_col_tile_idx = 0;

    local_row_offsets_start = shmem;
  }

  //   __device__ __forceinline__ index_t
  //   row_elems_in_tile(const index_t &global_row_idx, const index_t
  //   &block_row_idx,
  //                     const index_t &tile_boundary)
  //   {
  //     assert(store_end_offsets_in_shmem);

  //     index_t total_remaining_nonzeros = 0;

  //     index_t begin = local_row_offsets_start[block_row_idx];
  //     index_t end = local_row_offsets_end[block_row_idx];

  //     while (begin < end)
  //     {
  //       index_t mid = floor((begin + end) / 2);

  //       index_t col_key = col_idx[mid];

  //       if (col_key > tile_boundary)
  //       {
  //         end = mid;
  //       }
  //       else
  //       {
  //         begin = mid + 1;
  //       }
  //     }

  //     index_t actual_end = end - 1;

  //     return (actual_end - local_row_offsets_start[block_row_idx]);
  //   }

  //   __device__ __forceinline__ bool all_device_tiles_finished()
  //   {
  //     // How many evenly-sized tiles are there?
  //     index_t number_of_gpu_tiles_in_matrix = (num_rows / rows_per_gpu_tile);

  //     // Remainder tile
  //     if (num_rows % rows_per_gpu_tile)
  //       number_of_gpu_tiles_in_matrix++;

  //     // cur_row_tile_idx incremented only after the primary tile is finished
  //     // (with all its member column tiles)
  //     if (cur_row_tile_idx >= number_of_gpu_tiles_in_matrix)
  //     {
  //       return true;
  //     }
  //     else
  //     {
  //       return false;
  //     }
  //   }

  //   __device__ __forceinline__ void load_device_tile()
  //   {
  //     if (blockIdx.x == 0 && threadIdx.x == 0 && debug)
  //     {
  //       printf("Loading Metadata for tile (%d,...) into shmem\n",
  //              cur_row_tile_idx);
  //     }
  //     // Need to simultaneously keep track of the current row in the tile as
  //     well
  //     // as the row index in the global coordinates

  //     int cur_row_in_gpu_tile = blockIdx.x * rows_per_block_tile +
  //     threadIdx.x; int cur_row_in_matrix =
  //         tile2global(cur_row_in_gpu_tile, cur_row_tile_idx,
  //         rows_per_gpu_tile);

  //     int cur_row_in_block_tile = threadIdx.x;

  //     int stride = blockDim.x;

  //     // Iterate over all rows in the current tile
  //     for (; cur_row_in_matrix < num_rows &&
  //            cur_row_in_block_tile < rows_per_block_tile;
  //          cur_row_in_matrix += stride, cur_row_in_block_tile += stride,
  //          cur_row_in_gpu_tile += stride)
  //     {
  //       local_row_offsets_start[cur_row_in_block_tile] =
  //           row_offsets[cur_row_in_matrix];

  //       if (store_end_offsets_in_shmem)
  //       {
  //         local_row_offsets_end[cur_row_in_block_tile] =
  //             row_offsets[cur_row_in_matrix + 1];
  //       }
  //       // printf(
  //       //     "Block %d Loading matrix row %d block tile idx %d gpu tile idx
  //       %d "
  //       //     "offset %d\n",
  //       //     blockIdx.x, cur_row_in_matrix, cur_row_in_block_tile,
  //       //     cur_row_in_gpu_tile,
  //       local_row_offsets[cur_row_in_block_tile]);
  //     }

  //     __syncthreads();
  //   }

  //   __device__ __forceinline__ void load_block_tile(){};

  //   __device__ __forceinline__ void evict_device_tile() {}

  //   __device__ __forceinline__ void evict_block_tile()
  //   {
  //     // In the src-first implementation, there is nothing to do for this
  //     function
  //     // except maybe resetting the L2 cache
  //   }

  __device__ __forceinline__ bool device_tile_finished() {
    return (cur_col_tile_idx >= tile_indexer.get_num_col_tiles(TILE_DEVICE));
  };

  __device__ __forceinline__ void process_all_device_tiles() {
    while (!all_device_tiles_finished()) {
      load_device_tile();
      process_device_tile();
      // evict_device_tile();
    }
  }

  __device__ __forceinline__ void process_device_tile() {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("Processing Tile (%d,...)\n", cur_row_tile_idx);
    // }
    while (!device_tile_finished()) {
      // load_block_tile();
      process_block_tile();
      // evict_block_tile();
    }

    cur_row_tile_idx++;
    cur_col_tile_idx = 0;
  }

  //   __device__ __forceinline__ void lb_warp_per_row() {}

  //   __device__ __forceinline__ void lb_thread_per_row()
  //   {
  //     cg::grid_group grid = cg::this_grid();

  //     if (debug)
  //     {
  //       lb_stats[blockIdx.x] = 0;
  //       __syncthreads();
  //     }

  //     int cur_row_in_gpu_tile = blockIdx.x * rows_per_block_tile +
  //     threadIdx.x; int cur_row_in_matrix =
  //         tile2global(cur_row_in_gpu_tile, cur_row_tile_idx,
  //         rows_per_gpu_tile);

  //     int cur_row_in_block_tile = threadIdx.x;

  //     int stride = blockDim.x;

  //     // End of the col tile boundary
  //     index_t tile_boundary =
  //         min(num_cols, (cur_col_tile_idx + 1) * tile_col_size);

  //     // Iterate over all rows in the current tile
  //     for (; cur_row_in_matrix < num_rows &&
  //            cur_row_in_block_tile < rows_per_block_tile;
  //          cur_row_in_matrix += stride, cur_row_in_block_tile += stride,
  //          cur_row_in_gpu_tile += stride)
  //     {
  //       // Process a row

  //       printf(
  //           "Block %d, Thread %d, Global Row %d, Local Row %d, Tile Nonzeros
  //           "
  //           "%d\n",
  //           blockIdx.x, threadIdx.x, cur_row_in_matrix,
  //           cur_row_in_block_tile, row_elems_in_tile(cur_row_in_matrix,
  //           cur_row_in_block_tile,
  //                             tile_boundary));

  //       value_t sum = 0.0;
  //       index_t offset = local_row_offsets_start[cur_row_in_block_tile];

  //       index_t max_offset;

  //       if (store_end_offsets_in_shmem)
  //       {
  //         max_offset = local_row_offsets_end[cur_row_in_block_tile];
  //       }
  //       else
  //       {
  //         max_offset = row_offsets[cur_row_in_block_tile + 1];
  //       }

  //       while (true)
  //       {
  //         if (offset >= max_offset)
  //           break;

  //         index_t col = col_idx[offset];

  //         if (col >= tile_boundary)
  //         {
  //           // printf("Col %d greater than boundary %d\n", col,
  //           tile_boundary); break;
  //         }
  //         else
  //         {
  //           // printf("Processing col %d\n", col);
  //         }
  //         // atomicAdd(&block_nonzeros, 1);
  //         sum += nonzeros[offset] * input[col];

  //         if (debug)
  //           atomicAdd(&lb_stats[blockIdx.x], 1);

  //         offset++;
  //       }

  //       // Finished with the row

  //       // Save the offset for the next iteration
  //       local_row_offsets_start[cur_row_in_block_tile] = offset;
  //       if (sum != 0)
  //       {
  //         output[cur_row_in_matrix] += sum;
  //       }
  //     }

  //     // Must sync at the end of the tile to preserve cache reuse

  //     grid.sync();

  //     if (threadIdx.x == 0 && debug)
  //     {
  //       printf("Tile (%d,%d) block %d has %d nonzeros\n", cur_row_tile_idx,
  //              cur_col_tile_idx, blockIdx.x, lb_stats[blockIdx.x]);
  //     }

  //     cur_col_tile_idx++;
  //   }

  //   __device__ __forceinline__ void process_block_tile()
  //   {
  //     // if (blockIdx.x == 0 && threadIdx.x == 0) {
  //     //   printf("Processing Tile (%d,%d)\n", cur_row_tile_idx,
  //     //   cur_col_tile_idx);
  //     // }

  //     lb_thread_per_row();
  //     // lb_warp_per_row();
  //   }

  // private:
  //   // SPMV operator properties
  //   const index_t num_rows;
  //   const index_t num_cols;
  //   const index_t num_nonzeros;
  //   const index_t *row_offsets;
  //   const index_t *col_idx;
  //   const value_t *nonzeros;
  //   const value_t *input;
  //   value_t *output;

  //   // Tiling metadata
  //   index_t rows_per_block_tile;
  //   index_t rows_per_gpu_tile;
  //   index_t tile_col_size;

  //   index_t cur_row_tile_idx;
  //   index_t cur_col_tile_idx;

  //   // shmem
  //   index_t *local_row_offsets_start;
  //   index_t *local_row_offsets_end;

  //   index_t *lb_stats;

  //   const bool store_end_offsets_in_shmem;

  //   const bool debug;

 private:
  // Store the inputs and outputs
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  size_t shmem_size;
};