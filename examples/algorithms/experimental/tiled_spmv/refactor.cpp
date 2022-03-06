template <typename graph_t, typename vector_t, typename shmem_t>
class TileIterator {
 public:
  __device__ TileIterator(const graph_t _graph,
                          const vector_t* _input,
                          vector_t* _output,
                          int* _queue_counter,
                          const size_t _tile_row_size,
                          const size_t _tile_col_size,
                          shmem_t* _shmem,
                          const size_t _shmem_size)
      : graph(_graph),
        input(_input),
        output(_output),
        queue_counter(_queue_counter),
        tile_row_size(_tile_row_size),
        tile_col_size(_tile_col_size),
        shmem(_shmem),
        shmem_size(_shmem_size) {
    cur_tile_col_idx = 0;
    cur_tile_row_idx = blockIdx.x;

    num_row_tiles = graph.get_number_of_rows() / tile_row_size;

    // Handle the remainder
    if (graph.get_number_of_rows() % tile_row_size != 0) {
      num_row_tiles++;
    }

    num_col_tiles = graph.get_number_of_columns() / tile_col_size;

    if (graph.get_number_of_columns() % tile_col_size != 0) {
      num_col_tiles++;
    }

    // Setup a piece of memory for the block to to communicate which row it's
    // working on
    extern __shared__ shmem_t shared_temp[];

    block_temp = shared_temp;  // Size of 1

    // Update the shared memory
    shmem = (shmem_t*)&shared_temp[1];
    shmem_size -= sizeof(shmem_t);

    // TODO recompute tile size based on remaining shmem

    print_device("Remaining shmem size: %d\n", shmem_size);

    // Reset the row tile queue counter
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      queue_counter[0] = blockDim.x;
    }

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();
  }

  __device__ __forceinline__ bool all_columns_finished() {
    if (cur_tile_col_idx >= num_col_tiles) {
      return true;
    }

    return false;
  }

  __device__ __forceinline__ void process_block_row_tile() {
    print_block(" - Processing block row tile %d\n", (int)cur_tile_row_idx);
  }

  __device__ __forceinline__ void get_next_block_row_tile() {
    // Atomically increment the current tile row index
    // Note that this needs to be to a GLOBAL variable so that all blocks
    // can see it.

    // TODO will need to change this to use the externally-added shmem
    __shared__ int shared_cur_tile_row_idx;

    if (threadIdx.x == 0) {
      cur_tile_row_idx = atomicAdd(&queue_counter[0], 1);
      shared_cur_tile_row_idx = cur_tile_row_idx;
    }

    __syncthreads();

    cur_tile_row_idx = shared_cur_tile_row_idx;
  }

  __device__ __forceinline__ void process_gpu_col_tile() {
    // Iterate over the row tiles as long as it's in bounds.

    print_device("Processing GPU col tile %d\n", (int)cur_tile_col_idx);

    // The GPU has its tile row index and col index. Need to first load in
    // metadata, then do the load balancing, then do the computation, then
    // unload any metadata we need to save to global mem for the next time we
    // come back to this row. Finally, need to increment an atomic and go on to
    // the next row tile if there are more to process.

    // Then do a grid-wide synchronization.

    // All blocks iterate over the row tiles
    while (cur_tile_row_idx < num_row_tiles) {
      process_block_row_tile();
      get_next_block_row_tile();
    }

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();
  }

  __device__ __forceinline__ void get_next_gpu_col_tile() {
    // Reset the tile metadata
    cur_tile_row_idx = blockIdx.x;
    cur_tile_col_idx += 1;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      queue_counter[0] = gridDim.x;
    }

    print_device("Starting column tile %d\n", (int)cur_tile_col_idx);

    // Sync
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // TODO do something to reset the caching here?
  }

  __device__ __forceinline__ void process_all_tiles() {
    while (!all_columns_finished()) {
      process_gpu_col_tile();
      get_next_gpu_col_tile();
    }
  }

 private:
  // Store the inputs and outputs
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  size_t shmem_size;
  int* queue_counter;

  // Tiling metadata
  const size_t tile_row_size;
  const size_t tile_col_size;
  shmem_t* block_temp;  // In shared memory

  size_t cur_tile_row_idx;
  size_t cur_tile_col_idx;
  size_t num_row_tiles;
  size_t num_col_tiles;
};