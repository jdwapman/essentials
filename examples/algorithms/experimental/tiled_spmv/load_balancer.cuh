#pragma once

template <typename graph_t, typename vector_t, typename shmem_t>
class RowPerThreadLB {
  __device__ RowPerThreadLB(const graph_t _graph,
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
        shmem_size(_shmem_size) {}

  // Calculate the amount of shmem needed for any helpers
  __device__ size_t get_shmem_size() { return 0; }

  __device__ void process_tile() {
    size_t rows_in_tile = shmem_size / sizeof(shmem_t);

    size_t tile_boundary =
        min(graph.num_cols(), (cur_col_tile_idx + 1) * tile_col_size);

    for (int thread_row_in_tile = threadIdx.x;
         thread_row_in_tile < rows_in_tile; thread_row_in_tile += blockDim.x) {
    }
  }

 private:
  // Store the inputs and outputs
  const graph_t graph;
  const vector_t* input;
  vector_t* output;
  shmem_t* shmem;
  size_t shmem_size;

  // Tiling metadata
  const size_t tile_row_size;
  const size_t tile_col_size;

  size_t cur_tile_row_idx;
  size_t cur_tile_col_idx;
  size_t num_row_tiles;
  size_t num_col_tiles;
};