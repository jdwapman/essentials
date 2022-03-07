#pragma once

#include <gtest/gtest.h>
#include "./tile_iterator.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void small_to_large_kernel(int* row, int* col) {
  TileIndexer<4> tile_indexer(ROWMAJOR);

  tile_indexer.add_tile_info(0, 1000, 1000);
  tile_indexer.add_tile_info(1, 100, 100);
  tile_indexer.add_tile_info(2, 10, 10);

  Point pt(4, 5);

  // Set up the tile info
  TileIdx top;
  TileIdx middle(2, 2, &top);
  TileIdx bottom(3, 3, &middle);

  Point new_point = tile_indexer.convert_index(pt, &bottom, 0);

  row[0] = new_point.row;
  col[0] = new_point.col;
}

__global__ void tile_test(int* success) {
  TileIndexer<4> tile_indexer(ROWMAJOR);

  tile_indexer.add_tile_info(0, 1000, 1000);
  tile_indexer.add_tile_info(1, 100, 100);
  tile_indexer.add_tile_info(2, 10, 10);
  tile_indexer.add_tile_info(3, 5, 5);

  // Check that the tile was created correctly
  if (tile_indexer.tile_row_dim[0] == 1000 &&
      tile_indexer.tile_col_dim[0] == 1000 &&
      tile_indexer.tile_row_dim[1] == 100 &&
      tile_indexer.tile_col_dim[1] == 100 &&
      tile_indexer.tile_row_dim[2] == 10 &&
      tile_indexer.tile_col_dim[2] == 10 && tile_indexer.tile_row_dim[3] == 5 &&
      tile_indexer.tile_col_dim[3] == 5) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAIL: Tile dimensions were not set correctly\n");
    return;
  }

  // Get the number of row tiles at the given level of the hierarchy
  if (tile_indexer.num_row_tiles(0) == 1 &&
      tile_indexer.num_row_tiles(1) == 10 &&
      tile_indexer.num_row_tiles(2) == 10 &&
      tile_indexer.num_row_tiles(3) == 2) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAIL: Number of row tiles was not calculated correctly\n");
    return;
  }

  // Get the number of column tiles at the given level of the hierarchy
  if (tile_indexer.num_col_tiles(0) == 1 &&
      tile_indexer.num_col_tiles(1) == 10 &&
      tile_indexer.num_col_tiles(2) == 10 &&
      tile_indexer.num_col_tiles(3) == 2) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAIL: Number of column tiles was not calculated correctly\n");
    return;
  }

  // Get the number of child tiles at the given level of the hierarchy
  if (tile_indexer.num_child_tiles_row(0) == 10 &&
      tile_indexer.num_child_tiles_row(1) == 10 &&
      tile_indexer.num_child_tiles_row(2) == 2 &&
      tile_indexer.num_child_tiles_row(3) == 1) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAIL: Number of child row tiles was not calculated correctly\n");
    return;
  }

  // Get the number of child tiles at the given level of the hierarchy
  if (tile_indexer.num_child_tiles_col(0) == 10 &&
      tile_indexer.num_child_tiles_col(1) == 10 &&
      tile_indexer.num_child_tiles_col(2) == 2 &&
      tile_indexer.num_child_tiles_col(3) == 1) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAIL: Number of child column tiles was not calculated correctly\n");
    return;
  }
}

// TEST(IndexConverterTest, SmallToLargeTest) {
//   // Set up cuda vectors
//   thrust::device_vector<int> row_d(1);
//   thrust::device_vector<int> col_d(1);

//   // Run kernel
//   small_to_large_kernel<<<1, 1>>>(row_d.data().get(), col_d.data().get());

//   // Synchronize
//   CHECK_CUDA(cudaDeviceSynchronize());

//   // Google test assert row == 101, col == 101
//   ASSERT_EQ(row_d[0], 234);
//   ASSERT_EQ(col_d[0], 235);
// }

TEST(TileSetupTest, SetupTest) {
  // Set up cuda vectors
  thrust::device_vector<int> success_d(1);

  // Run kernel
  tile_test<<<1, 1>>>(success_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  // Google test assert success == 1
  ASSERT_EQ(success_d[0], 1);
}