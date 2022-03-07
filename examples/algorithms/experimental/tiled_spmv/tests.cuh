#pragma once

#include <gtest/gtest.h>
#include "./tile_iterator.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void large_to_small_kernel(int* row, int* col) {
  TileIndexer<4> tile_indexer(ROWMAJOR);

  tile_indexer.add_tile_info(0, 1000, 1000);
  tile_indexer.add_tile_info(1, 100, 100);
  tile_indexer.add_tile_info(2, 10, 10);
  tile_indexer.add_tile_info(3, 5, 5);

  Point pt(405, 871);

  // Set up the tile info
  TileIdx top;

  Point new_point_0 = tile_indexer.convert_index(pt, &top, 0);
  Point new_point_1 = tile_indexer.convert_index(pt, &top, 1);
  Point new_point_2 = tile_indexer.convert_index(pt, &top, 2);
  Point new_point_3 = tile_indexer.convert_index(pt, &top, 3);

  row[0] = new_point_0.row;
  col[0] = new_point_0.col;
  row[1] = new_point_1.row;
  col[1] = new_point_1.col;
  row[2] = new_point_2.row;
  col[2] = new_point_2.col;
  row[3] = new_point_3.row;
  col[3] = new_point_3.col;
}

__global__ void small_to_large_kernel(int* row, int* col) {
  TileIndexer<4> tile_indexer(ROWMAJOR);

  tile_indexer.add_tile_info(0, 10000, 10000);
  tile_indexer.add_tile_info(1, 1000, 1000);
  tile_indexer.add_tile_info(2, 100, 100);
  tile_indexer.add_tile_info(3, 10, 10);

  TileIdx t0;
  TileIdx t1(3, 4, &t0);
  TileIdx t2(2, 3, &t1);
  TileIdx t3(1, 2, &t2);

  Point pt(5, 6);

  Point pt_0 = tile_indexer.convert_index(pt, &t0, 0);
  Point pt_1 = tile_indexer.convert_index(pt, &t1, 0);
  Point pt_2 = tile_indexer.convert_index(pt, &t2, 0);
  Point pt_3 = tile_indexer.convert_index(pt, &t3, 0);

  row[0] = pt_0.row;
  col[0] = pt_0.col;
  row[1] = pt_1.row;
  col[1] = pt_1.col;
  row[2] = pt_2.row;
  col[2] = pt_2.col;
  row[3] = pt_3.row;
  col[3] = pt_3.col;
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

TEST(IndexConverterTest, LargeToSmallTest) {
  // Set up cuda vectors
  thrust::device_vector<int> row_d(4);
  thrust::device_vector<int> col_d(4);

  // Run kernel
  large_to_small_kernel<<<1, 1>>>(row_d.data().get(), col_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  ASSERT_EQ(row_d[0], 405);
  ASSERT_EQ(col_d[0], 871);

  ASSERT_EQ(row_d[1], 5);
  ASSERT_EQ(col_d[1], 71);

  ASSERT_EQ(row_d[2], 5);
  ASSERT_EQ(col_d[2], 1);

  ASSERT_EQ(row_d[3], 0);
  ASSERT_EQ(col_d[3], 1);
}

TEST(IndexConverterTest, SmallToLargeTest) {
  // Set up cuda vectors
  thrust::device_vector<int> row_d(4);
  thrust::device_vector<int> col_d(4);

  // Run kernel
  small_to_large_kernel<<<1, 1>>>(row_d.data().get(), col_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  ASSERT_EQ(row_d[0], 5);
  ASSERT_EQ(col_d[0], 6);

  ASSERT_EQ(row_d[1], 3005);
  ASSERT_EQ(col_d[1], 4006);

  ASSERT_EQ(row_d[2], 3205);
  ASSERT_EQ(col_d[2], 4306);

  ASSERT_EQ(row_d[3], 3215);
  ASSERT_EQ(col_d[3], 4326);
}

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