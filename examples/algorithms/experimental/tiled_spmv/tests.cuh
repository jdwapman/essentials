#pragma once

#include <gtest/gtest.h>
#include "./tile_iterator.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void large_to_small_kernel(int* row, int* col) {
  auto r0 = 10000;
  auto r1 = 1000;
  auto r2 = 100;
  auto r3 = 10;

  auto c0 = 10000;
  auto c1 = 1000;
  auto c2 = 100;
  auto c3 = 10;

  auto layout_0 = make_layout(r0, c0);
  auto layout_1 = make_layout(r1, c1, layout_0);
  auto layout_2 = make_layout(r2, c2, layout_1);
  auto layout_3 = make_layout(r3, c3, layout_2);

  auto layout = layout_3;

  // Set up the tile info
  auto t0 = make_tile_index(0, 0);
  auto t1 = make_tile_index(4, 5, t0);
  auto t2 = make_tile_index(5, 3, t1);
  auto t3 = make_tile_index(9, 7, t2);

  Point<int, int> pt(405, 871);

  auto new_point_0 = layout.remap_point(pt, t0, 0);
  auto new_point_1 = layout.remap_point(pt, t0, 1);
  auto new_point_2 = layout.remap_point(pt, t0, 2);
  auto new_point_3 = layout.remap_point(pt, t0, 3);

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
  auto r0 = 10000;
  auto r1 = 1000;
  auto r2 = 100;
  auto r3 = 10;

  auto c0 = 10000;
  auto c1 = 1000;
  auto c2 = 100;
  auto c3 = 10;

  auto layout_0 = make_layout(r0, c0);
  auto layout_1 = make_layout(r1, c1, layout_0);
  auto layout_2 = make_layout(r2, c2, layout_1);
  auto layout_3 = make_layout(r3, c3, layout_2);

  auto layout = layout_3;

  // Set up the tile info
  auto t0 = make_tile_index(0, 0);
  auto t1 = make_tile_index(1, 1, t0);
  auto t2 = make_tile_index(2, 2, t1);
  auto t3 = make_tile_index(3, 3, t2);

  Point<int, int> pt(5, 5);

  auto new_point_0 = layout.remap_point(pt, t3, 0);
  auto new_point_1 = layout.remap_point(pt, t3, 1);
  auto new_point_2 = layout.remap_point(pt, t3, 2);
  auto new_point_3 = layout.remap_point(pt, t3, 3);

  row[0] = new_point_0.row;
  col[0] = new_point_0.col;
  row[1] = new_point_1.row;
  col[1] = new_point_1.col;
  row[2] = new_point_2.row;
  col[2] = new_point_2.col;
  row[3] = new_point_3.row;
  col[3] = new_point_3.col;
}

__global__ void tile_test(int* success) {
  auto r0 = 1004;
  auto r1 = 200;
  auto r2 = 20;
  auto r3 = 10;

  auto c0 = 503;
  auto c1 = 250;
  auto c2 = 25;
  auto c3 = 5;

  auto layout_0 = make_layout(r0, c0);
  auto layout_1 = make_layout(r1, c1, layout_0);
  auto layout_2 = make_layout(r2, c2, layout_1);
  auto layout_3 = make_layout(r3, c3, layout_2);

  auto layout = layout_3;

  // Check that we can get the tile dimensions correctly
  if (layout.rows_in_tile(0) == r0 && layout.cols_in_tile(0) == c0 &&
      layout.rows_in_tile(1) == r1 && layout.cols_in_tile(1) == c1 &&
      layout.rows_in_tile(2) == r2 && layout.cols_in_tile(2) == c2 &&
      layout.rows_in_tile(3) == r3 && layout.cols_in_tile(3) == c3) {
    success[0] = 1;
  } else {
    success[0] = 0;
    printf("FAILED\n");
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n", r0, c0, r1, c1, r2, c2, r3,
           c3);
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n", layout.rows_in_tile(0),
           layout.cols_in_tile(0), layout.rows_in_tile(1),
           layout.cols_in_tile(1), layout.rows_in_tile(2),
           layout.cols_in_tile(2), layout.rows_in_tile(3),
           layout.cols_in_tile(3));
    return;
  }

  // Check that we get the number of child tiles correctly
  if (layout.num_child_row_tiles(0) == ceil((float)r0 / (float)r1) &&
      layout.num_child_col_tiles(0) == ceil((float)c0 / (float)c1) &&
      layout.num_child_row_tiles(1) == ceil((float)r1 / (float)r2) &&
      layout.num_child_col_tiles(1) == ceil((float)c1 / (float)c2) &&
      layout.num_child_row_tiles(2) == ceil((float)r2 / (float)r3) &&
      layout.num_child_col_tiles(2) == ceil((float)c2 / (float)c3) &&
      layout.num_child_row_tiles(3) == 1 &&
      layout.num_child_col_tiles(3) == 1) {
    success[1] = 1;
  } else {
    printf("FAILED\n");
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n", r0, c0, r1, c1, r2, c2, r3,
           c3);
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n",
           layout.num_child_row_tiles(0), layout.num_child_col_tiles(0),
           layout.num_child_row_tiles(1), layout.num_child_col_tiles(1),
           layout.num_child_row_tiles(2), layout.num_child_col_tiles(2),
           layout.num_child_row_tiles(3), layout.num_child_col_tiles(3));
    success[1] = 0;
    return;
  }

  // Check that we can get the number of tiles at a level correctly
  if (layout.num_row_tiles_at_level(0) == 1 &&
      layout.num_col_tiles_at_level(0) == 1 &&
      layout.num_row_tiles_at_level(1) == ceil((float)r0 / (float)r1) &&
      layout.num_col_tiles_at_level(1) == ceil((float)c0 / (float)c1) &&
      layout.num_row_tiles_at_level(2) == ceil((float)r1 / (float)r2) &&
      layout.num_col_tiles_at_level(2) == ceil((float)c1 / (float)c2) &&
      layout.num_row_tiles_at_level(3) == ceil((float)r2 / (float)r3) &&
      layout.num_col_tiles_at_level(3) == ceil((float)c2 / (float)c3)) {
    success[2] = 1;
  } else {
    printf("FAILED\n");
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n", r0, c0, r1, c1, r2, c2, r3,
           c3);
    printf("(%d,%d), (%d,%d), (%d,%d), (%d,%d)\n",
           layout.num_row_tiles_at_level(0), layout.num_col_tiles_at_level(0),
           layout.num_row_tiles_at_level(1), layout.num_col_tiles_at_level(1),
           layout.num_row_tiles_at_level(2), layout.num_col_tiles_at_level(2),
           layout.num_row_tiles_at_level(3), layout.num_col_tiles_at_level(3));
    success[2] = 0;
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

  EXPECT_EQ(row_d[0], 405);
  EXPECT_EQ(col_d[0], 871);

  EXPECT_EQ(row_d[1], 405);
  EXPECT_EQ(col_d[1], 871);

  EXPECT_EQ(row_d[2], 5);
  EXPECT_EQ(col_d[2], 71);

  EXPECT_EQ(row_d[3], 5);
  EXPECT_EQ(col_d[3], 1);
}

TEST(IndexConverterTest, SmallToLargeTest) {
  // Set up cuda vectors
  thrust::device_vector<int> row_d(4);
  thrust::device_vector<int> col_d(4);

  // Run kernel
  small_to_large_kernel<<<1, 1>>>(row_d.data().get(), col_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  EXPECT_EQ(row_d[3], 5);
  EXPECT_EQ(col_d[3], 5);

  EXPECT_EQ(row_d[2], 35);
  EXPECT_EQ(col_d[2], 35);

  EXPECT_EQ(row_d[1], 235);
  EXPECT_EQ(col_d[1], 235);

  EXPECT_EQ(row_d[0], 1235);
  EXPECT_EQ(col_d[0], 1235);
}

TEST(TileSetupTest, SetupTest) {
  // Set up cuda vectors
  thrust::device_vector<int> success_d(3);

  // Run kernel
  tile_test<<<1, 1>>>(success_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  // Google test assert success == 1
  EXPECT_EQ(success_d[0], 1);
  EXPECT_EQ(success_d[1], 1);
  EXPECT_EQ(success_d[2], 1);
}