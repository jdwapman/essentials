#pragma once

#include <gtest/gtest.h>
#include "./tile_iterator.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void large_to_small_kernel(int* row, int* col) {
  TileIndexer<4> tile_indexer(ROWMAJOR);

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

  Point pt(405, 871);

  // Set up the tile info
  auto t0 = make_tile_index(0,0);
  auto t1 = make_tile_index(4, 5, t0);
  auto t2 = make_tile_index(5,3, t1);
  auto t3 = make_tile_index(9, 7, t2);

  Point new_point_0 = tile_indexer.convert_index(pt, t3, 0);
  Point new_point_1 = tile_indexer.convert_index(pt, t3, 1);
  Point new_point_2 = tile_indexer.convert_index(pt, t3, 2);
  Point new_point_3 = tile_indexer.convert_index(pt, t3, 3);

  row[0] = new_point_0.row;
  col[0] = new_point_0.col;
  row[1] = new_point_1.row;
  col[1] = new_point_1.col;
  row[2] = new_point_2.row;
  col[2] = new_point_2.col;
  row[3] = new_point_3.row;
  col[3] = new_point_3.col;
}

// __global__ void small_to_large_kernel(int* row, int* col) {
//   TileIndexer<4> tile_indexer(ROWMAJOR);

//   tile_indexer.add_tile_info(0, 10000, 10000);
//   tile_indexer.add_tile_info(1, 1000, 1000);
//   tile_indexer.add_tile_info(2, 100, 100);
//   tile_indexer.add_tile_info(3, 10, 10);

//   TileIdx t0;
//   TileIdx t1(3, 4, &t0);
//   TileIdx t2(2, 3, &t1);
//   TileIdx t3(1, 2, &t2);

//   Point pt(5, 6);

//   Point pt_0 = tile_indexer.convert_index(pt, &t0, 0);
//   Point pt_1 = tile_indexer.convert_index(pt, &t1, 0);
//   Point pt_2 = tile_indexer.convert_index(pt, &t2, 0);
//   Point pt_3 = tile_indexer.convert_index(pt, &t3, 0);

//   row[0] = pt_0.row;
//   col[0] = pt_0.col;
//   row[1] = pt_1.row;
//   col[1] = pt_1.col;
//   row[2] = pt_2.row;
//   col[2] = pt_2.col;
//   row[3] = pt_3.row;
//   col[3] = pt_3.col;
// }

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

  ASSERT_EQ(row_d[0], 405);
  ASSERT_EQ(col_d[0], 871);

  ASSERT_EQ(row_d[1], 5);
  ASSERT_EQ(col_d[1], 71);

  ASSERT_EQ(row_d[2], 5);
  ASSERT_EQ(col_d[2], 1);

  ASSERT_EQ(row_d[3], 0);
  ASSERT_EQ(col_d[3], 1);
}

// TEST(IndexConverterTest, SmallToLargeTest) {
//   // Set up cuda vectors
//   thrust::device_vector<int> row_d(4);
//   thrust::device_vector<int> col_d(4);

//   // Run kernel
//   small_to_large_kernel<<<1, 1>>>(row_d.data().get(), col_d.data().get());

//   // Synchronize
//   CHECK_CUDA(cudaDeviceSynchronize());

//   ASSERT_EQ(row_d[0], 5);
//   ASSERT_EQ(col_d[0], 6);

//   ASSERT_EQ(row_d[1], 3005);
//   ASSERT_EQ(col_d[1], 4006);

//   ASSERT_EQ(row_d[2], 3205);
//   ASSERT_EQ(col_d[2], 4306);

//   ASSERT_EQ(row_d[3], 3215);
//   ASSERT_EQ(col_d[3], 4326);
// }

TEST(TileSetupTest, SetupTest) {
  // Set up cuda vectors
  thrust::device_vector<int> success_d(3);

  // Run kernel
  tile_test<<<1, 1>>>(success_d.data().get());

  // Synchronize
  CHECK_CUDA(cudaDeviceSynchronize());

  // Google test assert success == 1
  ASSERT_EQ(success_d[0], 1);
  ASSERT_EQ(success_d[1], 1);
  ASSERT_EQ(success_d[2], 1);
}