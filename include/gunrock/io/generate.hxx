/**
 * @file generate.hxx
 * @author Jonathan Wapman (jdwapman@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2022-07-09
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gunrock/formats/formats.hxx>
#include <gunrock/memory.hxx>
#include <gunrock/error.hxx>
#include <random>

namespace gunrock {
namespace io {
template <typename row_t, typename col_t>
auto generate_dense(row_t num_rows, col_t num_cols) {
  format::coo_t<memory_space_t::host, row_t, col_t, float> coo(
      num_rows, num_cols, num_rows * num_cols);

  // Generate random nonzero values in the range [lower_bound, upper_bound)
  for (row_t i = 0; i < num_rows; i++) {
    for (col_t j = 0; j < num_cols; j++) {
      coo.row_indices[i * num_cols + j] = i;
      coo.column_indices[i * num_cols + j] = j;
      // Random float in range [0,1)
      coo.nonzero_values[i * num_cols + j] =
          (float)std::rand() / (float)RAND_MAX;
    }
  }
  return coo;
}
}  // namespace io
}  // namespace gunrock
