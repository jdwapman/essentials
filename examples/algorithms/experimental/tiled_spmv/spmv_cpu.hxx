#pragma once

#include <chrono>
#include <vector>
#include <queue>
#include <iomanip>

template <typename csr_t, typename vector_t>
void cpu_spmv(csr_t& A, vector_t& x, vector_t& y) {
  thrust::host_vector<int> row_offsets(A.row_offsets);
  thrust::host_vector<int> column_indices(A.column_indices);
  thrust::host_vector<float> nonzero_values(A.nonzero_values);

  // Loop over all the rows of A
  for (int row = 0; row < A.number_of_rows; row++) {
    y[row] = 0.0;
    // Loop over all the non-zeroes within A's row
    for (auto k = row_offsets[row]; k < row_offsets[row + 1]; ++k)
      y[row] += nonzero_values[k] * x[column_indices[k]];
  }
}

template <typename T>
static bool equal(T f1, T f2) {
  // return(std::fabs(f1 - f2) <= 1e-4);
//   T eps = std::numeric_limits<T>::epsilon() * 100;
//   // // T eps = 0.2;
  T eps = 1e-3;
  return (std::fabs(f1 - f2) <= eps * std::fmax(std::fabs(f1), std::fabs(f2)));
}

template <typename vector_t>
int check_spmv(vector_t& a, vector_t& b) {
  int num_errors = 0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!equal(a[i], b[i])) {
      std::cout << "i = " << i << ": " << std::setprecision(20) << a[i]
                << " != " << b[i] << std::endl;
      std::cout << "Error = " << std::fabs(a[i] - b[i]) << std::endl;
      double error_percent =
          std::fabs(a[i] - b[i]) / std::fmax(std::fabs(a[i]), std::fabs(b[i]));
      std::cout << "Error % = " << error_percent << std::endl;
      num_errors++;
    }
  }

  return num_errors;
}