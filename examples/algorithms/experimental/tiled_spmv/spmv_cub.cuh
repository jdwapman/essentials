#pragma once

#include <cub/cub.cuh>  // or equivalently <cub/device/device_spmv.cuh>
// Declare, allocate, and initialize device-accessible pointers for input matrix
// A, input vector x, and output vector y

#include "spmv_utils.cuh"
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.

template <typename csr_t, typename vector_t>
double spmv_cub(csr_t& A, vector_t& input, vector_t& output) {
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Check that I'm actually setting the stream correctly

  CHECK_CUDA(cub::DeviceSpmv::CsrMV(
      d_temp_storage, temp_storage_bytes, A.nonzero_values.data().get(),
      A.row_offsets.data().get(), A.column_indices.data().get(),
      input.data().get(), output.data().get(), A.number_of_rows,
      A.number_of_columns, A.number_of_nonzeros));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes))
  // Run SpMV

  printf("CUB Allocating %d bytes\n", (int)temp_storage_bytes);

  int* d_row_offsets = A.row_offsets.data().get();
  int* d_col_idx = A.column_indices.data().get();
  float* d_values = A.nonzero_values.data().get();

  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUDA(cub::DeviceSpmv::CsrMV(
      d_temp_storage, temp_storage_bytes, d_values, d_row_offsets, d_col_idx,
      input.data().get(), output.data().get(), A.number_of_rows,
      A.number_of_columns, A.number_of_nonzeros, 0, true));
  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  CHECK_CUDA(cudaFree(d_temp_storage))

  return timer.milliseconds();
}