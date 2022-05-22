#pragma once

#include <cub/cub.cuh>  // or equivalently <cub/device/device_spmv.cuh>
// Declare, allocate, and initialize device-accessible pointers for input matrix
// A, input vector x, and output vector y

#include "spmv_utils.cuh"
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cuda/annotated_ptr>

template <typename csr_t, typename vector_t, typename args_t>
double spmv_cub(cudaStream_t stream,
                csr_t& A,
                vector_t& input,
                vector_t& output,
                args_t pargs) {
  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  // Set up annotated pointers
  int* d_row_offsets = A.row_offsets.data().get();
  int* d_col_idx = A.column_indices.data().get();
  float* d_values = A.nonzero_values.data().get();
  float* d_input = input.data().get();
  float* d_output = output.data().get();

  // Check if we're pinning the memory
  if (pargs.count("pin")) {
    printf("Setting up CUB memory pinning\n");
    // cuda::apply_access_property(d_input, (size_t)input.size() * sizeof(float),
    //                             cuda::access_property::persisting{});

    d_row_offsets = cuda::associate_access_property(d_row_offsets, cuda::access_property::streaming{});
    d_col_idx = cuda::associate_access_property(d_col_idx, cuda::access_property::streaming{});
    d_values = cuda::associate_access_property(d_values, cuda::access_property::streaming{});
    d_input = cuda::associate_access_property(d_input, cuda::access_property::persisting{});
    d_output = cuda::associate_access_property(d_output, cuda::access_property::streaming{});
  }

  // Annotated pointer
  //   cuda::annotated_ptr<int, cuda::access_property::streaming>
  //       pinned_row_offsets(d_row_offsets);

  // Apply
  cuda::apply_access_property(d_row_offsets,
                              (size_t)A.row_offsets.size() * sizeof(int),
                              cuda::access_property::persisting{});

  // Associate
  //   int* pinned_row_offsets = cuda::associate_access_property(
  //       d_row_offsets, cuda::access_property::streaming{});

  CHECK_CUDA(cub::DeviceSpmv::CsrMV(
      d_temp_storage, temp_storage_bytes, d_values, d_row_offsets, d_col_idx,
      d_input, d_output, A.number_of_rows, A.number_of_columns,
      A.number_of_nonzeros, stream));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Allocate temporary storage
  CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes))
  // Run SpMV

  printf("CUB Allocating %d bytes\n", (int)temp_storage_bytes);

  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUDA(cub::DeviceSpmv::CsrMV(
      d_temp_storage, temp_storage_bytes, d_values, d_row_offsets, d_col_idx,
      d_input, d_output, A.number_of_rows, A.number_of_columns,
      A.number_of_nonzeros, 0, stream));
  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  CHECK_CUDA(cudaFree(d_temp_storage))

  // Reset the pinned memory
  if (pargs.count("pin")) {
    printf("Resetting CUB memory pinning\n");
    cuda::apply_access_property(d_input, (size_t)input.size() * sizeof(float),
                                cuda::access_property::normal{});
  }

  return timer.milliseconds();
}