#pragma once

#include <cusparse.h>          // cusparseSpMV
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <thrust/device_vector.h>
#include "spmv_utils.cuh"
#include <gunrock/util/timer.hxx>

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
             cusparseGetErrorString(status), status);                          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

// Helper code from CUDALibrarySamples
// https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE

template <typename csr_t, typename vector_t>
double spmv_cusparse(csr_t& A, vector_t& input, vector_t& output) {
  // Host problem definition
  float alpha = 1.0f;
  float beta = 0.0f;  // TODO parametrize these

  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void* dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(
      &matA, A.number_of_rows, A.number_of_columns, A.number_of_nonzeros,
      A.row_offsets.data().get(), A.column_indices.data().get(),
      A.nonzero_values.data().get(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A.number_of_columns,
                                     input.data().get(), CUDA_R_32F))
  // Create dense vector y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A.number_of_rows,
                                     output.data().get(), CUDA_R_32F))
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
      CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  // execute SpMV
  gunrock::util::timer_t timer;
  timer.begin();
  CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                              matA, vecX, &beta, vecY, CUDA_R_32F,
                              CUSPARSE_MV_ALG_DEFAULT, dBuffer))

  CHECK_CUDA(cudaDeviceSynchronize());
  timer.end();

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
  CHECK_CUSPARSE(cusparseDestroy(handle))

  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer));


  return timer.milliseconds();
}