#pragma once

#include <moderngpu/kernel_segreduce.hxx>
#include "spmv_utils.cuh"

template <typename csr_t, typename vector_t>
double spmv_mgpu(csr_t& A, vector_t& input, vector_t& output)
{

    // ... GPU SPMV
    // GPU device context, print
    mgpu::standard_context_t context(false);

    auto values = A.nonzero_values.data().get();
    auto indices = A.column_indices.data().get();
    auto offsets = A.row_offsets.data().get();

    int offsets_size = A.number_of_rows;
    int nnz = A.number_of_nonzeros;

    gunrock::util::timer_t timer;
    timer.begin();
    mgpu::spmv(values, indices, input.data().get(), nnz, offsets, offsets_size, output.data().get(), context);

    // Synchronize the device
    context.synchronize();

    timer.end();

    return timer.milliseconds();
}