#include <gunrock/algorithms/experimental/async/bfs.hxx>
#include "spmv_cpu.hxx"
#include "spmv_cusparse.cuh"

using namespace gunrock;
using namespace experimental;
using namespace memory;

enum SPMV_t { MGPU, CUB, CUSPARSE, TILED };
enum LB_t { THREAD_PER_ROW, WARP_PER_ROW, BLOCK_PER_ROW, MERGE_PATH };

// double run_test() {
//   // Reset the output vector
//   thrust::fill(dout.begin(), dout.end(), 0);

//   double elapsed_time = 0;

//   //   Run on appropriate GPU implementation
//   if (spmv_impl == MGPU) {
//     elapsed_time = spmv_mgpu(sparse_matrix, din, dout);
//   } else if (spmv_impl == CUB) {
//     // elapsed_time = spmv_cub(sparse_matrix, din, dout);
//   } else if (spmv_impl == CUSPARSE) {
//     elapsed_time = spmv_cusparse(sparse_matrix, din, dout);
//   } else if (spmv_impl == TILED) {
//     elapsed_time = spmv_tiled(sparse_matrix, din, dout, debug);
//   } else {
//     std::cout << "Unsupported SPMV implementation" << std::endl;
//   }

//   printf("GPU finished in %lf ms\n", elapsed_time);

//   //   Copy results to CPU
//   if (check) {
//     thrust::host_vector<float> h_output = dout;

//     // Run on CPU
//     thrust::host_vector<float> cpu_ref(sparse_matrix.num_rows);
//     cpu_spmv(sparse_matrix, hin, cpu_ref);

//     for (index_t row = 0; row < sparse_matrix.num_rows; row++) {
//       cpu_ref[row] = 0.0;
//       // Loop over all the non-zeroes within A's row
//       for (auto k = sparse_matrix.row_offsets[row];
//            k < sparse_matrix.row_offsets[row + 1]; ++k)
//         cpu_ref[row] +=
//             sparse_matrix.nonzero_vals[k] * hin[sparse_matrix.col_idx[k]];
//     }

//     util::display(hin, "cpu_in");
//     util::display(din, "gpu_in");
//     util::display(cpu_ref, "cpu_out");
//     util::display(dout, "gpu_out");

//     // Validate
//     bool passed = validate(h_output, cpu_ref);
//     if (passed) {
//       std::cout << "Validation Successful" << std::endl;
//       return elapsed_time;
//     } else {
//       std::cout << "Validation Failed" << std::endl;
//       return -1;
//     }
//   }
//   return elapsed_time;
// }

void test_spmv(int num_arguments, char** argument_array) {
  if (num_arguments != 2) {
    std::cerr << "usage: ./bin/<program-name> filename.mtx" << std::endl;
    exit(1);
  }

  // --
  // Define types

  using row_t = int;
  using edge_t = int;
  using nonzero_t = float;

  using csr_t = format::csr_t<memory_space_t::device, row_t, edge_t, nonzero_t>;

  // --
  // IO

  csr_t csr;
  std::string filename = argument_array[1];

  if (util::is_market(filename)) {
    io::matrix_market_t<row_t, edge_t, nonzero_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  // --
  // Build graph + metadata

  // auto sparse_matrix = graph::build::from_csr<memory_space_t::device,
  //                                             graph::view_t::csr>(
  //     csr.number_of_rows,               // rows
  //     csr.number_of_columns,            // columns
  //     csr.number_of_nonzeros,           // nonzeros
  //     csr.row_offsets.data().get(),     // row_offsets
  //     csr.column_indices.data().get(),  // column_indices
  //     csr.nonzero_values.data().get()   // values
  // );

  thrust::host_vector<nonzero_t> x_host(csr.number_of_columns);

  srand(0);
  for (size_t idx = 0; idx < x_host.size(); idx++)
    x_host[idx] = rand() % 64;

  thrust::device_vector<nonzero_t> x_device = x_host;
  thrust::device_vector<nonzero_t> y_device(csr.number_of_rows);

  thrust::host_vector<nonzero_t> y_ref_host(csr.number_of_rows);

  // --
  // Run the algorithm

  bool verify = true;

  double elapsed_cusparse = spmv_cusparse(csr, x_device, y_device);

  double elapsed_tiled = 0;
  // double elapsed_tiled = run_tiled(sparse_matrix, x_device, y_device);

  int num_errors_cusparse = 0;
  if (verify) {
    // Get the output from the device
    thrust::host_vector<nonzero_t> y_host = y_device;

    // Compute the reference solution
    cpu_spmv(csr, x_host, y_ref_host);

    // Print the cuSparse computed vector
    display(y_host, "y_cusparse");

    // Print the reference computed vector
    display(y_ref_host, "y_ref_host");

    num_errors_cusparse = check_spmv(y_ref_host, y_host);
  }

  printf("%s,%d,%d,%d,%f,%f\n", filename.c_str(), csr.number_of_rows,
         csr.number_of_columns, csr.number_of_nonzeros, elapsed_cusparse,
         elapsed_tiled);

  // Print the number of errors
  printf("Errors: %d\n", num_errors_cusparse);
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
