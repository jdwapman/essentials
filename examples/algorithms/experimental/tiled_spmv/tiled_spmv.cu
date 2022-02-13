#include <gunrock/algorithms/experimental/async/bfs.hxx>
#include "spmv_cpu.hxx"
#include "spmv_cusparse.cuh"
#include "spmv_cub.cuh"
#include "spmv_moderngpu.cuh"

using namespace gunrock;
using namespace experimental;
using namespace memory;

enum SPMV_t { MGPU, CUB, CUSPARSE, TILED };
enum LB_t { THREAD_PER_ROW, WARP_PER_ROW, BLOCK_PER_ROW, MERGE_PATH };

template <typename csr_t, typename vector_t>
double test_spmv(SPMV_t spmv_impl,
                 csr_t& sparse_matrix,
                 vector_t& d_input,
                 vector_t& d_output,
                 bool cpu_verify,
                 bool debug) {
  // Reset the output vector
  thrust::fill(d_output.begin(), d_output.end(), 0);

  double elapsed_time = 0;

  //   Run on appropriate GPU implementation
  if (spmv_impl == MGPU) {
    elapsed_time = spmv_mgpu(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == CUB) {
    elapsed_time = spmv_cub(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == CUSPARSE) {
    elapsed_time = spmv_cusparse(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == TILED) {
    // elapsed_time = spmv_tiled(sparse_matrix, d_input, d_output);
  } else {
    std::cout << "Unsupported SPMV implementation" << std::endl;
  }

  if (debug)
    printf("GPU finished in %lf ms\n", elapsed_time);

  //   Copy results to CPU
  if (cpu_verify) {
    thrust::host_vector<float> h_output = d_output;
    thrust::host_vector<float> h_input = d_input;

    // Run on CPU
    thrust::host_vector<float> cpu_ref(sparse_matrix.number_of_rows);
    cpu_spmv(sparse_matrix, h_input, cpu_ref);

    if (debug) {
      display(d_input, "d_input");
      display(d_output, "d_output");
      display(cpu_ref, "cpu_ref");
    }

    // Validate
    int num_errors = check_spmv(cpu_ref, h_output);

    // Print the number of errors
    if (debug)
      printf("Errors: %d\n", num_errors);

    if (!num_errors) {
      if (debug)
        std::cout << "Validation Successful" << std::endl;
      return elapsed_time;
    } else {
      if (debug)
        std::cout << "Validation Failed" << std::endl;
      return -1;
    }
  }

  return elapsed_time;
}

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

  thrust::host_vector<nonzero_t> x_host(csr.number_of_columns);

  srand(0);
  for (size_t idx = 0; idx < x_host.size(); idx++)
    x_host[idx] = rand() % 64;

  thrust::device_vector<nonzero_t> x_device = x_host;
  thrust::device_vector<nonzero_t> y_device(csr.number_of_rows);

  // --
  // Run the algorithm

  bool cpu_verify = true;
  bool debug = true;

  double elapsed_cusparse =
      test_spmv(CUSPARSE, csr, x_device, y_device, cpu_verify, debug);

  double elapsed_cub =
      test_spmv(CUB, csr, x_device, y_device, cpu_verify, debug);

  double elapsed_mgpu =
      test_spmv(MGPU, csr, x_device, y_device, cpu_verify, debug);

  printf("%s,%d,%d,%d,%f,%f,%f\n", filename.c_str(), csr.number_of_rows,
         csr.number_of_columns, csr.number_of_nonzeros, elapsed_cusparse,
         elapsed_cub, elapsed_mgpu);
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
