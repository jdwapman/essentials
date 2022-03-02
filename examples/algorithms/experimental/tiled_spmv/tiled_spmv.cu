#include <gunrock/algorithms/algorithms.hxx>
#include <cuda_runtime_api.h>
#include <cxxopts.hpp>

#include "spmv_cpu.hxx"
#include "spmv_cusparse.cuh"
#include "spmv_cub.cuh"
#include "spmv_moderngpu.cuh"
#include "test_tiled.h"
#include "spmv_utils.cuh"

using namespace gunrock;
// using namespace experimental;
using namespace memory;

enum SPMV_t { MGPU, CUB, CUSPARSE, TILED };
enum LB_t { THREAD_PER_ROW, WARP_PER_ROW, BLOCK_PER_ROW, MERGE_PATH };

template <typename vector_t>
void setup_ampere_cache(vector_t* pinned_mem) {
  // --
  // Set up cache configuration
  int device = 0;
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, device))

  cudaStream_t stream;
  cudaStreamCreate(&stream);  // Create CUDA stream

  // Stream level attributes data structure
  cudaStreamAttrValue stream_attribute;

  if (deviceProp.major >= 8) {
    // Using Ampere

    size_t size =
        min(int(deviceProp.l2CacheSize), deviceProp.persistingL2CacheMaxSize);

    // set-aside the full L2 cache for persisting accesses or the max allowed
    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size));

    int num_bytes =
        (int)pinned_mem
            ->size();  // TODO update this for bytes rather than elements
    size_t window_size = min(deviceProp.accessPolicyMaxWindowSize,
                             num_bytes);  // Select minimum of user defined
                                          // num_bytes and max window size.

    // Global Memory data pointer
    stream_attribute.accessPolicyWindow.base_ptr =
        reinterpret_cast<void*>(pinned_mem->data()->get());

    // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.num_bytes =
        pinned_mem->size() / sizeof(pinned_mem[0]);

    // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitRatio = 0.6;

    // Persistence Property
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;

    // Type of access property on cache miss
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // Set the attributes to a CUDA Stream
    CHECK_CUDA(cudaStreamSetAttribute(
        stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
  } else {
    // Using Volta or below
    printf(
        "WARNING: L2 Cache Management available only for compute capabilities "
        ">= 8\n");
  }
}

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
    printf("=== RUNNING MODERNGPU SPMV ===\n");
    elapsed_time = spmv_mgpu(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == CUB) {
    printf("=== RUNNING CUB SPMV ===\n");
    elapsed_time = spmv_cub(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == CUSPARSE) {
    printf("=== RUNNING CUSPARSE SPMV ===\n");
    elapsed_time = spmv_cusparse(sparse_matrix, d_input, d_output);
  } else if (spmv_impl == TILED) {
    printf("=== RUNNING TILED SPMV ===\n");
    elapsed_time = spmv_tiled(sparse_matrix, d_input, d_output);
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
  cxxopts::Options options(argument_array[0],
                           "Gunrock commandline parser test");

  options.add_options()  // Allows to add options.
      ("c,csr", "CSR binary file",
       cxxopts::value<std::string>())  // CSR
      ("m,market", "Matrix-market format file",
       cxxopts::value<std::string>())  // Market
      ("d,device", "Device to run on",
       cxxopts::value<int>()->default_value("0"))  // Device
      ("v,verbose", "Verbose output",
       cxxopts::value<bool>()->default_value("false"))  // Verbose (not used)
      ("h,help", "Print help");                         // Help

  auto result = options.parse(num_arguments, argument_array);

  if (result.count("help") ||
      (result.count("market") == 0 && result.count("csr") == 0)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  std::string filename = "";
  if (result.count("market") == 1) {
    filename = result["market"].as<std::string>();
    if (util::is_market(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }
  } else if (result.count("csr") == 1) {
    filename = result["csr"].as<std::string>();
    if (util::is_binary_csr(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }
  } else {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
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
  // std::string filename = argument_array[1];

  if (util::is_market(filename)) {
    io::matrix_market_t<row_t, edge_t, nonzero_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (util::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  printf("Matrix: %s\n", filename.c_str());
  printf("- Rows: %d\n", csr.number_of_rows);
  printf("- Nonzeros: %d\n", csr.number_of_nonzeros);

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

  double elapsed_tiled =
      test_spmv(TILED, csr, x_device, y_device, cpu_verify, debug);

  printf("%s,%d,%d,%d,%f,%f,%f,%f\n", filename.c_str(), csr.number_of_rows,
         csr.number_of_columns, csr.number_of_nonzeros, elapsed_cusparse,
         elapsed_cub, elapsed_mgpu, elapsed_tiled);

  /* ========== RESET THE GPU ========== */

  // if (deviceProp.major >= 8)
  // {
  //   // Setting the window size to 0 disable it
  //   stream_attribute.accessPolicyWindow.num_bytes = 0;

  //   // Overwrite the access policy attribute to a CUDA Stream
  //   CHECK_CUDA(cudaStreamSetAttribute(
  //       stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute));

  //   // Remove any persistent lines in L2
  //   CHECK_CUDA(cudaCtxResetPersistingL2Cache());
  // }
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
