#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/io/generate.hxx>
#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <iostream>
#include <ctime>

#include "spmv_cpu.hxx"
#include "spmv_cusparse.cuh"
#include "spmv_cub.cuh"
#include "spmv_moderngpu.cuh"
#include "spmv_tiled.cuh"
#include "spmv_utils.cuh"
#include "launch_params.cuh"
#include "log.h"
#include <gunrock/algorithms/spmv.hxx>
#include <nlohmann/json.hpp>
#include <typeinfo>
#include <unistd.h>
#include <fstream>

// for convenience
using json = nlohmann::json;

enum SPMV_t { MGPU, CUB, CUSPARSE, GUNROCK, TILED };
enum LB_t {
  THREAD_PER_ROW,
  WARP_PER_ROW,
  BLOCK_PER_ROW,
  MERGE_PATH,
  NONZERO_SPLIT,
  TWC
};

auto to_string(SPMV_t t) {
  switch (t) {
    case MGPU:
      return "mgpu";
    case CUB:
      return "cub";
    case CUSPARSE:
      return "cusparse";
    case GUNROCK:
      return "gunrock";
    case TILED:
      return "tiled";
    default:
      return "unknown";
  }
}

template <typename csr_t, typename vector_t, typename args_t>
double test_spmv(SPMV_t spmv_impl,
                 csr_t& sparse_matrix,
                 vector_t& d_input,
                 vector_t& d_output,
                 args_t pargs,
                 json& _results) {
  // Reset the output vector
  thrust::fill(d_output.begin(), d_output.end(), 0);

  cudaStream_t stream;
  if (pargs.count("pin")) {
    stream = setup_ampere_cache(d_input, _results);
  } else {
    CHECK_CUDA(cudaStreamCreate(&stream));
  }

  double elapsed_time = 0;

  //   Run on appropriate GPU implementation
  if (spmv_impl == MGPU) {
    printf("=== RUNNING MODERNGPU SPMV ===\n");
    elapsed_time = spmv_mgpu(stream, sparse_matrix, d_input, d_output, pargs);
  } else if (spmv_impl == CUB) {
    printf("=== RUNNING CUB SPMV ===\n");
    elapsed_time = spmv_cub(stream, sparse_matrix, d_input, d_output, pargs);
  } else if (spmv_impl == CUSPARSE) {
    printf("=== RUNNING CUSPARSE SPMV ===\n");
    elapsed_time =
        spmv_cusparse(stream, sparse_matrix, d_input, d_output, pargs);
  } else if (spmv_impl == TILED) {
    printf("=== RUNNING TILED SPMV ===\n");
    elapsed_time =
        spmv_tiled(stream, sparse_matrix, d_input, d_output, pargs, _results);
  } else if (spmv_impl == GUNROCK) {
    printf("=== RUNNING GUNROCK SPMV ===\n");
    auto G = gunrock::graph::build::from_csr<gunrock::memory_space_t::device,
                                             gunrock::graph::view_t::csr>(
        sparse_matrix.number_of_rows, sparse_matrix.number_of_columns,
        sparse_matrix.number_of_nonzeros,
        sparse_matrix.row_offsets.data().get(),
        sparse_matrix.column_indices.data().get(),
        sparse_matrix.nonzero_values.data().get());

    // Create the context
    std::shared_ptr<gunrock::gcuda::multi_context_t> context =
        std::shared_ptr<gunrock::gcuda::multi_context_t>(
            new gunrock::gcuda::multi_context_t(0, stream));
    elapsed_time = gunrock::spmv::run(G, d_input.data().get(),
                                      d_output.data().get(), context);
  } else {
    std::cout << "Unsupported SPMV implementation" << std::endl;
  }

  if (pargs.count("pin")) {
    reset_ampere_cache(stream);
  }

  if (pargs.count("verbose"))
    printf("GPU finished in %lf ms\n", elapsed_time);

  // Copy data to CPU
  if (pargs.count("cpu")) {
    thrust::host_vector<float> h_output = d_output;
    thrust::host_vector<float> h_input = d_input;

    // Run on CPU
    thrust::host_vector<float> cpu_ref(sparse_matrix.number_of_rows);
    cpu_spmv(sparse_matrix, h_input, cpu_ref);

    if (pargs.count("verbose")) {
      display(d_input, "d_input", 40);
      display(d_output, "d_output", 40);
      display(cpu_ref, "cpu_ref", 40);
    }

    // Validate
    int num_errors = check_spmv(cpu_ref, h_output, pargs);

    _results["num_errors"][to_string(spmv_impl)] = num_errors;

    // Print the number of errors
    if (pargs.count("verbose"))
      printf("Errors: %d\n", num_errors);

    if (!num_errors) {
      if (pargs.count("verbose"))
        std::cout << "Validation Successful" << std::endl;
      return elapsed_time;
    } else {
      if (pargs.count("verbose"))
        std::cout << "Validation Failed" << std::endl;
      return elapsed_time;
    }
  }

  return elapsed_time;
}

void test_spmv(int num_arguments, char** argument_array) {
  cxxopts::Options options(argument_array[0], "Tiled SPMV");

  options.add_options()  // Allows to add options.
      ("b,bin", "CSR binary file",
       cxxopts::value<std::string>())  // CSR
      ("m,market", "Matrix-market format file",
       cxxopts::value<std::string>())  // Market
      ("rows", "Number of rows for dense",
       cxxopts::value<int>()->default_value("0"))  // Rows
      ("cols", "Number of cols for dense",
       cxxopts::value<int>()->default_value("0"))  // Cols
      ("j,jsonfile", "json output filename. Can also be stdout",
       cxxopts::value<std::string>()->default_value("results.json"))  // JSON
      ("c,cpu", "Run a CPU comparison",
       cxxopts::value<bool>()->default_value("false"))  // CPU
      ("cub", "Run CUB SPMV",
       cxxopts::value<bool>()->default_value("false"))  // CUB
      ("mgpu", "Run ModernGPU SPMV",
       cxxopts::value<bool>()->default_value("false"))  // MGPU
      ("cusparse", "Run cuSparse SPMV",
       cxxopts::value<bool>()->default_value("false"))  // cuSparse
      ("gunrock", "Run Gunrock SPMV",
       cxxopts::value<bool>()->default_value("false"))  // Gunrock
      ("tiled", "Run Tiled SPMV",
       cxxopts::value<bool>()->default_value("false"))  // Tiled
      ("p,pin", "Use Ampere L2 cache pinning",
       cxxopts::value<bool>()->default_value("false"))  // Ampere L2
      ("f,fraction",
       "Fraction or multiple of L2 cache to use for input vector (-1 for "
       "untiled)",
       cxxopts::value<double>()->default_value("1.0"))  // Ampere L2 fraction
      ("i,iter", "Number of iterations",
       cxxopts::value<int>()->default_value("1"))  // Number of iterations
      ("d,device", "Device to run on",
       cxxopts::value<int>()->default_value("0"))  // GPU
      ("v,verbose", "Verbose output",
       cxxopts::value<bool>()->default_value("false"))  // Verbose
      ("h,help", "Print help");                         // Help

  json results;

  // Save command line options to the json

  auto args = options.parse(num_arguments, argument_array);

  log_cmd_args(results, args);

  // Save the current date and time to the json. But strip the newline
  time_t now = time(0);
  char* dt = ctime(&now);
  // Strip the newline from dt
  dt[strlen(dt) - 1] = '\0';
  results["time_local"] = dt;

  // Strip the newline from utc_time
  auto utc_time = asctime(gmtime(&now));
  utc_time[strlen(utc_time) - 1] = '\0';
  results["time_utc"] = utc_time;

  // Save the hostname
  char hostname[1024];
  gethostname(hostname, 1024);
  results["hostname"] = hostname;

  // Save the current git commit

  if (args.count("help") ||
      (args.count("market") == 0 && args.count("bin") == 0 &&
       args.count("rows") == 0 && args.count("cols") == 0)) {
    std::cout << options.help({""}) << std::endl;
    std::exit(0);
  }

  // Get the number of GPUs in the system
  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);
  std::cout << "Number of GPUs: " << num_gpus << std::endl;

  // Check if the GPU is valid
  if (args["device"].as<int>() >= num_gpus) {
    std::cout << "Invalid GPU" << std::endl;
    return;
  }

  printf("Using GPU %d\n", args["device"].as<int>());
  CHECK_CUDA(cudaSetDevice(args["device"].as<int>()));

  std::string filename = "";
  if (args.count("market") == 1) {
    filename = args["market"].as<std::string>();
    if (util::is_market(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(1);
    }
  } else if (args.count("bin") == 1) {
    filename = args["bin"].as<std::string>();
    if (util::is_binary_csr(filename)) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(1);
    }
  } else if (args.count("rows") == 1 && args.count("cols") == 1) {
    filename = "";
    int rows = args["rows"].as<int>();
    int cols = args["cols"].as<int>();
    if (rows > 0 && cols > 0) {
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(1);
    }
  } else {
    std::cout << options.help({""}) << std::endl;
    std::exit(1);
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
  if (args.count("market") == 1 && util::is_market(filename)) {
    io::matrix_market_t<row_t, edge_t, nonzero_t> mm;
    csr.from_coo(mm.load(filename));
  } else if (args.count("bin") == 1 && util ::is_binary_csr(filename)) {
    csr.read_binary(filename);
  } else if (args.count("rows") == 1 && args.count("cols") == 1) {
    auto dense_coo = gunrock::io::generate_dense(10, 10);
    csr.from_coo(dense_coo);
  } else {
    std::cerr << "Unknown file format: " << filename << std::endl;
    exit(1);
  }

  // Print the GPU stats
  print_gpu_stats(results);

  // Print the matrix stats
  printf("Matrix: %s\n", filename.c_str());
  printf("- Rows: %d\n", csr.number_of_rows);
  printf("- Columns: %d\n", csr.number_of_columns);
  printf("- Nonzeros: %d\n", csr.number_of_nonzeros);
  display(csr.row_offsets, "row_offsets", 40);
  display(csr.column_indices, "column_indices", 40);
  display(csr.nonzero_values, "values", 40);

  results["matrix"]["filename"] = filename.c_str();
  results["matrix"]["rows"] = csr.number_of_rows;
  results["matrix"]["columns"] = csr.number_of_columns;
  results["matrix"]["nonzeros"] = csr.number_of_nonzeros;

  thrust::host_vector<nonzero_t> x_host(csr.number_of_columns);

  srand(0);
  for (size_t idx = 0; idx < x_host.size(); idx++)
    x_host[idx] = rand() % 64;

  // --
  // Init the vectors
  thrust::device_vector<nonzero_t> x_device = x_host;
  thrust::device_vector<nonzero_t> y_device(csr.number_of_rows);

  // --
  // Run the algorithm

  double elapsed_cusparse = 0;
  double elapsed_cub = 0;
  double elapsed_mgpu = 0;
  double elapsed_gunrock = 0;
  double elapsed_tiled = 0;

  int n = args["iter"].as<int>();

  if (args.count("cusparse")) {
    double elapsed_cusparse_data[n];

    for (int i = 0; i < n; i++) {
      elapsed_cusparse_data[i] =
          test_spmv(CUSPARSE, csr, x_device, y_device, args, results);
    }

    // Take the average
    for (int i = 0; i < n; i++) {
      elapsed_cusparse += elapsed_cusparse_data[i];
    }
    elapsed_cusparse /= (double)n;
  }

  if (args.count("cub")) {
    double elapsed_cub_data[n];

    for (int i = 0; i < n; i++) {
      elapsed_cub_data[i] =
          test_spmv(CUB, csr, x_device, y_device, args, results);
    }

    // Take the average
    for (int i = 0; i < n; i++) {
      elapsed_cub += elapsed_cub_data[i];
    }
    elapsed_cub /= (double)n;
  }

  if (args.count("mgpu")) {
    double elapsed_mgpu_data[n];

    for (int i = 0; i < n; i++) {
      elapsed_mgpu_data[i] =
          test_spmv(MGPU, csr, x_device, y_device, args, results);
    }

    // Take the average
    for (int i = 0; i < n; i++) {
      elapsed_mgpu += elapsed_mgpu_data[i];
    }
    elapsed_mgpu /= (double)n;
  }

  if (args.count("gunrock")) {
    double elapsed_gunrock_data[n];

    for (int i = 0; i < n; i++) {
      elapsed_gunrock_data[i] =
          test_spmv(GUNROCK, csr, x_device, y_device, args, results);
    }

    // Take the average
    for (int i = 0; i < n; i++) {
      elapsed_gunrock += elapsed_gunrock_data[i];
    }
    elapsed_gunrock /= (double)n;
  }

  if (args.count("tiled")) {
    double elapsed_tiled_data[n];

    for (int i = 0; i < n; i++) {
      elapsed_tiled_data[i] =
          test_spmv(TILED, csr, x_device, y_device, args, results);
    }

    // Take the average
    for (int i = 0; i < n; i++) {
      elapsed_tiled += elapsed_tiled_data[i];
    }
    elapsed_tiled /= (double)n;
  }

  results["runtime"]["cusparse"] = elapsed_cusparse;
  results["runtime"]["cub"] = elapsed_cub;
  results["runtime"]["mgpu"] = elapsed_mgpu;
  results["runtime"]["gunrock"] = elapsed_gunrock;
  results["runtime"]["tiled"] = elapsed_tiled;

  printf("%s,%d,%d,%d,%d,%f,%f,%f,%f,%f\n", filename.c_str(),
         csr.number_of_rows, csr.number_of_columns, csr.number_of_nonzeros,
         args["pin"].as<bool>(), elapsed_cusparse, elapsed_cub, elapsed_mgpu,
         elapsed_gunrock, elapsed_tiled);

  // Log a success
  results["success"] = true;

  // Save the JSON file
  auto json_filename = args["jsonfile"].as<std::string>();

  if (json_filename == "stdout") {
    std::cout << results.dump(4) << std::endl;
  } else {
    std::ofstream json_file(json_filename);
    json_file << results.dump(4);
    json_file.close();
  }
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
