#include <gunrock/algorithms/experimental/async/bfs.hxx>
#include "spmv_cpu.hxx"

using namespace gunrock;
using namespace experimental;
using namespace memory;

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

  using csr_t = 
      format::csr_t<memory_space_t::device, row_t, edge_t, nonzero_t>;

  // --
  // IO
  printf("Hello World!\n");
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

  thrust::device_vector<nonzero_t> x(csr.number_of_rows);
  thrust::device_vector<nonzero_t> y(csr.number_of_rows);

  // --
  // Build graph + metadata

  auto sparse_matrix =
      graph::build::from_csr<memory_space_t::device,
                             graph::view_t::csr>(
          csr.number_of_rows,               // rows
          csr.number_of_columns,            // columns
          csr.number_of_nonzeros,           // nonzeros
          csr.row_offsets.data().get(),     // row_offsets
          csr.column_indices.data().get(),  // column_indices
          csr.nonzero_values.data().get()   // values
      );

  double elapsed_cusparse = 0;

  printf("%s,%d,%d,%d,%f\n", filename.c_str(), csr.number_of_rows,
         csr.number_of_columns, csr.number_of_nonzeros,
         elapsed_cusparse);
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
