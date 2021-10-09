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
}

int main(int argc, char** argv) {
  test_spmv(argc, argv);
  return EXIT_SUCCESS;
}
