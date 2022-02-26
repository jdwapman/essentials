#pragma once

#define CHECK_CUDA(func)                                                   \
  {                                                                        \
    cudaError_t status = (func);                                           \
    if (status != cudaSuccess) {                                           \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, \
             cudaGetErrorString(status), status);                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

template <typename vector_t>
void display(vector_t v, std::string name, bool verbose = true) {
  if (verbose) {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < v.size() && (i < 40); i++)
      std::cout << v[i] << " ";

    if (v.size() >= 40) {
      std::cout << "...";
    }
    std::cout << "]" << std::endl;
  }
}