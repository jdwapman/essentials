from math import ceil
from scipy.sparse import csr_matrix
import scipy.io as sio
import sys

tile_size = 10
num_SMs = 2

# Load the matrix file in the first input argument
coo_mat = sio.mmread(sys.argv[1])
csr_mat = csr_matrix(coo_mat)

print("Loaded matrix file:", sys.argv[1])
print("Matrix Statistics:")
print("\t- Number of rows:", csr_mat.shape[0])
print("\t- Number of columns:", csr_mat.shape[1])
print("\t- Number of non-zero elements:", csr_mat.nnz)

print("Using a tile size of:", tile_size)

# Determine load imbalance between tiles
num_row_tiles = ceil(csr_mat.shape[0] / tile_size)
num_col_tiles = ceil(csr_mat.shape[1] / tile_size)
print("\t- Number of row tiles:", num_row_tiles)
print("\t- Number of column tiles:", num_col_tiles)

iter = 0
for row_tile_idx in range(0, num_row_tiles, num_SMs):
    for col_tile_idx in range(0, num_col_tiles):
        print("Iteration: ", iter)
        iter += 1
        for sm_idx in range(0, num_SMs):
            print("\t- SM %d: (%d,%d)" % (sm_idx, row_tile_idx, col_tile_idx * num_SMs + sm_idx))
