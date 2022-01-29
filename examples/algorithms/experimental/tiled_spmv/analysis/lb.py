from math import ceil
from scipy.sparse import csr_matrix
import scipy.io as sio
import sys
import numpy as np

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

for col_tile_idx in range(0, num_col_tiles):
    print("Column tile:", col_tile_idx)
    for row_tile_idx in range(0, num_row_tiles, num_SMs):
        print("\tRow tile:", row_tile_idx)

        # Create a numpy array equal to the number of SMs
        # This will be used to store the number of non-zero elements
        # in each SM
        sm_nnz = [0] * num_SMs
        for sm_idx in range(0, num_SMs):
            print("\t- SM %d: (%d,%d)" %
                  (sm_idx, row_tile_idx + sm_idx, col_tile_idx))

            # Get the tile corresponding to this SM
            row_start = (row_tile_idx + sm_idx) * tile_size
            row_end = min(row_start + tile_size, csr_mat.shape[0])
            col_start = col_tile_idx * tile_size
            col_end = min(col_start + tile_size, csr_mat.shape[1])
            tile = csr_mat[row_start:row_end, col_start:col_end]
            print("\t\t- Row range:", row_start, row_end)
            print("\t\t- Column range:", col_start, col_end)
            print("\t\t- Nonzero elements in tile:", tile.nnz)
            sm_nnz[sm_idx] = tile.nnz

        # Compute the load imbalance using the coefficient of variation
        # of the number of non-zero elements in each SM. Use numpy functions
        sm_nnz_mean = np.mean(sm_nnz)
        sm_nnz_std = np.std(sm_nnz)
        print("\t- Mean number of non-zero elements per SM:", sm_nnz_mean)
        print("\t- Standard deviation of non-zero elements per SM:", sm_nnz_std)
        print("\t- Load imbalance:", sm_nnz_std / sm_nnz_mean)
