from math import ceil
from scipy.sparse import csr_matrix
import scipy.io as sio
import sys
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee
import argparse

tile_size = 10
num_SMs = 2

# Parse the input arguments
# Usage: python lb.py [-r reoder] [-t tile_size] [-n num_SMs] <matrix_file>
# -r: reorder the matrix
# -t: tile size
# -n: number of SMs
# -h: print help
# Use argparse to parse the input arguments
parser = argparse.ArgumentParser(description='Parse the input arguments')
parser.add_argument('-r', '--reorder', type=str, default='default',
                    help='reorder the matrix')
# parser.add_argument('-t', '--tile_size', type=int,
#                     default=tile_size, help='tile size')
parser.add_argument('-n', '--num_SMs', type=int,
                    default=num_SMs, help='number of SMs')
parser.add_argument('matrix_file', type=str,
                    help='matrix file')


args = parser.parse_args()

# Get the tile size from the input if it's not default
# if args.tile_size != tile_size:
#     tile_size = args.tile_size

# Get the number of SMs from the input if it's not default
if args.num_SMs != num_SMs:
    num_SMs = args.num_SMs

# Load the matrix file in the first input argument
coo_mat = sio.mmread(sys.argv[1])
csr_mat = csr_matrix(coo_mat)

analyze_mat = csr_mat

if args.reorder == 'rcm':
    print("RCM reordering")
    perm = reverse_cuthill_mckee(csr_mat)
    rcm_mat = csr_mat[perm, :][:, perm]
    analyze_mat = rcm_mat
elif args.reorder == 'random':
    print("Random reordering")
    # Shuffle the sparse matrix
    perm = np.random.permutation(csr_mat.shape[0])
    analyze_mat = csr_mat[perm, :][:, perm]

print("Loaded matrix file:", sys.argv[1])
print("Matrix Statistics:")
print("\t- Number of rows:", analyze_mat.shape[0])
print("\t- Number of columns:", analyze_mat.shape[1])
print("\t- Number of non-zero elements:", analyze_mat.nnz)

tile_size_rows = 6336
tile_size_cols = 786432
print("Using a tile size of:", tile_size_rows, "x", tile_size_cols)

# Determine load imbalance between tiles
num_row_tiles = ceil(analyze_mat.shape[0] / tile_size_rows)
num_col_tiles = ceil(analyze_mat.shape[1] / tile_size_cols)
print("\t- Number of row tiles:", num_row_tiles)
print("\t- Number of column tiles:", num_col_tiles)

col_nnz = [0] * num_col_tiles
for col_tile_idx in range(0, num_col_tiles):
    print("Column tile:", col_tile_idx)
    for row_tile_idx in range(0, num_row_tiles):
        print("\tRow tile:", row_tile_idx)

        # Get the tile corresponding to this SM
        row_start = row_tile_idx * tile_size_rows
        row_end = min(row_start + tile_size_rows, analyze_mat.shape[0])
        col_start = col_tile_idx * tile_size_cols
        col_end = min(col_start + tile_size_cols, analyze_mat.shape[1])
        tile = analyze_mat[row_start:row_end, col_start:col_end]
        print("\t\t- Row range:", row_start, row_end)
        print("\t\t- Column range:", col_start, col_end)
        print("\t\t- Nonzero elements in tile:", tile.nnz)
        col_nnz[col_tile_idx] += tile.nnz

# Compute the load imbalance using the coefficient of variation
# of the number of non-zero elements in each SM. Use numpy functions
nnz_mean = np.mean(col_nnz)
nnz_std = np.std(col_nnz)
print("- Mean:", nnz_mean)
print("- Standard deviation:", nnz_std)
print("- Load imbalance:", nnz_std / nnz_mean)
