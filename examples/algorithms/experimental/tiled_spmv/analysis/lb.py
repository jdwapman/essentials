from math import ceil
from scipy.sparse import csr_matrix
import scipy.io as sio
import sys
import numpy as np
from scipy.sparse.csgraph import reverse_cuthill_mckee
import argparse
import statistics

# SETUP TILE SIZES
spatial_tile_rows = 1000
spatial_tile_cols = 1000
temporal_tile_rows = 100
temporal_tile_cols = 100
block_temporal_tile_rows = 10
block_temporal_tile_cols = 100

num_SMs = 108
occupancy = 2
num_blocks = num_SMs * occupancy

# Parse the input arguments
# Usage: python lb.py [-r reoder] <matrix_file>
# -r: reorder the matrix
# -t: tile size
# -n: number of SMs
# -h: print help
# Use argparse to parse the input arguments
parser = argparse.ArgumentParser(description='Parse the input arguments')
parser.add_argument('-r', '--reorder', type=str, default='default',
                    help='reorder the matrix')
parser.add_argument('matrix_file', type=str,
                    help='matrix file')


args = parser.parse_args()

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

# Update the tile sizes
spatial_tile_rows = min(spatial_tile_rows, analyze_mat.shape[0])
spatial_tile_cols = analyze_mat.shape[1]
temporal_tile_rows = min(temporal_tile_rows, analyze_mat.shape[0])
temporal_tile_cols = min(temporal_tile_cols, spatial_tile_cols)
block_temporal_tile_rows = min(block_temporal_tile_rows, analyze_mat.shape[0])
block_temporal_tile_cols = min(block_temporal_tile_cols, spatial_tile_cols)

num_spatial_row_tiles = ceil(analyze_mat.shape[0] / spatial_tile_rows)
num_spatial_col_tiles = ceil(analyze_mat.shape[1] / spatial_tile_cols)
num_temporal_row_tiles = ceil(num_spatial_row_tiles / temporal_tile_rows)
num_temporal_col_tiles = ceil(num_spatial_col_tiles / temporal_tile_cols)
num_block_temporal_row_tiles = ceil(
    num_temporal_row_tiles / block_temporal_tile_rows)
num_block_temporal_col_tiles = ceil(
    num_temporal_col_tiles / block_temporal_tile_cols)

for spatial_tile_row_idx in range(0, num_spatial_row_tiles):
    for spatial_tile_col_idx in range(0, num_spatial_col_tiles):

        # Get the spatial matrix tile
        row_start = spatial_tile_row_idx * spatial_tile_rows
        row_end = min(row_start + spatial_tile_rows, analyze_mat.shape[0])
        col_start = spatial_tile_col_idx * spatial_tile_cols
        col_end = min(col_start + spatial_tile_cols, analyze_mat.shape[1])
        spatial_tile = analyze_mat[row_start:row_end, col_start:col_end]

        # Temporal tile loop
        for temporal_tile_row_idx in range(0, num_temporal_row_tiles):
            for temporal_tile_col_idx in range(0, num_temporal_col_tiles):
                # Get the temporal matrix tile
                row_start = temporal_tile_row_idx * temporal_tile_rows
                row_end = min(row_start + temporal_tile_rows,
                              spatial_tile.shape[0])
                col_start = temporal_tile_col_idx * temporal_tile_cols
                col_end = min(col_start + temporal_tile_cols,
                              spatial_tile.shape[1])

                temporal_tile = spatial_tile[row_start:row_end,
                                             col_start:col_end]

                # Get the loab imbalance between each temporal tile block
                block_nnzs = []

                for block_idx in range(0, num_blocks):
                    row_start = block_idx * block_temporal_tile_rows
                    row_end = min(row_start + block_temporal_tile_rows,
                                  temporal_tile.shape[0])
                    col_start = 0
                    col_end = min(col_start + block_temporal_tile_cols,
                                  temporal_tile.shape[1])

                    block_tile = temporal_tile[row_start:row_end,
                                               col_start:col_end]
                    block_nnzs.append(block_tile.nnz)

                # Print the data
                # print("Spatial Tile:", spatial_tile_row_idx, spatial_tile_col_idx)
                # print("Temporal Tile:", temporal_tile_row_idx,
                #       temporal_tile_col_idx)
                # print("Block NNZs:", block_nnzs)

                # Compute the load imbalance
                m = statistics.mean(block_nnzs)
                if m > 0:
                    imbalance = max(block_nnzs) / m
                    print("Imbalance:", imbalance)
                else:
                    imbalance = 0
                    print("Imbalance:", imbalance)
