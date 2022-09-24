import numpy as np
import scipy.io as sio
import scipy.sparse as sps

# GPU-constrained tile sizes. Rows are constrained by the amount of L1 shmem
# and columns are constrained by the amount of L2 cache.
tile_num_rows = 6868
tile_num_cols = 10485760

num_SMs = 108 # For A100
occupancy = 2
num_blocks = num_SMs * occupancy

# This is arbitrary, and influences how large the end matrix is.
num_spatial_tiles = 1
num_temporal_tiles = 1

# Create an empty CSR matrix
tile = sps.lil_matrix((tile_num_rows, tile_num_cols))

# Iterate over the rows of the CSR tile:
for row in range(tile_num_rows):
    # Get a random integer between 0 and tile_num_rows/32
    dense_row_start_index = np.random.randint(0, tile_num_rows/32) * 32

    # Given the start index, set the following 32 elements to 1 in the CSR matrix
    # ONLY for the first row of the matrix
    tile[row, dense_row_start_index:dense_row_start_index+32] = 1

# Stack the tile vertically by the number of SMs
mat = sps.vstack([tile] * num_blocks)

# Stack the matrix vertically by the number of spatial tiles
mat = sps.vstack([mat] * num_spatial_tiles)

# Stack the matrix horizontally by the number of temporal tiles
mat = sps.hstack([mat] * num_temporal_tiles)

# Print the shape
print("Shape: ", mat.shape)

# Print the number of nonzeros
print("Nonzeros: ", mat.nnz)

# Save to mtx format on disk
sio.mmwrite("mat.mtx", mat)