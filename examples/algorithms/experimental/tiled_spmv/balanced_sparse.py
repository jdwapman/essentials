# Generate a matrix with M x N rows with a user-specified % sparsity for each
# row, so that all rows have the same # nonzeros
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import scipy.io as sio
num_rows = 6868 * 216
num_cols = 1971281 * 4
percent_sparsity = 0.99999

# Make a random matrix with the specified sparsity
import numpy as np
import scipy.sparse as sp
import random

mat = lil_matrix((num_rows, num_cols), dtype=np.int8)

# Iterate over the CSR rows, and set the number of nonzeros to the desired
# percentage.
for row in range(num_rows):
    print("Row:", row)
    # Get the number of nonzeros in the row
    num_nonzeros = int((1.0-percent_sparsity) * num_cols)
    # Get a list of random indices for the nonzeros
    nonzeros = random.sample(range(num_cols), num_nonzeros)
    # Set the nonzeros to 1
    for col in nonzeros:
        mat[row, col] = 1

coo_mat = mat.tocoo()
# Save the COO matrix to a file
sio.mmwrite('coo_matrix.mtx', coo_mat)