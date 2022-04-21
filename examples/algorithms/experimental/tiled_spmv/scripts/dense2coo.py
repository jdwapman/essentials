#!/usr/bin/env python3

from scipy.sparse import csr_matrix
import numpy as np
import argparse
import sys
import scipy.io as sio

# Python helper to create a dense matrix and write it to disk in coo format
# Usage: python dense2coo.py <matrix_file> <rows> <cols> <data_type>
# Note that all

parser = argparse.ArgumentParser(description='Dense Matrix to COO Matrix')
parser.add_argument('--rows', type=int,
                    help='number of rows')
parser.add_argument('--cols', type=int,
                    help='number of columns')
parser.add_argument('--file', type=str,
                    help='output matrix file')
parser.add_argument('--lower_bound', type=int, nargs='?',
                    default=0, help='lower bound')
parser.add_argument('--upper_bound', type=int, nargs='?',
                    default=1, help='upper bound')
parser.add_argument('--data_type', type=str, nargs='?', default='float',
                    help='data type. Options: float, double, float32, float64, int, int32, int64')
parser.add_argument('--sparse', type=int, help='percent sparsity (0-100)')
parser.add_argument('--tile_rows', type=int, help='tile rows')
parser.add_argument('--tile_cols', type=int, help='tile cols')

args = parser.parse_args()

# Exit if the number of rows or columns is not specified
if args.rows is None or args.cols is None:
    print("Please specify the number of rows and columns")
    sys.exit(1)

if args.sparse is not None:
    sparse = args.sparse
    if sparse < 0 or sparse > 100:
        print("Please specify a valid percent sparsity (0-100)")
        sys.exit(1)

# Create the random matrix using the provided dimensions and data type.
mat = np.random.rand(args.rows, args.cols)

# Size to the upper and lower bounds
mat = mat * (args.upper_bound - args.lower_bound) + args.lower_bound

# Convert the matrix to the specified data type
if args.data_type == 'float':
    mat = mat.astype(np.float)
elif args.data_type == 'double':
    mat = mat.astype(np.double)
elif args.data_type == 'float32':
    mat = mat.astype(np.float32)
elif args.data_type == 'float64':
    mat = mat.astype(np.float64)
elif args.data_type == 'int':
    mat = np.round(mat).astype(np.int)
elif args.data_type == 'int32':
    mat = np.round(mat).astype(np.int32)
elif args.data_type == 'int64':
    mat = np.round(mat).astype(np.int64)
else:
    print("ERROR: Unsupported data type:", args.data_type)
    sys.exit(1)

# Sparsify
# Iterate over the tiles in the matrix (first assuming that it's an
# even multiple of the tile size) and set each tile 

sparse_matrix = csr_matrix(mat)

# Write to disk
# sio.mmwrite(args.file, sparse_matrix)

# Open the file
with open(args.file, 'w') as f:
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write("%d %d %d\n" % (args.rows, args.cols, sparse_matrix.nnz))
    for i in range(sparse_matrix.shape[0]):
        for j in range(sparse_matrix.indptr[i], sparse_matrix.indptr[i+1]):
            f.write("%d %d %f\n" %
                    (i+1, sparse_matrix.indices[j]+1, sparse_matrix.data[j]))
