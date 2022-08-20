# Generate increasingly large dense matrices, and compare speedup of tiled
# vs cusparse implementations.
# Note that this must be run in the top level of the repo
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
import subprocess
from scipy import interpolate

MAX_M = 10000
MAX_N = 10000

BIN = "./build_release/bin/tiled_spmv"
ARGS = "--tiled --cusparse -f -1 --iter 3"

N_SAMPLES = 10

# M x N Numpy array to save speedups
speedups = np.ones((MAX_M+1, MAX_N+1))

for sample in range(N_SAMPLES):
    # 1. Pick a random matrix size within the range [1, MAX_M], [1, MAX_N]
    m = random.randint(1, MAX_M)
    n = random.randint(1, MAX_N)

    # 2. Generate a random matrix with the specified dimensions and save it to
    # a file
    mat = np.random.rand(m, n)
    # Convert to COO
    coo_mat = coo_matrix(mat)

    sio.mmwrite("matrix.mtx", coo_mat)

    # 3. Run the program with the specified arguments using the subprocess module
    cmd = BIN + " -m " + " matrix.mtx " + ARGS
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    # 4. Read the output from the program
    output = p.stdout.read()

    # 5. Get the last line of the output
    last_line = output.splitlines()[-1]
    last_line = last_line.decode("utf-8")

    # Format is filename,nrows,ncols,nnz,cusparse,ignore,ignore,ignore,tiled.
    # Want to extract the cusparse and tiled times.
    times = last_line.split(",")
    cusparse_time = float(times[5])
    tiled_time = float(times[-1])

    # 6. Save the speedup to the speedups array
    speedups[m, n] = cusparse_time / tiled_time

print(speedups)

# Save the speedups to a human readable file
sio.savemat("speedups.mat", {"speedups": speedups})

# Plot the speedups where M and N are the x and y axes
# and use the 3rd dimension to plot the speedup
ax = plt.axes(projection='3d')

# As an example, the numpy array is something like:

# [0 1.5 2.5, 0 0.4 0, 2 0.1 1.2]

# Want to use the x and y coordinates to plot the z coordinate
# So we need to create a meshgrid of the x and y coordinates
# and then use that to create a 3D surface plot

x = np.arange(0, MAX_M+1)
y = np.arange(0, MAX_N+1)
X, Y = np.meshgrid(x, y)

# Interpolate the speedups
f = interpolate.interp2d(x, y, speedups, kind='cubic')

speedups_interpolated = f(x, y)

print(speedups_interpolated)

# Set speedups where the value is 1 to nan
# NOT speedups_interpolated
speedups[speedups == 1] = np.nan

# 3d scatter plot of speedupts
ax.scatter3D(X, Y, speedups, cmap='viridis')


# Save the plot
plt.savefig("speedups.png")

# Print the geomean speedup
# Get the locations where the speedup is not nan
# (i.e. where the matrix size is not too large)
locations = np.where(~np.isnan(speedups))
# Get the speedup values at those locations
speedups_at_locations = speedups[locations]
# Take the geometric mean of the speedups at those locations
# ie the square root of all values multiplied together
geomean = np.prod(speedups_at_locations) ** (1.0/len(speedups_at_locations))
print("Geometric mean speedup:", geomean)