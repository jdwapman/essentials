import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from statistics import geometric_mean
import json

# Takes a csv file generated by the tiled spmv implementation and creates lots
# of plots.

if len(sys.argv) != 3:
    print("Usage: python3 ./plot/py datadir plotdir")
    sys.exit(1)

datadir = sys.argv[1]
plotdir = sys.argv[2]

# Create plotdir if it doesn't already exist
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

# Load all json files in the plotdir folder
jsonfiles = []
for filename in os.listdir(datadir):
    f = os.path.join(datadir, filename)
    if os.path.isfile(f):
        f = json.load(open(f))
        jsonfiles.append(f)

# For each json file, extract the "runtime" field
nonzeros = []
runtimes = []
pinned = []
for f in jsonfiles:
    if "pin" in f["argv"]:
        nonzeros.append(f["matrix"]["nonzeros"])
        runtimes.append(f["runtime"])


# Plot the runtimes against the nonzeros
# Note that there are multiple fields in "runtimes"
# (e.g. "runtime", "runtime_cub", "runtime_cusparse", etc.)


df = pd.DataFrame.from_dict(runtimes)
df2 = pd.DataFrame(nonzeros, columns=["nonzeros"])
df3 = pd.concat([df2, df], axis=1)

sns.set_theme()
data = df3

# Get columns 0, 4, 5, and 6 of the "data" DataFrame as a new dataframe
new_df = data[['nonzeros', 'cusparse', 'cub', 'tiled']]

# Melt the dataframe to make it easier to plot
dfm = new_df.melt('nonzeros', var_name='method', value_name='time')
plot = sns.scatterplot(data=dfm, x='nonzeros', y='time', hue='method')

# Set the x and y axis as log scale
plot.set(xscale="log", yscale="log")

# Set the y axis to use decimal notation
plot.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Set the x and y axis labels
plot.set(xlabel='nonzeros', ylabel='time (s)')

# Set the title
plot.set_title('SpMV Benchmark')

# Save the plot as a file
plot.get_figure().savefig(os.path.join(plotdir, 'nnz_time.png'))