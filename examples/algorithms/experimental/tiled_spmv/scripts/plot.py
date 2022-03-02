import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Takes a csv file generated by the tiled spmv implementation and creates lots
# of plots.

if len(sys.argv) != 3:
    print("Usage: python3 ./plot/py parsed_datafile.csv plotdir")
    sys.exit(1)

datafile = sys.argv[1]
plotdir = sys.argv[2]

# Create plotdir if it doesn't already exist
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

data = pd.read_csv(datafile)

xvals = ['rows', 'cols', 'nnz']

sns.set_theme()

# Get columns 0, 4, 5, and 6 of the "data" DataFrame as a new dataframe
new_df = data[['nnz', 'cusparse', 'cub', 'mgpu']]

dfm = new_df.melt('nnz', var_name='method', value_name='time')

plot = sns.scatterplot(data=dfm, x='nnz', y='time', hue='method')

# Set the x and y axis as log scale
plot.set(xscale="log", yscale="log")

# Set the y axis to use decimal notation
plot.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))

# Set the x and y axis labels
plot.set(xlabel='nnz', ylabel='time (s)')

# Set the title
plot.set_title('SpMV Benchmark')

# Save the plot as a file
plot.get_figure().savefig(os.path.join(plotdir, 'nnz_time.png'))