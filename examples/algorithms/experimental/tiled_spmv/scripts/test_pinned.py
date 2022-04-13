#!/usr/bin/env python3
# SBATCH -p bowser --gpus=V100:1

import subprocess
import os
from datetime import datetime
import time


def strip_path(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]


# Setup Paths for binary and datasets
BIN = "~/Research/Gunrock/essentials/build_release/bin/tiled_spmv"
DATASET_BASE = "/media/jwapman/SSD1/"  # Must have a "/" at the end

# assert that DATASET_BASE ends with a "/"
assert DATASET_BASE[-1] == "/"

DATASET = "DIMACS10"

# Search the dataset tree for all .mtx files
if os.path.exists("datasets.txt"):
    os.remove("datasets.txt")

find_dataset_command = "find " + DATASET_BASE + \
    DATASET + " -type f -name \"*.mtx\" > datasets.txt"
# print(find_dataset_command)
subprocess.run(find_dataset_command, shell=True)

now = datetime.now()

RESULTS_FILE = "results_" + now.strftime("%Y%m%d_%H:%M:%S") + ".csv"
print(RESULTS_FILE)

results = open(RESULTS_FILE, "w")

results.write("File,rows,cols,nnz,pin,cusparse,cub,mgpu,gunrock,tiled\n")

PROFILEDIR = "profiles_" + now.strftime("%Y%m%d_%H:%M:%S")
os.mkdir(PROFILEDIR)

with open("datasets.txt", "r") as datasets:
    for dataset in datasets:
        for pin in [0, 1]:

            benchmark_cmd = BIN + " --cusparse --cub --mgpu --gunrock --tiled -m " + \
                dataset.rstrip()

            if pin:
                benchmark_cmd += " -p"


            benchmark_cmd += " | tail -n 1 > temp_spmvbenchmark.txt"

            print("Running command " + benchmark_cmd)
            retval = subprocess.run(benchmark_cmd, shell=True, capture_output=True)
            # Sleep 0.5 sec
            time.sleep(0.5)
            print(retval)

            if "Exited with exit code 1" in str(retval.stderr) or "File is not a sparse matrix" in str(retval.stderr):
                print("Error: " + dataset)
                continue
            else:
                print("Got return code 0 for " + dataset)

                # Print the program output
                print(retval.stdout.decode("utf-8"))

                subprocess.run("cat temp_spmvbenchmark.txt >> " +
                            RESULTS_FILE, shell=True)

os.remove("temp_spmvbenchmark.txt")

print("All Tests Completed")