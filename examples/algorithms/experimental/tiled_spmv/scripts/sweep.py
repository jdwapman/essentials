#!/usr/bin/env python3
# SBATCH -p bowser --gpus=V100:1

import subprocess
import os
from datetime import datetime
import time


def strip_path(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]


ESSENTIALS_BASE = "/home/jwapman/Gunrock/essentials/"

# Setup Paths for binary and datasets
BIN = ESSENTIALS_BASE + "build_release/bin/tiled_spmv"
DATASET_BASE = "/data/suitesparse_dataset/MM/"  # Must have a "/" at the end

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

PROFILEDIR = "profiles_" + now.strftime("%Y%m%d_%H:%M:%S")
os.mkdir(PROFILEDIR)

with open("datasets.txt", "r") as datasets:
    for dataset in datasets:
        for pin in [0, 1]:

            benchmark_cmd = BIN + " --cusparse --gunrock --tiled --mgpu --cub -m " + \
                dataset.rstrip()

            if pin:
                benchmark_cmd += " -p"

            # Add the json ouput in PROFILEDIR/dataset_name.json
            benchmark_cmd += " -j " + PROFILEDIR + "/" + \
                strip_path(dataset.rstrip())

            if pin:
                benchmark_cmd += "_pinned"

            benchmark_cmd += ".json"

            print(benchmark_cmd)

            retval = subprocess.run(
                benchmark_cmd, shell=True, capture_output=True)
            # Sleep 0.5 sec
            time.sleep(0.5)

print("All Tests Completed")
