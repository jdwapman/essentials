#!/usr/bin/env python3
# SBATCH -p bowser --gpus=V100:1

import subprocess
import os
from datetime import datetime


def strip_path(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]


# Setup Paths for binary and datasets
BIN = "/home/jwapman/Gunrock/essentials/build_release/bin/tiled_spmv"
DATASET_BASE = "/data/suitesparse_dataset/MM/"
DATASET = "DIMACS10/ak2010"

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

results.write("File,rows,cols,nnz,cusparse,cub,mgpu\n")

PROFILEDIR = "profiles_" + now.strftime("%Y%m%d_%H:%M:%S")
os.mkdir(PROFILEDIR)


with open("datasets.txt", "r") as datasets:
    for dataset in datasets:
        benchmark_cmd = "srun " + BIN + " -m " + \
            dataset.rstrip() + " | tail -n 1 > temp_spmvbenchmark.txt"
        retval = subprocess.run(benchmark_cmd, shell=True, capture_output=True)
        print(retval)

        if "Exited with exit code 1" in str(retval.stderr):
            print("Error: " + dataset)
            continue
        else:
            print("Got return code 0 for " + dataset)
            subprocess.run("cat temp_spmvbenchmark.txt >> " +
                            RESULTS_FILE, shell=True)

        #     # Do profiling
        #     MTXNAME = strip_path(dataset)
        #     print("Profiling " + MTXNAME)

        #     profile_cmd = "ncu --target-processes application-only --replay-mode kernel --kernel-regex-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Deprecated --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --check-exit-code yes --page raw --csv --log-file " + \
        #         PROFILEDIR + "/" + MTXNAME + ".log " + BIN + " " + dataset
        #     subprocess.run(profile_cmd, shell=True)

os.remove("temp_spmvbenchmark.txt")

print("All Tests Completed")
