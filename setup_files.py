#!/usr/bin/env python
"""Copy the read-only input scripts and data into the writable working dir."""
import glob
import os
import shutil


def main():
    os.makedirs("/kaggle/working/scripts", exist_ok=True)
    for p in glob.glob("/kaggle/input/sis-cloud-opt/scripts/*.py"):
        shutil.copy(p, "/kaggle/working/scripts")
    shutil.copy("/kaggle/input/sis-cloud-opt/sub_best.csv",
                "/kaggle/working/sub_best.csv")
    shutil.copy("/kaggle/input/sis-cloud-opt/targets_deep40.txt",
                "/kaggle/working/targets_deep40.txt")
    os.makedirs("/kaggle/working/scripts/out/hop", exist_ok=True)
    os.makedirs("/kaggle/working/sq_out", exist_ok=True)
    print("staging done:",
          len(glob.glob("/kaggle/working/scripts/*.py")), "scripts")


if __name__ == "__main__":
    main()
