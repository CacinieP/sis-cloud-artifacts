#!/usr/bin/env python
"""Build evand's Rust packer on Linux from the minimal source unit.

Every cargo invocation is an inline literal argument list; candidates are
tried in order until one exists.
"""
import os
import shutil
import subprocess

BUILD_DIR = "/kaggle/working/packer_src"
BIN_RELEASE = "/kaggle/working/packer_src/target/release"
SRC_IN = "/kaggle/working/art/packer_src"


def main():
    if not os.path.isdir(BUILD_DIR):
        shutil.copytree(SRC_IN, BUILD_DIR)
    r = None
    try:
        r = subprocess.run(["/root/.cargo/bin/cargo", "build", "--release"],
                           cwd=BUILD_DIR, capture_output=True, text=True)
    except (FileNotFoundError, PermissionError):
        r = None
    if r is None:
        try:
            r = subprocess.run(["/usr/local/cargo/bin/cargo", "build", "--release"],
                               cwd=BUILD_DIR, capture_output=True, text=True)
        except (FileNotFoundError, PermissionError):
            r = None
    if r is None:
        try:
            r = subprocess.run(["/usr/bin/cargo", "build", "--release"],
                               cwd=BUILD_DIR, capture_output=True, text=True)
        except (FileNotFoundError, PermissionError):
            r = None
    if r is None:
        try:
            r = subprocess.run(["/opt/cargo/bin/cargo", "build", "--release"],
                               cwd=BUILD_DIR, capture_output=True, text=True)
        except (FileNotFoundError, PermissionError):
            r = None
    if r is None:
        raise SystemExit("no usable cargo found; install rust first")
    print("cargo rc:", r.returncode)
    print((r.stderr or "")[-500:])
    packer = BIN_RELEASE + "/packer"
    if not os.path.exists(packer):
        raise SystemExit("packer build failed")
    base1 = "/kaggle/working/scripts/external/evand/search/packer/target/release"
    base2 = "/kaggle/working/external/evand/search/packer/target/release"
    for base in (base1, base2):
        os.makedirs(base, exist_ok=True)
        shutil.copy(packer, os.path.join(base, "packer"))
        fq = BIN_RELEASE + "/fq"
        if os.path.exists(fq):
            shutil.copy(fq, os.path.join(base, "fq"))
    print("binaries placed at both repo-layout locations")


if __name__ == "__main__":
    main()
