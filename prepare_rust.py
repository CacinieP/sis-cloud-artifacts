#!/usr/bin/env python
"""Ensure a Rust toolchain exists (Kaggle CPU image may lack cargo).

Downloads the official rustup-init static binary and runs it with an
argument list (no shell).  Skips everything when cargo is already present.
"""
import os
import stat
import subprocess


def which(binname):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, binname)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def main():
    cargo = which("cargo")
    if cargo:
        print("cargo already present:", cargo)
        return cargo
    home_cargo = os.path.expanduser("~/.cargo/bin/cargo")
    if os.path.isfile(home_cargo) and os.access(home_cargo, os.X_OK):
        print("cargo already present:", home_cargo)
        return home_cargo
    print("installing rustup ...")
    url = "https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"
    dst = "/tmp/rustup-init"
    urllib.request.urlretrieve(url, dst)
    st = os.stat(dst)
    os.chmod(dst, st.st_mode | stat.S_IEXEC)
    r = subprocess.run([dst, "-y", "--profile", "minimal",
                        "--default-toolchain", "stable"],
                       capture_output=True, text=True)
    print("rustup-init rc:", r.returncode)
    print("cargo after install:", home_cargo,
          os.path.isfile(home_cargo))
    return home_cargo


import urllib.request  # noqa: E402

if __name__ == "__main__":
    main()
