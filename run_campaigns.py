#!/usr/bin/env python
"""Run the hop + fq-squeeze campaigns on the v6 base with a wall-clock guard.

Phase 1: hop on the top-90 excess puzzles (12 min each, 4 procs).
Phase 2: fq squeeze on the 175 newly adopted packings (7 min each, 4 procs).
Each phase is skipped if the elapsed budget is already exhausted; Kaggle CPU
kernels cap at 12 h, so we target ~10.5 h of work.
"""
import subprocess
import time

BUDGET_S = 10.5 * 3600
T0 = time.time()


def remaining():
    return BUDGET_S - (time.time() - T0)


def main():
    print("=== hop campaign (top-90 excess, 12 min each, 4 procs) ===", flush=True)
    subprocess.run(["/usr/bin/python3", "-u",
                    "/kaggle/working/scripts/hop_search.py",
                    "/kaggle/working/sub_best.csv",
                    "/kaggle/working/art/targets_hop.txt",
                    "12", "4", "101"],
                   capture_output=False, text=True)
    print(f"hop done at +{(time.time() - T0) / 3600:.2f}h", flush=True)

    if remaining() < 40 * 60:
        print("budget exhausted, skip squeeze phase", flush=True)
        return

    print("=== fq squeeze campaign (175 new packings, 7 min each, 4 procs) ===", flush=True)
    subprocess.run(["/usr/bin/python3", "-u",
                    "/kaggle/working/scripts/squeeze_campaign.py",
                    "/kaggle/working/sub_best.csv",
                    "/kaggle/working/art/targets_hop.txt",
                    "420", "4", "77",
                    "/kaggle/working/sq_out/results.jsonl"],
                   capture_output=False, text=True)
    print(f"campaigns done at +{(time.time() - T0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
