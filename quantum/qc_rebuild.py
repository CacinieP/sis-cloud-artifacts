#!/usr/bin/env python
"""Rebuild ANGLE_k circuits from a factorization npz on a custom Givens tree.

Runs inside the qualtran-capable environment created by the driver.  The
Givens tree edges are taken from the reference submission npz, the angle
words are re-encoded from the given factorization, the kickmix circuits are
rebuilt, scored, and written to the output directory as a verifier-style
run directory (angle_k files plus submission.npz).

Usage: qc_rebuild.py FACTORIZATION_NPZ REFERENCE_NPZ OUT_DIR BROTA BATCH
"""
import json
import os
import sys

import numpy as np

SK = "/kaggle/working/sk"
sys.path.insert(0, SK)

from starter.task1_factorization.solution_io import read_factorization
from starter.task2_angles.encode import encode_inputs
from starter.task3_circuits.angle_k_kickmix import write_kmx_family
from starter.task3_circuits.build_circuits import build_circuits
from verifier.reference import FACTORIZATION_KEYS, load_inputs
from verifier.score import KEEP_BITS, evaluate_circuits, load_inputs


def main():
    fact_npz, ref_npz, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    brot, batch = int(sys.argv[4]), int(sys.argv[5])
    with np.load(ref_npz, allow_pickle=False) as zf:
        tree_edges = np.asarray(zf["tree_edges"], dtype=np.int64)
    params, _ = read_factorization(fact_npz)
    inputs = load_inputs("FeMoCo76o113e", params)
    words = encode_inputs(inputs, tree_edges, brot, "nearest")
    circuits = build_circuits(words, brot, batch)
    result = evaluate_circuits(inputs, circuits, tree_edges, KEEP_BITS)
    if not np.array_equal(result["angle_words"], words):
        print("REBUILD_MISMATCH")
        raise SystemExit(2)
    os.makedirs(out_dir, exist_ok=True)
    write_kmx_family(circuits, out_dir, header=["custom tree rebuild"],
                     overwrite=True)
    np.savez(os.path.join(out_dir, "submission.npz"),
             tree_edges=tree_edges, angle_words=words,
             **{key: inputs.params[key] for key in FACTORIZATION_KEYS})
    res = result["resources"]
    met = result["metrics"]
    print("REBUILD_OK " + json.dumps({
        "score": float(result["score"]),
        "lambda_eff": float(met["lambda_eff"]),
        "toffoli": float(res["toffoli"]),
        "qubits": int(res["qubits"]),
        "error_1": float(met["error_1"]),
        "error_2": float(met["error_2"]),
    }))
    print("REBUILD_DIR " + out_dir)


if __name__ == "__main__":
    main()
