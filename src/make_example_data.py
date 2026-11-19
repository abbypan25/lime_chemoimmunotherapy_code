"""Generate SYNTHETIC example input files (no patient data) so that the pipeline can be run end to end.

    python make_example_data.py [output_dir]          # default ../example_data

Writes, in the format expected by RvsNR_model.py and external_validate.py:
    ec_all_mut_filtered.csv   discovery matrix: column "key" (site) + one "mutation_rate_<sample>" column per sample
    ec_set_label.csv          discovery labels: test_id, best_response (CR/PR = responder, SD/PD = non-responder)
    ec_ext_mut_all.csv        external matrix (same sites, different samples)
    ec_ext_set_label.csv      external labels
A few sites carry a planted signal; all values are random and have no biological meaning.
"""
import os
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "example_data")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
N_SITES, N_SIGNAL = 300, 12
sites = [f"chr{1 + i % 22}:{100000 + 137 * i}:{'ACGT'[i % 4]}" for i in range(N_SITES)]


def make(prefix, n):
    ids = [f"{prefix}{i + 1:03d}" for i in range(n)]
    resp = rng.random(n) < 0.45
    label = np.where(resp, rng.choice(["CR", "PR"], n), rng.choice(["SD", "PD"], n))
    mat = rng.beta(2, 8, size=(N_SITES, n))                       # modification rates in [0, 1]
    mat[:N_SIGNAL, resp] += 0.15                                  # planted signal in the first sites
    mat = np.clip(mat, 0, 1)
    mat[rng.random(mat.shape) < 0.03] = np.nan                    # a few missing values
    df = pd.DataFrame(mat, columns=[f"mutation_rate_{i}" for i in ids])
    df.insert(0, "key", sites)
    return df, pd.DataFrame({"test_id": ids, "best_response": label})


for tag, prefix, n, xf, yf in (("discovery", "D", 80, "ec_all_mut_filtered.csv", "ec_set_label.csv"),
                               ("external", "E", 40, "ec_ext_mut_all.csv", "ec_ext_set_label.csv")):
    x, y = make(prefix, n)
    x.to_csv(os.path.join(out, xf), index=False)
    y.to_csv(os.path.join(out, yf), index=False)
    print(f"{tag}: {n} samples, {N_SITES} sites -> {xf}, {yf}")
