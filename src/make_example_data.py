"""Generate SYNTHETIC example input files (no patient data) so that the pipeline can be run end to end.

    python make_example_data.py [output_dir]          # default ../example_data

Writes, in the format expected by RvsNR_model.py and external_validate.py:
    example_mut_matrix.csv   discovery matrix: columns key, chr, site, base + one "mutation_rate_<patient_id>" column
                              per sample (depth columns are not needed)
    example_labels.csv          discovery labels: patient_id, best_response (CR/PR = responder, SD/PD = non-responder)
    example_ext_mut_matrix.csv        external matrix (same sites, different samples)
    example_ext_labels.csv      external labels: patient_id, best_response
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
chrom = [f"chr{1 + i % 22}" for i in range(N_SITES)]
pos = [100000 + 137 * i for i in range(N_SITES)]
base = ["ACGT"[i % 4] for i in range(N_SITES)]
sites = [f"{c}|{p}|{b}->N" for c, p, b in zip(chrom, pos, base)]


def make(prefix, n, id_col):
    ids = [f"{prefix}{i + 1:03d}" for i in range(n)]
    resp = rng.random(n) < 0.45
    label = np.where(resp, rng.choice(["CR", "PR"], n), rng.choice(["SD", "PD"], n))
    mat = rng.beta(2, 8, size=(N_SITES, n))                       # modification rates in [0, 1]
    mat[:N_SIGNAL, resp] += 0.15                                  # planted signal in the first sites
    mat = np.clip(mat, 0, 1)
    mat[rng.random(mat.shape) < 0.03] = np.nan                    # a few missing values
    df = pd.DataFrame(mat, columns=[f"mutation_rate_{i}" for i in ids])
    for j, (name, vals) in enumerate((("key", sites), ("chr", chrom), ("site", pos), ("base", base))):
        df.insert(j, name, vals)
    return df, pd.DataFrame({id_col: ids, "best_response": label})


for tag, prefix, n, xf, yf, id_col in (("discovery", "ec_", 80, "example_mut_matrix.csv", "example_labels.csv", "patient_id"),
                                       ("external", "ec_ext_", 40, "example_ext_mut_matrix.csv", "example_ext_labels.csv", "patient_id")):
    x, y = make(prefix, n, id_col)
    x.to_csv(os.path.join(out, xf), index=False)
    y.to_csv(os.path.join(out, yf), index=False)
    print(f"{tag}: {n} samples, {N_SITES} sites -> {xf}, {yf}")
