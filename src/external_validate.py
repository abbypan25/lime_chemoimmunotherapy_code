"""External validation of a locked cfRNA-modification model (output of RvsNR_model.py, parts C + F).

The fitted pipeline (feature filtering, imputation, scaling, selection and classifier) and the
Youden cutoff were fixed on the discovery cohort; nothing is re-fitted or re-calibrated here.

Usage (fs_transformers.py must be in the same directory as this script):
    python external_validate.py --x my_ext_mut_matrix.csv --y my_ext_labels.csv \
        --out models_RvsNR/mut_<label file name>/ --prefix my_external --name MyCohort
    (file names are placeholders; the input files are not part of the repository)

Inputs
    --x      external modification-rate matrix: column "key" (site, same format as the training file) and
             one "mutation_rate_<sample>" column per sample. Only the final selected features of the model
             (F_final_features.csv) are required; other sites may be absent and are ignored by the model.
    --y      external labels: columns patient_id, best_response (CR/PR = responder = 1, SD/PD = non-responder = 0)
    --out    directory written by RvsNR_model.py (parts C + F) that holds
             F_final_<model>.joblib, F_feature_columns.txt, F_final_features.csv, F_final_<model>_cutoff.txt
Options
    --model   which final model to apply (default RF)
    --prefix  prefix of the output files (default external)
    --name    label printed in the report, e.g. EC or GC (default External)
    --n-boot  bootstrap resamples for the AUC 95% CI (default 1000)
    --seed    random seed of the bootstrap (default 0)
Outputs (written to --out)
    <prefix>_predictions.csv, <prefix>_roc.csv, <prefix>_metrics.csv
"""
import os
import sys
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# fs_transformers must be importable for joblib.load (the pipeline was pickled with these classes)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--x", required=True, help="external modification-rate matrix (csv)")
ap.add_argument("--y", required=True, help="external label file (csv)")
ap.add_argument("--out", required=True, help="model directory written by RvsNR_model.py")
ap.add_argument("--model", default="RF", help="final model to apply (default RF)")
ap.add_argument("--prefix", default="external", help="prefix of the output files")
ap.add_argument("--name", default="External", help="label used in the printed report")
ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the AUC CI (default 1000)")
ap.add_argument("--seed", type=int, default=0, help="bootstrap random seed (default 0)")
a = ap.parse_args()
OUT = a.out

# ---------- 1. locked model, training columns, final features, cutoff ----------
est = joblib.load(os.path.join(OUT, f"F_final_{a.model}.joblib"))
with open(os.path.join(OUT, "F_feature_columns.txt")) as fh:
    cols = [line.strip() for line in fh if line.strip()]
final_features = pd.read_csv(os.path.join(OUT, "F_final_features.csv"))
final_features = final_features.loc[final_features.model == a.model, "feature"].astype(str).str.strip().tolist()
cutoff_path = os.path.join(OUT, f"F_final_{a.model}_cutoff.txt")
if os.path.exists(cutoff_path):
    with open(cutoff_path) as fh:
        cutoff = float(fh.read().strip())
else:
    cutoff = None
    print(f"WARNING: {cutoff_path} not found; only the default 0.5 cutoff will be reported.")
print("Training input features:", len(cols), "| final features:", len(final_features), "| Youden cutoff:", cutoff)

# ---------- 2. external matrix ----------
m = pd.read_csv(a.x, encoding="utf-8-sig")
m["key"] = m["key"].astype(str).str.strip()
if m["key"].duplicated().any():
    raise ValueError("Duplicated modification site keys.")
m = m.set_index("key")
X = m[[c for c in m.columns if c.startswith("mutation_rate_")]].T
X.index = X.index.str.replace("mutation_rate_", "", regex=False)
print("External matrix:", X.shape)

missing_final = [f for f in final_features if f not in X.columns]
print(f"Final {a.model} features present in external set: {len(final_features) - len(missing_final)} / {len(final_features)}")
if missing_final:
    raise ValueError(f"Final {a.model} features missing: {missing_final}")

# The fitted pipeline expects the full training column layout. Sites other than the final features are filled
# with NaN (imputed with training medians and then discarded by the feature selection), so they do not affect
# the prediction; only the final features carry the external data.
X = X.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
print("Missing values in the final features:", int(X[final_features].isna().sum().sum()), "/", X[final_features].size)

# ---------- 3. labels ----------
lab = pd.read_csv(a.y, encoding="utf-8-sig")
if "patient_id" not in lab.columns:
    sys.exit(f"{a.y} has no 'patient_id' column (columns: {list(lab.columns)})")
lab = lab.set_index("patient_id")
lab.index = lab.index.astype(str).str.strip()
lab["best_response"] = lab["best_response"].astype(str).str.strip().str.upper()
lab["label"] = lab["best_response"].map({"CR": 1, "PR": 1, "SD": 0, "PD": 0})
common = X.index.intersection(lab.index[lab["label"].notna()], sort=False)
X_use, lab_use = X.loc[common], lab.loc[common]
y = lab_use["label"].astype(int)
print(f"External validation samples: {len(y)} (R {(y == 1).sum()} / NR {(y == 0).sum()})")
if len(y) == 0:
    raise ValueError("No matching labeled samples.")

# ---------- 4. predict with the full locked pipeline ----------
pos = np.flatnonzero(est.classes_ == 1)
if len(pos) != 1:
    raise ValueError("Positive class 1 not found.")
Xv = X_use.to_numpy(dtype=float)
score = est.predict_proba(Xv)[:, pos[0]]                 # P(responder)
pred_default = est.predict(Xv)                           # model default (0.5)

pred_table = pd.DataFrame({"sample": X_use.index, "best_response": lab_use["best_response"].values,
                           "label": y.values, "score_P_R": score, "pred_default": pred_default})
if cutoff is not None:
    pred_youden = (score >= cutoff).astype(int)          # cutoff fixed in the discovery cohort
    pred_table["pred_youden"] = pred_youden
pred_table.to_csv(os.path.join(OUT, f"{a.prefix}_predictions.csv"), index=False)


# ---------- 5. metrics ----------
def thr_metrics(yt, pr):
    tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec == prec and prec + sens else np.nan
    return dict(Accuracy=(tp + tn) / len(yt), BalancedACC=(sens + spec) / 2, Sensitivity=sens,
                Specificity=spec, Precision=prec, F1=f1, TN=tn, FP=fp, FN=fn, TP=tp)


if y.nunique() < 2:
    print("Only one class present; AUC undefined.")
    sys.exit(0)

auc = roc_auc_score(y, score)
rng, boots = np.random.default_rng(a.seed), []
for _ in range(a.n_boot):                                # bootstrap resampling of validation samples
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y.values[i])) == 2:
        boots.append(roc_auc_score(y.values[i], score[i]))
lo, hi = np.percentile(boots, [2.5, 97.5])

rows = [dict(cutoff_type="default (0.5)", cutoff=0.5, **thr_metrics(y, pred_default))]
if cutoff is not None:
    rows.append(dict(cutoff_type="Youden (locked, from discovery cohort)", cutoff=cutoff,
                     **thr_metrics(y, pred_youden)))
met = pd.DataFrame(rows).assign(AUC=auc, AUC_CI_low=lo, AUC_CI_high=hi, n_boot=a.n_boot, n=len(y),
                                n_R=int((y == 1).sum()), n_NR=int((y == 0).sum()))
met.to_csv(os.path.join(OUT, f"{a.prefix}_metrics.csv"), index=False)

fpr, tpr, thr = roc_curve(y, score)
pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
    os.path.join(OUT, f"{a.prefix}_roc.csv"), index=False)

print(f"\n================================\n{a.name} External Validation ({a.model})\n"
      "================================")
print(f"AUC         : {auc:.3f}  (95% CI {lo:.3f} - {hi:.3f}; {a.n_boot} bootstrap resamples)")
print(met.drop(columns=["AUC", "AUC_CI_low", "AUC_CI_high", "n_boot", "n", "n_R", "n_NR"])
      .round(3).to_string(index=False))
print("\nResults saved to:", OUT)
