# cfRNA modification models for response to first-line chemoimmunotherapy

Code for training and externally validating plasma cell-free RNA (cfRNA) modification-based classifiers of
response to first-line chemoimmunotherapy in esophageal (EC) and gastric (GC) cancer, from LIME-seq data.

> Patient-level data are **not** included in this repository. Raw sequencing data are available under controlled
> access at GSA-Human (BioProject PRJCA080607). `example_data/` contains synthetic files in the correct
> format for testing the code only.

## Contents

| File | Purpose |
|---|---|
| `src/RvsNR_model.py` | Discovery-cohort analysis: repeated-split evaluation (Part C), final model (F), SHAP (S), label-permutation test (P) |
| `src/fs_transformers.py` | Feature-selection transformers used by the pipeline (must sit next to the scripts; needed to load the saved model) |
| `src/external_validate.py` | Applies the locked final model and discovery-cohort cutoff to an external cohort |
| `src/make_example_data.py` | Writes synthetic example inputs |
| `requirements.txt` | Python dependencies |

## Installation

Developed and run with Python 3.12.2; the exact package versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The saved final model (`F_final_<model>.joblib`) is a pickled scikit-learn pipeline, so it should be loaded with
the same scikit-learn and XGBoost versions that were used to train it.

## Input formats

* **Modification-rate matrix** (`RVSNR_MUT_FILE`, `--x`): CSV with a `key` column (modification site) and one
  `mutation_rate_<sample>` column per sample.
* **Label file** (3rd argument, `--y`): CSV with `test_id` and `best_response`. `CR`/`PR` = responder (1),
  `SD`/`PD` = non-responder (0); RECIST v1.1.
* Optional abundance input (`read` feature set): CSV with a `Geneid` column and one RPM column per sample.

## Pipeline (fit inside every cross-validation fold, training rows only)

missing/variance filter → median imputation → Mann-Whitney top-200 → scaling → model-specific selector
(5–40 features) → classifier. Four classifiers (logistic regression, linear/RBF SVM, random forest, XGBoost) are
tuned by grid search with 5-fold cross-validation (AUC). Part C repeats a stratified 70/30 split 50 times
(seeds 0–49); the primary result is the mean test AUC over the splits. Threshold metrics use a Youden cutoff
derived from out-of-fold scores of the training rows only.

## Usage

Run Part C first: `best` is read from its output, and the permutation test takes its observed AUCs from it.

```bash
cd src
# 1. Discovery cohort: repeated splits (C), final model (F), SHAP (S), permutation test (P)
RVSNR_DATA_DIR=../data RVSNR_MUT_FILE=ec_all_mut_filtered.csv \
    python RvsNR_model.py CFPS mut ec_set_label.csv

# 2. External validation with the locked model and cutoff
python external_validate.py --x ../data/ec_ext_mut_all.csv --y ../data/ec_ext_set_label.csv \
    --out ../data/models_RvsNR/mut_ec_set_label/ --prefix ec_external --name EC
```

Run the same commands with the GC files for gastric cancer. Parts can be combined freely (for example `S` alone
re-runs SHAP from the saved final model).

### Settings (environment variables of `RvsNR_model.py`)

| Variable | Default | Meaning |
|---|---|---|
| `RVSNR_DATA_DIR` | `./data/` | input directory |
| `RVSNR_OUT_BASE` | `<data dir>/models_RvsNR/` | output base; results go to `<feature set>_<label>/` |
| `RVSNR_MUT_FILE` / `RVSNR_READ_FILE` | see script | modification / abundance input file |
| `RVSNR_N_SPLITS_C` | 50 | repeated splits in Part C |
| `RVSNR_N_PERM` | 100 | label permutations in Part P |
| `RVSNR_PERM_SPLITS` | 50 | splits per permutation (must be ≤ `RVSNR_N_SPLITS_C`) |
| `RVSNR_FINAL_MODELS` | `best` | models for F, P and S: `best`, `all`, or e.g. `RF,XGB` |
| `RVSNR_N_JOBS` | 36 | maximum parallel model fits |
| `RVSNR_SHAP_TOP` | 20 | features shown in SHAP plots |

With the defaults, Part P runs 100 permutations × 50 splits, i.e. roughly 100 times the cost of Part C for each
model listed in `RVSNR_FINAL_MODELS`. It is resumable: finished permutations are kept in
`P_null_distribution.csv`.

External validation reports the AUC with a 95% CI from 1,000 bootstrap resamples of the validation samples
(`--n-boot`, `--seed`), and threshold metrics at the default 0.5 cutoff and at the cutoff fixed in the discovery
cohort. Nothing is re-fitted or re-calibrated on the external cohort.

## Outputs of `RvsNR_model.py`

`C_repeated_splits.csv`, `C_test_predictions.csv`, `C_summary.csv`, `C_roc_mean.*` (Part C);
`F_final_<model>.joblib`, `F_final_<model>_cutoff.txt`, `F_feature_columns.txt`, `F_final_features.csv`,
`F_sample_predictions.csv` (Part F); `S_shap_*` (Part S); `P_permutation_test.csv`, `P_null_*` (Part P).
`F_sample_predictions.csv` contains an in-sample `final_score` (optimistic by construction) and `PartC_mean_score`,
the mean out-of-sample score over the splits in which the sample was held out.

## Quick test on synthetic data

```bash
cd src
python make_example_data.py                      # writes ../example_data/
RVSNR_DATA_DIR=../example_data RVSNR_MUT_FILE=ec_all_mut_filtered.csv RVSNR_N_SPLITS_C=2 \
RVSNR_N_PERM=2 RVSNR_PERM_SPLITS=2 RVSNR_FINAL_MODELS=RF RVSNR_N_JOBS=4 \
    python RvsNR_model.py CFPS mut ec_set_label.csv
```

The full grid search is computationally heavy even on small inputs; expect this test to take several minutes.

## License and citation

Add a LICENSE file of your choice and the citation of the associated article here.
