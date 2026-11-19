"""R vs NR: the primary analysis only - repeated-split evaluation (C), final model (F), permutation test (P),
SHAP explanation of the final models (S).
Stand-alone script (needs fs_transformers.py in the same directory).

Pipeline (fit inside every CV fold, training rows only):
    [read_f prefilter] -> missing/variance filter -> median impute -> Mann-Whitney top-200 -> scale
    -> model-specific selector (5..40 features) -> classifier
    SVM: linear-SVM RFE | LR: L1-LR embedded (top-n by |coef|), plain L2 LR on top
    RF : RFE on RF impurity importance | XGB: RFE on XGB gain importance

Parts (1st argument, any combination, default "CFPS"):
  C  Evaluation. N stratified 7:3 splits (seeds 0..N-1), 5-fold inner CV tunes selector size + classifier.
     PRIMARY RESULT = mean test AUC over the N splits. Writes every test-set score, so the mean ROC can be drawn.
       C_repeated_splits.csv   one row per split x model (AUC, BalancedACC, Sens, Spec, F1, best params ...)
       C_test_predictions.csv  one row per split x model x test sample (label, score, pred; 1 = R, 0 = NR)
       Threshold metrics (ACC/BalancedACC/F1/Sens/Spec/confusion matrix, pred) use a Youden cutoff, not 0.5 / 0:
       per split, the cutoff maximising sens+spec-1 on the out-of-fold inner-CV scores of the training rows
       (column "cutoff"), then applied to the test set. AUC is unaffected.
       C_summary.csv / C_auc_range.csv / C_feature_stability.csv / C_n_features_chosen.csv
       C_roc_mean.csv/.png/.pdf  vertically averaged ROC; band = 2.5-97.5 percentile over splits
       (the splits share samples, so the band is the spread over splits, not a CI of the mean)
  F, P and S run only for the FINAL model(s), set by RVSNR_FINAL_MODELS (default "best" = the model with the
  highest Part C mean test AUC, so Part C must have been run first; or "all", or a list such as "RF,XGB").
  F  Final model. The chosen model's whole procedure (selection + tuning) is refit on ALL samples with
     repeated 5x3 CV. Its expected performance is the Part C estimate, NOT its own CV AUC (that CV
     score is the maximum over the grid, so it is optimistic).
       F_final_<model>.joblib    fitted pipeline: predict_proba / decision_function on a samples x features
                                 frame with the same columns as the training matrix (F_feature_columns.txt)
       F_final_models.csv        best params, n features, CV AUC, Part C mean AUC and rank among the 4 models
       F_final_features.csv      selected features per model: importance/coef, MWU p, mean R / NR
       F_sample_predictions.csv  one row per sample x model:
                                   final_score       the final model on every sample. IN-SAMPLE: the model was
                                                     trained on these samples, so it is optimistic (near-perfect)
                                   PartC_mean_score  mean test-set score over the Part C splits in which the sample
                                                     was held out (~30% of splits) - an honest out-of-sample score
                                 score = P(R) for LR/RF/XGB; SVM gives a decision value (>0 = R), not a probability
  S  SHAP for each saved final model (reads F_final_<model>.joblib, so it reruns without refitting), on all
     samples, in the space the classifier sees (selected features). RF: TreeExplainer, SHAP in P(R);
     XGB: XGBoost's built-in TreeSHAP, log-odds; LR: LinearExplainer, log-odds; SVM: linear kernel LinearExplainer /
     rbf KernelExplainer, decision value. Dot colour = the feature's raw value (median-imputed).
       S_shap_beeswarm_<model>.png/pdf, S_shap_bar_<model>.png/pdf   top RVSNR_SHAP_TOP features (default 20)
       S_shap_values_<model>.csv  samples x features;  S_shap_importance.csv  mean |SHAP| per model x feature
  P  Label-permutation test. The labels are shuffled N_PERM times; each time the SAME procedure is run on
     the first PERM_SPLITS splits. The observed statistic is the mean test AUC over the same first
     PERM_SPLITS splits with the true labels (taken from Part C). p = (1 + #null >= observed) / (1 + N_PERM).
     With "best", the model was picked as the best of four on the same data, which the single-model null
     does not account for, so its p is slightly optimistic. RVSNR_FINAL_MODELS=all adds a row that compares
     the best observed AUC with the max over models in each permutation, which corrects for the pick.
     Resumable: finished permutations are kept in P_null_distribution.csv and skipped on a rerun.
       P_null_distribution.csv / P_permutation_test.csv / P_null_hist.png/.pdf

Feature sets (2nd argument, default "mut"):
    mut            <RVSNR_MUT_FILE>, mutation_rate_* columns
    read           <RVSNR_READ_FILE>, log2(RPM + 1)
    read_f<R>_<F>  read + keep genes with RPM >= R in >= F of the training samples (fit inside each fold)
3rd argument: label file (default 0_ec_set_label.csv); best_response CR/PR = R, SD/PD = NR.

Environment variables:
    RVSNR_DATA_DIR      input directory (default ./data/)
    RVSNR_OUT_BASE      output base (default <data dir>/models_RvsNR/); output dir <feature set>_<label>/
    RVSNR_MUT_FILE      mutation file (default 4_ec_all_mut_filtered.csv)
    RVSNR_READ_FILE     expression file (default 4_ec_all_rpm_filtered.csv)
    RVSNR_N_SPLITS_C    splits in Part C (default 50)
    RVSNR_N_PERM        permutations in Part P (default 100)
    RVSNR_PERM_SPLITS   splits per permutation (default 50; must be <= RVSNR_N_SPLITS_C)
    RVSNR_FINAL_MODELS  models for F, P and S: "best" (default), "all", or e.g. "RF" / "RF,XGB". One permutation
                        costs about PERM_SPLITS / N_SPLITS_C of a Part C run for each model in the list.
    RVSNR_N_JOBS        max parallel model fits (default 36); BLAS threads are pinned to 1 per worker
    RVSNR_SHAP_TOP      features shown in the SHAP plots (default 20)

Usage (needs fs_transformers.py in the same directory):
    python RvsNR_model.py CFPS mut 0_ec_set_label.csv        # C, then F/S/P for the best model
    python RvsNR_model.py S mut 0_ec_set_label.csv           # SHAP only, from the saved final model
    RVSNR_MUT_FILE=4_gc_all_mut_filtered.csv python RvsNR_model.py CFS mut 0_gc_set_label.csv
    RVSNR_N_PERM=200 RVSNR_FINAL_MODELS=RF python RvsNR_model.py P mut 0_ec_set_label.csv
    RVSNR_READ_FILE=4_gc_all_rpm_filtered.csv python RvsNR_model.py CFPS read 0_gc_set_label.csv
Run C first: "best" is read from C's output, and P takes its observed AUCs from it.
"""
import os, sys, time, warnings, tempfile, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
N_JOBS = int(os.environ.get("RVSNR_N_JOBS", 36))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd, joblib
from scipy.stats import mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_curve)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fs_transformers import MissingVarFilter, MWUTopK, L1Select, ExprFilter

warnings.filterwarnings("ignore")
SEED = 42
D = os.path.join(os.environ.get("RVSNR_DATA_DIR", "./data/"), "")      # trailing separator guaranteed
PARTS = sys.argv[1] if len(sys.argv) > 1 else "CFPS"
FEATURE_SET = sys.argv[2] if len(sys.argv) > 2 else "mut"
assert FEATURE_SET in ("mut", "read") or FEATURE_SET.startswith("read_f"), FEATURE_SET
LABEL_FILE = sys.argv[3] if len(sys.argv) > 3 else "0_ec_set_label.csv"
EXPR = tuple(map(float, FEATURE_SET[len("read_f"):].split("_"))) if FEATURE_SET.startswith("read_f") else None
OUT = os.path.join(os.environ.get("RVSNR_OUT_BASE", D + "models_RvsNR/"),
                   f"{FEATURE_SET}_{os.path.splitext(os.path.basename(LABEL_FILE))[0]}", "")   # ends with "/"
N_SPLITS = int(os.environ.get("RVSNR_N_SPLITS_C", 50))
N_PERM = int(os.environ.get("RVSNR_N_PERM", 100))
PERM_SPLITS = int(os.environ.get("RVSNR_PERM_SPLITS", 50))
SHAP_TOP = int(os.environ.get("RVSNR_SHAP_TOP", 20))
COLORS = {"SVM": "#2a78d6", "LR": "#eb6834", "RF": "#1baf7a", "XGB": "#eda100"}
MODELS = list(COLORS)
FINAL_MODELS = os.environ.get("RVSNR_FINAL_MODELS", "best")
assert FINAL_MODELS in ("best", "all") or set(FINAL_MODELS.split(",")) <= set(MODELS), FINAL_MODELS
MWU_K = 200
N_FEAT = [5, 10, 15, 20, 25, 30, 35, 40]

# ---------- data ----------
if FEATURE_SET == "mut":
    DATA_FILE = os.environ.get("RVSNR_MUT_FILE", "4_ec_all_mut_filtered.csv")
    mut = pd.read_csv(D + DATA_FILE).set_index("key")
    X = mut[[c for c in mut.columns if c.startswith("mutation_rate_")]].T
    X.index = X.index.str.replace("mutation_rate_", "", regex=False)
else:
    DATA_FILE = os.environ.get("RVSNR_READ_FILE", "4_ec_all_rpm_filtered.csv")
    X = np.log2(pd.read_csv(D + DATA_FILE).set_index("Geneid").T + 1)   # samples x genes, RPM
clin = pd.read_csv(D + LABEL_FILE, encoding="utf-8-sig").set_index("test_id")
y = clin.best_response.map({"CR": 1, "PR": 1, "SD": 0, "PD": 0}).dropna().astype(int)   # 1 = R, 0 = NR
X = X.loc[y.index]
keys = X.columns.values

# ---------- pipeline ----------
def _prefilter():
    return ExprFilter(min_rpm=EXPR[0], min_frac=EXPR[1]) if EXPR else "passthrough"

def classifier(m, spw):
    """spw = n_NR / n_R in the training rows (XGBoost positive class is R)."""
    return {
        "SVM": lambda: SVC(class_weight="balanced"),
        "LR": lambda: LogisticRegression(class_weight="balanced", max_iter=10000),  # L2; L1 does the selecting
        "RF": lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced",
                                             random_state=SEED, n_jobs=1),
        "XGB": lambda: XGBClassifier(subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                     eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=1,
                                     verbosity=0),
    }[m]()

def selector(m, spw):
    if m == "SVM":
        return RFE(SVC(kernel="linear", C=1, class_weight="balanced"), step=0.1)
    if m == "LR":
        return L1Select()
    if m == "RF":
        return RFE(RandomForestClassifier(n_estimators=200, max_features="sqrt", class_weight="balanced",
                                          random_state=SEED, n_jobs=1), step=0.1)
    return RFE(XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, importance_type="gain", eval_metric="logloss",
                             tree_method="hist", random_state=SEED, n_jobs=1, verbosity=0), step=0.1)

def make_pipe(m, spw, cache):
    return Pipeline([
        ("expr", _prefilter()),
        ("filt", MissingVarFilter()),
        ("imp", SimpleImputer(strategy="median")),
        ("mwu", MWUTopK(k=MWU_K)),
        ("sc", StandardScaler()),
        ("sel", selector(m, spw)),
        ("clf", classifier(m, spw)),
    ], memory=cache)

def grid(m):
    g = {
        "SVM": [{"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1]},
                {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}],
        "LR": [{"sel__C": [0.1, 0.5, 1], "sel__l1_ratio": [1.0], "clf__C": [0.01, 0.1, 1, 10]}],
        "RF": [{"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3, 5]}],
        "XGB": [{"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1],
                 "clf__min_child_weight": [1, 3]}],
    }[m]
    return [{"sel__n_features_to_select": N_FEAT, **d} for d in g]

def fit(m, X_tr, y_tr, cv, cache):
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    gs = GridSearchCV(make_pipe(m, spw, cache), grid(m), scoring="roc_auc", cv=cv, n_jobs=N_JOBS, refit=True)
    return gs.fit(X_tr, y_tr)

def score(est, Xs):
    """P(R) for LR/RF/XGB, the decision value for SVM; only ranks within one split are ever compared."""
    return est.decision_function(Xs) if isinstance(est[-1], SVC) else est.predict_proba(Xs)[:, 1]

def youden_cutoff(est, X_tr, y_tr, cv):
    """Youden-optimal cutoff (max sensitivity + specificity - 1) on the out-of-fold training scores of the selected
    model (same inner CV folds, selection + classifier refit per fold). Test rows never touch it, so the test-set
    Sensitivity/Specificity/BalancedACC/F1 are not optimistic. Same score scale as score(): P(R), or SVM decision value."""
    method = "decision_function" if isinstance(est[-1], SVC) else "predict_proba"
    o = cross_val_predict(clone(est), X_tr, y_tr, cv=cv, method=method, n_jobs=N_JOBS)
    if o.ndim == 2: o = o[:, 1]
    fpr, tpr, thr = roc_curve(y_tr, o)
    return float(thr[np.argmax(tpr - fpr)])

def metrics(ys, s, pr):
    cm = confusion_matrix(ys, pr, labels=[0, 1])
    return dict(AUC=roc_auc_score(ys, s), ACC=accuracy_score(ys, pr), BalancedACC=balanced_accuracy_score(ys, pr),
                F1_R=f1_score(ys, pr), Sensitivity_R=cm[1, 1] / cm[1].sum(), Specificity_NR=cm[0, 0] / cm[0].sum(),
                TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1])

def selected(est):
    e = est.named_steps["expr"]
    k = keys if e == "passthrough" else keys[e.mask_]
    return k[est.named_steps["filt"].mask_][est.named_steps["mwu"].idx_][est.named_steps["sel"].get_support()]

def run_splits(yv, n, models, cache, keep=True, tag="C"):
    """The repeated-split procedure on labels yv (true or permuted). keep=False returns only per-split AUCs."""
    rows, preds, featcount = [], [], {m: {} for m in models}
    yv = pd.Series(np.asarray(yv), index=X.index)
    for r in range(n):
        X_tr, X_te, y_tr, y_te = train_test_split(X, yv, test_size=0.3, stratify=yv, random_state=r)
        cv = StratifiedKFold(5, shuffle=True, random_state=r)
        msg = []
        for m in models:
            gs = fit(m, X_tr.values, y_tr.values, cv, cache)
            est = gs.best_estimator_; s = score(est, X_te.values)
            if not keep:
                rows.append({"split": r, "model": m, "AUC": roc_auc_score(y_te, s)}); continue
            cut = youden_cutoff(est, X_tr.values, y_tr.values, cv)     # from training rows only
            pr, f = (s >= cut).astype(int), selected(est)
            rows.append({"split": r, "model": m, "CV_AUC": gs.best_score_, "n_features": len(f),
                         "cutoff": cut, **metrics(y_te, s, pr),
                         "best_params": str({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})})
            preds.append(pd.DataFrame({"split": r, "model": m, "sample": X_te.index, "label": y_te.values,
                                       "score": s, "pred": pr}))
            for x in f: featcount[m][x] = featcount[m].get(x, 0) + 1
            msg.append(f"{m}={rows[-1]['AUC']:.3f}")
        if keep:
            print(f"[{tag}] split {r+1}/{n} " + " ".join(msg), flush=True)
            pd.DataFrame(rows).to_csv(OUT + "C_repeated_splits.csv", index=False)      # checkpoint
            pd.concat(preds).to_csv(OUT + "C_test_predictions.csv", index=False)
    return pd.DataFrame(rows), (pd.concat(preds) if preds else None), featcount

# ---------- mean ROC ----------
FPR_GRID = np.linspace(0, 1, 101)

def mean_roc(pred, out_prefix=None, title=None):
    """Vertically averaged ROC from a C_test_predictions-style table (split, model, label, score): each split's
    ROC is interpolated onto FPR_GRID and the TPR averaged; band = 2.5-97.5 percentile of TPR over splits.
    Rebuild the figure without refitting:  mean_roc(pd.read_csv(OUT + "C_test_predictions.csv"), OUT + "C_roc_mean")"""
    rows, auc = [], {}
    for m in [m for m in MODELS if m in set(pred.model)]:
        tprs, aucs = [], []
        for _, g in pred[pred.model == m].groupby("split"):
            fpr, tpr, _ = roc_curve(g.label, g.score)
            t = np.interp(FPR_GRID, fpr, tpr); t[0] = 0.0; t[-1] = 1.0
            tprs.append(t); aucs.append(roc_auc_score(g.label, g.score))
        tprs, aucs = np.array(tprs), np.array(aucs)
        auc[m] = (aucs.mean(), *np.percentile(aucs, [2.5, 97.5]), len(aucs))
        rows.append(pd.DataFrame({"model": m, "fpr": FPR_GRID, "tpr_mean": tprs.mean(0), "tpr_sd": tprs.std(0, ddof=1),
                                  "tpr_2.5pct": np.percentile(tprs, 2.5, axis=0),
                                  "tpr_97.5pct": np.percentile(tprs, 97.5, axis=0), "n_splits": len(tprs)}))
    roc = pd.concat(rows, ignore_index=True)
    if out_prefix:
        roc.to_csv(out_prefix + ".csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls="--", c="#bbbbbb", lw=1)
        for m in auc:
            d = roc[roc.model == m]; a, lo, hi, n = auc[m]
            ax.fill_between(d.fpr, d["tpr_2.5pct"], d["tpr_97.5pct"], color=COLORS[m], alpha=0.12, lw=0)
            ax.plot(d.fpr, d.tpr_mean, c=COLORS[m], lw=2, label=f"{m}  AUC {a:.3f} [{lo:.3f}-{hi:.3f}]")
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
        ax.set_title(title or f"Mean test ROC, {n} splits (band: 2.5-97.5%)", fontsize=10)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        fig.tight_layout(); fig.savefig(out_prefix + ".png", dpi=200); fig.savefig(out_prefix + ".pdf"); plt.close(fig)
    return roc, auc

# ---------- Part C ----------
def part_c(cache):
    print(f"[C] {N_SPLITS} splits, model-specific selection", flush=True)
    df, pred, featcount = run_splits(y.values, N_SPLITS, MODELS, cache)
    df.to_csv(OUT + "C_repeated_splits.csv", index=False); pred.to_csv(OUT + "C_test_predictions.csv", index=False)
    summ = df.groupby("model")[["CV_AUC", "AUC", "BalancedACC", "Sensitivity_R", "Specificity_NR",
                                "F1_R", "cutoff", "n_features"]].agg(["mean", "std", "median"]).reindex(MODELS)
    summ.to_csv(OUT + "C_summary.csv")
    q = df.groupby("model")["AUC"].quantile([0.025, 0.975]).unstack().reindex(MODELS)
    q.columns = ["AUC_2.5pct", "AUC_97.5pct"]; q.to_csv(OUT + "C_auc_range.csv")
    df.groupby("model")["n_features"].value_counts().unstack(fill_value=0).to_csv(OUT + "C_n_features_chosen.csv")
    pd.DataFrame([{"model": m, "feature": x, "selected_in_n_splits": c, "selected_frac": c / N_SPLITS}
                  for m in MODELS for x, c in featcount[m].items()]
                 ).sort_values(["model", "selected_in_n_splits"], ascending=[True, False]
                 ).to_csv(OUT + "C_feature_stability.csv", index=False)
    mean_roc(pred, OUT + "C_roc_mean", f"Mean test ROC, {N_SPLITS} splits (band: 2.5-97.5%)")
    return summ, q

# ---------- which model(s) get F / P / S ----------
def partc_auc():
    """Part C mean test AUC per model (empty if Part C has not been run in OUT)."""
    p = OUT + "C_repeated_splits.csv"
    if not os.path.exists(p): return pd.Series(dtype=float)
    c = pd.read_csv(p)
    if c.split.nunique() < N_SPLITS:
        print(f"[!] {p} has {c.split.nunique()} of {N_SPLITS} splits (unfinished Part C?)", flush=True)
    return c.groupby("model").AUC.mean().reindex(MODELS).dropna()

def final_models():
    if FINAL_MODELS == "all": return MODELS
    if FINAL_MODELS != "best": return FINAL_MODELS.split(",")
    a = partc_auc()
    if a.empty:
        raise SystemExit("RVSNR_FINAL_MODELS=best needs Part C output in " + OUT +
                         " - run part C first, or name the model(s), e.g. RVSNR_FINAL_MODELS=RF")
    print(f"[best] Part C mean test AUC: " + ", ".join(f"{m} {v:.3f}" for m, v in a.items())
          + f" -> final model {a.idxmax()}", flush=True)
    return [a.idxmax()]

# ---------- Part F ----------
def importance(est, m):
    c = est.named_steps["clf"]
    if m == "LR" or (m == "SVM" and c.kernel == "linear"): return c.coef_.ravel(), "coef (scaled)"
    if m in ("RF", "XGB"): return c.feature_importances_, "importance"
    return np.full(int(est.named_steps["sel"].get_support().sum()), np.nan), "n/a (rbf)"

def part_f(models):
    c_auc = partc_auc(); rank = c_auc.rank(ascending=False, method="min")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows, feats = [], []
    with open(OUT + "F_feature_columns.txt", "w") as fh: fh.write("\n".join(map(str, keys)) + "\n")
    for m in models:
        gs = fit(m, X.values, y.values, cv, None)        # no cache: the saved pipeline must not point to a tmp dir
        est = gs.best_estimator_; f = selected(est); imp, lab = importance(est, m)
        joblib.dump(est, OUT + f"F_final_{m}.joblib")
        cut = youden_cutoff(est, X.values, y.values, StratifiedKFold(5, shuffle=True, random_state=SEED))
        with open(OUT + f"F_final_{m}_cutoff.txt", "w") as fh: fh.write(repr(cut) + "\n")
        rows.append({"model": m, "n_features": len(f), "Youden_cutoff": cut, "CV_AUC_all_samples (optimistic)": gs.best_score_,
                     "PartC_mean_test_AUC": c_auc.get(m, np.nan), "PartC_rank_of_4": rank.get(m, np.nan),
                     "best_params": str({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})})
        mw = est.named_steps["mwu"]
        feats.append(pd.DataFrame({"model": m, "feature": f, "importance": imp, "importance_type": lab,
                                   "MWU_p_all": mw.p_[mw.idx_][est.named_steps["sel"].get_support()],
                                   "mean_R": X.loc[y == 1, f].mean().values, "mean_NR": X.loc[y == 0, f].mean().values}))
        print(f"[F] {m}: {len(f)} features, CV AUC on all samples {gs.best_score_:.3f} (optimistic)", flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT + "F_final_models.csv", index=False)
    fd = pd.concat(feats)
    if os.path.exists(OUT + "C_feature_stability.csv"):     # how often each final feature was picked in Part C
        st = pd.read_csv(OUT + "C_feature_stability.csv")[["model", "feature", "selected_frac"]]
        fd = fd.merge(st.rename(columns={"selected_frac": "PartC_selected_frac"}), on=["model", "feature"], how="left")
    fd.sort_values(["model", "MWU_p_all"]).to_csv(OUT + "F_final_features.csv", index=False)
    sample_predictions(models)
    return res

def sample_predictions(models):
    """Per-sample scores: the final model in-sample, and the Part C out-of-sample mean (when Part C exists)."""
    rows = []
    for m in models:
        if not os.path.exists(OUT + f"F_final_{m}.joblib"):
            print(f"[F] {OUT}F_final_{m}.joblib not found - run Part F for {m} first", flush=True); continue
        est = joblib.load(OUT + f"F_final_{m}.joblib")
        cp = OUT + f"F_final_{m}_cutoff.txt"
        if not os.path.exists(cp):
            print(f"[F] {cp} not found - rerun Part F for {m} (final_pred would fall back to the default cutoff)", flush=True)
        s_all = score(est, X.values)
        cut = float(open(cp).read()) if os.path.exists(cp) else (0.0 if m == "SVM" else 0.5)
        rows.append(pd.DataFrame({"sample": X.index, "label": y.values, "model": m,
                                  "final_score": s_all, "final_pred": (s_all >= cut).astype(int), "Youden_cutoff": cut,
                                  "score_type": "decision value" if m == "SVM" else "P(R)"}))
    if not rows: return None
    out = pd.concat(rows, ignore_index=True)
    if os.path.exists(OUT + "C_test_predictions.csv"):
        c = pd.read_csv(OUT + "C_test_predictions.csv")
        c = c.groupby(["model", "sample"]).agg(PartC_mean_score=("score", "mean"), PartC_sd_score=("score", "std"),
                                               PartC_frac_pred_R=("pred", "mean"), PartC_n_test=("score", "size"))
        out = out.merge(c.reset_index(), on=["model", "sample"], how="left")
    out.to_csv(OUT + "F_sample_predictions.csv", index=False)
    return out

# ---------- Part S ----------
def part_s(models):
    import shap
    imp_rows = []
    for m in models:
        path = OUT + f"F_final_{m}.joblib"
        if not os.path.exists(path):
            print(f"[S] {path} not found - run Part F first", flush=True); continue
        est = joblib.load(path); clf = est[-1]
        Z = est[:-1].transform(X.values)                       # what the classifier sees: selected, scaled
        f = selected(est)
        raw = X[f].astype(float); raw = raw.fillna(raw.median())   # raw values, for the dot colour only
        if m == "RF":
            e = shap.TreeExplainer(clf)(Z); unit = "SHAP (P(R))"
            v, b = e.values[..., 1], e.base_values[..., 1]      # both classes are returned; keep R
        elif m == "XGB":    # XGBoost's own exact TreeSHAP; shap's TreeExplainer misreads xgboost>=3 base_score
            import xgboost
            c_ = clf.get_booster().predict(xgboost.DMatrix(Z), pred_contribs=True)
            v, b = c_[:, :-1], c_[:, -1]; unit = "SHAP (log-odds of R)"
        elif m == "LR" or clf.kernel == "linear":
            e = shap.LinearExplainer((clf.coef_.ravel(), clf.intercept_[0]), Z)(Z)
            v, b = e.values, e.base_values
            unit = "SHAP (log-odds of R)" if m == "LR" else "SHAP (decision value)"
        else:                                                  # rbf SVM: model-agnostic, slower
            ke = shap.KernelExplainer(clf.decision_function, shap.kmeans(Z, min(10, len(Z))))
            v = np.asarray(ke.shap_values(Z, nsamples=min(2 * Z.shape[1] + 2048, 4096), silent=True))
            b = np.full(len(Z), ke.expected_value); unit = "SHAP (decision value)"
        ex = shap.Explanation(values=v, base_values=b, data=raw.values, feature_names=list(f))
        pd.DataFrame(v, index=X.index, columns=f).to_csv(OUT + f"S_shap_values_{m}.csv")
        imp_rows.append(pd.DataFrame({"model": m, "feature": f, "mean_abs_shap": np.abs(v).mean(0),
                                      "mean_shap_R": v[y.values == 1].mean(0), "mean_shap_NR": v[y.values == 0].mean(0),
                                      "unit": unit}).sort_values("mean_abs_shap", ascending=False))
        top = min(SHAP_TOP, len(f))
        for kind, plot in (("beeswarm", shap.plots.beeswarm), ("bar", shap.plots.bar)):
            plt.figure()
            plot(ex, max_display=top, show=False)
            fig = plt.gcf(); fig.set_size_inches(9, 0.35 * top + 1.8)
            fig.axes[0].set_xlabel(unit + (" ; >0 pushes towards R" if kind == "beeswarm" else " - mean |value|"))
            fig.axes[0].set_title(f"{m} final model: top {top} of {len(f)} features (n = {len(y)})", fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT + f"S_shap_{kind}_{m}.png", dpi=200, bbox_inches="tight")
            fig.savefig(OUT + f"S_shap_{kind}_{m}.pdf", bbox_inches="tight"); plt.close("all")
        print(f"[S] {m}: {len(f)} features, {unit}", flush=True)
    if imp_rows:
        pd.concat(imp_rows).to_csv(OUT + "S_shap_importance.csv", index=False)

# ---------- Part P ----------
def part_p(models, cache):
    k = PERM_SPLITS
    cpath = OUT + "C_repeated_splits.csv"
    c = pd.read_csv(cpath) if os.path.exists(cpath) else None
    if c is None or c.split.nunique() < k or not set(models) <= set(c.model):
        print(f"[P] true-label AUCs for the first {k} splits not found in Part C output - computing them", flush=True)
        c, _, _ = run_splits(y.values, k, models, cache, keep=False)
    obs = c[(c.split < k) & c.model.isin(models)].groupby("model").AUC.mean()
    npath = OUT + "P_null_distribution.csv"
    null = pd.read_csv(npath) if os.path.exists(npath) else pd.DataFrame(columns=["perm", "model", "mean_AUC", "n_splits"])
    if len(null) and (set(null.n_splits) != {k}):
        raise SystemExit(f"{npath} was made with n_splits {set(null.n_splits)}, not {k}; move it away to restart")
    done = {(int(p), m) for p, m in zip(null.perm, null.model)}
    print(f"[P] {N_PERM} permutations x {k} splits, models {models}; {len({p for p, _ in done})} already done",
          flush=True)
    for i in range(N_PERM):
        todo = [m for m in models if (i, m) not in done]
        if not todo: continue
        t0 = time.time()
        yp = np.random.default_rng(1000 + i).permutation(y.values)        # perm i always uses the same shuffle
        a, _, _ = run_splits(yp, k, todo, cache, keep=False)
        add = a.groupby("model").AUC.mean().rename("mean_AUC").reset_index().assign(perm=i, n_splits=k)
        null = pd.concat([null, add[["perm", "model", "mean_AUC", "n_splits"]]], ignore_index=True)
        null.to_csv(npath, index=False)                                   # checkpoint after every permutation
        print(f"[P] perm {i+1}/{N_PERM} " + " ".join(f"{r.model}={r.mean_AUC:.3f}" for r in add.itertuples())
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    null["perm"] = null.perm.astype(int); null = null[null.perm < N_PERM]
    rows = []
    for m in models:
        z = null.loc[null.model == m, "mean_AUC"].values
        rows.append({"model": m, "observed_mean_AUC": obs[m], "n_splits": k, "n_perm": len(z),
                     "null_mean": z.mean(), "null_95pct": np.percentile(z, 95), "null_max": z.max(),
                     "p_value": (1 + (z >= obs[m]).sum()) / (1 + len(z))})
    if len(models) > 1:   # best observed vs max over models: corrects for picking the best of several
        zmax = null.pivot(index="perm", columns="model", values="mean_AUC")[models].max(axis=1).values
        rows.append({"model": f"best model ({obs.idxmax()}) vs max over models", "observed_mean_AUC": obs.max(),
                     "n_splits": k, "n_perm": len(zmax), "null_mean": zmax.mean(),
                     "null_95pct": np.percentile(zmax, 95), "null_max": zmax.max(),
                     "p_value": (1 + (zmax >= obs.max()).sum()) / (1 + len(zmax))})
    res = pd.DataFrame(rows); res.to_csv(OUT + "P_permutation_test.csv", index=False)

    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        z = null.loc[null.model == m, "mean_AUC"].values; p = res.set_index("model").loc[m, "p_value"]
        ax.hist(z, bins=20, color=COLORS[m], alpha=0.5, edgecolor="white")
        ax.axvline(obs[m], color="#333333", lw=1.5)
        ax.set_title(f"{m}: observed {obs[m]:.3f}, p = {p:.3g}", fontsize=9)
        ax.set_xlabel(f"mean test AUC ({k} splits)")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axes[0][0].set_ylabel(f"permutations (n = {int(res.n_perm.iloc[0])})")
    fig.tight_layout(); fig.savefig(OUT + "P_null_hist.png", dpi=200); fig.savefig(OUT + "P_null_hist.pdf"); plt.close(fig)
    return res

def write_prespec():
    with open(OUT + f"prespecification_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt", "w") as fh:
        fh.write(f"""Pre-specification (written at run start, before any result of this run exists)
parts              : {PARTS}
data               : {DATA_FILE} (feature set {FEATURE_SET}), labels {LABEL_FILE}
samples            : {len(y)} ({int(y.sum())} R / {int((y == 0).sum())} NR), features {X.shape[1]}
Part C             : {N_SPLITS} stratified 7:3 splits (seeds 0..{N_SPLITS - 1}), 5-fold inner CV
                     MWU top-{MWU_K}, features per model {N_FEAT}, model-specific selection
PRIMARY RESULT     : mean test AUC over the {N_SPLITS} Part C splits
final model(s)     : RVSNR_FINAL_MODELS = {FINAL_MODELS}  ("best" = highest Part C mean test AUC; used by F, S, P)
Part F             : final model(s) refit on all samples (5x3 CV tuning); reported performance = Part C estimate
Part P             : {N_PERM} label permutations x {PERM_SPLITS} splits for the final model(s);
                     statistic = mean test AUC over splits 0..{PERM_SPLITS - 1}; p = (1 + #null >= obs) / (1 + n)
""")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    assert PERM_SPLITS <= N_SPLITS or "C" not in PARTS, "RVSNR_PERM_SPLITS must be <= RVSNR_N_SPLITS_C"
    print(f"{DATA_FILE} + {LABEL_FILE} -> {OUT}", flush=True)
    print(f"samples {len(y)} (R {y.sum()} / NR {(y == 0).sum()}), features {X.shape[1]}, parts {PARTS}", flush=True)
    write_prespec()
    pd.set_option("display.width", 250)
    with tempfile.TemporaryDirectory() as cache:
        if "C" in PARTS:
            summ, q = part_c(cache)
            print(f"\n=== PRIMARY RESULT: {N_SPLITS} splits ===\n" + summ.round(3).to_string() + "\n" + q.round(3).to_string())
        if any(p_ in PARTS for p_ in "FSP"):
            models = final_models()
            if "F" in PARTS:
                print("\n=== final model(s), all samples ===\n" + part_f(models).round(3).to_string(index=False))
            if "S" in PARTS:
                if "F" not in PARTS: sample_predictions(models)   # refresh per-sample scores from the saved model
                part_s(models)
            if "P" in PARTS:                                    # last: by far the longest part
                print("\n=== permutation test ===\n" + part_p(models, cache).round(4).to_string(index=False))
