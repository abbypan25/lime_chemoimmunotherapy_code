"""Custom feature-selection transformers (separate module so joblib can pickle them)."""
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin


class MissingVarFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing=0.2, min_var=1e-6):
        self.max_missing, self.min_var = max_missing, min_var
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mask_ = (np.isnan(X).mean(0) <= self.max_missing) & (np.nanvar(X, 0) > self.min_var)
        return self
    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]

class MWUTopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=200):
        self.k = k
    def fit(self, X, y):
        y = np.asarray(y)
        self.p_ = np.nan_to_num(mannwhitneyu(X[y == 1], X[y == 0], axis=0).pvalue, nan=1.0)
        self.idx_ = np.argsort(self.p_)[: min(self.k, X.shape[1])]
        return self
    def transform(self, X):
        return X[:, self.idx_]

from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression

class L1Select(SelectorMixin, BaseEstimator):
    """Embedded selection for LR: fit an L1/ElasticNet LR, keep non-zero coefficients,
    capped at the top n_features_to_select by |coef|."""
    def __init__(self, n_features_to_select=10, C=0.5, l1_ratio=1.0):
        self.n_features_to_select, self.C, self.l1_ratio = n_features_to_select, C, l1_ratio
    def fit(self, X, y):
        lr = LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=self.l1_ratio, C=self.C,
                                class_weight="balanced", max_iter=10000).fit(X, y)
        w = np.abs(lr.coef_.ravel())
        order = np.argsort(-w, kind="stable")
        keep = order[w[order] > 0][: self.n_features_to_select]
        if len(keep) == 0:  # everything shrunk to 0: fall back to the top Mann-Whitney feature (column 0)
            keep = np.array([0])
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[keep] = True
        return self
    def _get_support_mask(self):
        return self.support_

class ExprFilter(BaseEstimator, TransformerMixin):
    """Low-expression filter for log2(RPM+1) data: keep genes with RPM >= min_rpm
    in at least min_frac of the (training) samples."""
    def __init__(self, min_rpm=5, min_frac=0.2):
        self.min_rpm, self.min_frac = min_rpm, min_frac
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mask_ = (X >= np.log2(self.min_rpm + 1)).mean(0) >= self.min_frac
        return self
    def transform(self, X):
        return np.asarray(X, float)[:, self.mask_]

