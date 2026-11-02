"""CPU estimators adapted from notebook cells 161, 268, 271 and 273."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .common import SEED


class SignificantCorrelationSelector(TransformerMixin, BaseEstimator):
    """Select p < 0.05 features by |Pearson r| using only fit-time labels.

    Keeping selection inside the pipeline prevents validation/CV label leakage.
    """

    def __init__(self, top_n=463, alpha=0.05):
        self.top_n = top_n
        self.alpha = alpha

    def fit(self, X, y):
        if self.top_n < 1 or not 0 < self.alpha <= 1:
            raise ValueError("top_n must be positive and alpha must be in (0, 1]")
        y = np.asarray(y)
        records = []
        for column in X.columns:
            x = X[column].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            r, p = 0.0, 1.0
            if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
                r, p = pearsonr(x[mask], y[mask])
            records.append((column, abs(float(r)), float(p)))
        self.correlations_ = pd.DataFrame(records, columns=["feature", "abs_r", "p_value"])
        significant = self.correlations_.query("p_value < @self.alpha").sort_values(
            "abs_r", ascending=False, kind="stable")
        self.features_ = significant["feature"].head(self.top_n).tolist()
        if not self.features_:
            raise ValueError("No significant connectome features found on the training split")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "features_")
        return X.loc[:, self.features_]


def adhd_pipeline(model="logistic", seed=SEED):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scaler", StandardScaler())]
    if model == "svc":
        steps.extend([("pca", PCA(n_components=0.95, random_state=seed)),
                      ("clf", SVC(random_state=seed))])
    else:
        steps.append(("clf", LogisticRegression(C=0.01, penalty="l2", solver="lbfgs",
                                                max_iter=1000, random_state=seed)))
    return Pipeline(steps)


def sex_pipeline(top_n=463):
    return Pipeline([
        ("selector", SignificantCorrelationSelector(top_n=top_n)),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", GaussianNB()),
    ])
