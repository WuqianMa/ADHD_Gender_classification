"""Optional CPU model comparisons and tuning, separated from EDA and final fits.

Ports the quantitative comparisons/strategy search (cells 138–156), and the
connectome feature-count search (cells 254–265). Expensive searches are opt-in.
All preprocessing and supervised feature selection are fitted inside each fold.
"""

import itertools
from pathlib import Path

from .common import ADHD_FEATURES, ID_COLUMN, TARGET_COLUMNS, data_parser, positive_int


def classifiers(seed, boosting=False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForest": RandomForestClassifier(random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "KNeighbors": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "SVC": SVC(random_state=seed),
    }
    if boosting:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        models.update({
            "CatBoost": CatBoostClassifier(verbose=0, random_seed=seed, task_type="CPU"),
            "LightGBM": LGBMClassifier(verbosity=-1, random_state=seed, device_type="cpu"),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=seed, device="cpu"),
        })
    return models


def quantitative(args):
    import pandas as pd
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from .data import attach_targets, load_table, load_targets
    from .models import adhd_pipeline

    frame = attach_targets(load_table(args.data_dir, "train", "QUANTITATIVE"), load_targets(args.data_dir))
    frame = frame.drop(columns=[ID_COLUMN, "MRI_Track_Age_at_Scan"], errors="ignore")
    y = frame["ADHD_Outcome"]
    X = frame.drop(columns=TARGET_COLUMNS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       random_state=args.seed, stratify=y)
    folds = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    if args.task == "tune":
        # Same logistic grid as cell 146; preprocessing stays inside the search.
        grid = [
            {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1, 10, 100], "clf__solver": ["lbfgs", "saga"]},
            {"clf__penalty": [None], "clf__solver": ["lbfgs", "saga"]},
            {"clf__penalty": ["l1"], "clf__C": [0.01, 0.1, 1, 10, 100], "clf__solver": ["saga"]},
            {"clf__penalty": ["elasticnet"], "clf__C": [0.01, 0.1, 1, 10, 100],
             "clf__l1_ratio": [0.25, 0.5, 0.75], "clf__solver": ["saga"]},
        ]
        search = GridSearchCV(adhd_pipeline(seed=args.seed), grid, cv=folds, scoring="f1", n_jobs=args.jobs)
        search.fit(X_train[ADHD_FEATURES], y_train)
        pd.DataFrame(search.cv_results_).to_csv(args.output_dir / "logistic_grid.csv", index=False)
        predictions = search.predict(X_test[ADHD_FEATURES])
        return pd.DataFrame([{"best_params": search.best_params_, "cv_f1": search.best_score_,
                              "holdout_f1": f1_score(y_test, predictions, zero_division=0)}])

    models = classifiers(args.seed, args.boosting)
    if args.task == "strategies":
        sets = {"strong": ADHD_FEATURES[:5], "strong+moderate": ADHD_FEATURES[:7],
                "all_related": ADHD_FEATURES, "complete": list(X)}
        models = {name: models[name] for name in ["LogisticRegression", "SVC"]}
        options = itertools.product(["drop", "median"], sets, [False, True], [False, True])
    else:
        sets = {"complete": list(X)}
        options = [("median", "complete", True, True)]
    records = []
    for missing, feature_set, normalize, pca in options:
        features = sets[feature_set]
        # Do not drop rows based on held-out targets or impute before splitting.
        train_mask = X_train[features].notna().all(axis=1) if missing == "drop" else pd.Series(True, index=X_train.index)
        test_mask = X_test[features].notna().all(axis=1) if missing == "drop" else pd.Series(True, index=X_test.index)
        for name, classifier in models.items():
            steps = []
            if missing == "median":
                steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
            if normalize:
                steps.append(("scaler", StandardScaler()))
            if pca:
                steps.append(("pca", PCA(n_components=0.95, random_state=args.seed)))
            steps.append(("clf", clone(classifier)))
            pipeline = Pipeline(steps)
            scores = cross_validate(pipeline, X_train.loc[train_mask, features], y_train.loc[train_mask],
                                    cv=folds, scoring=["accuracy", "f1"], n_jobs=args.jobs, error_score="raise")
            pipeline.fit(X_train.loc[train_mask, features], y_train.loc[train_mask])
            predicted = pipeline.predict(X_test.loc[test_mask, features])
            records.append({"strategy": missing, "features": feature_set, "model": name,
                            "normalized": normalize, "pca": pca,
                            "cv_accuracy": scores["test_accuracy"].mean(), "cv_f1": scores["test_f1"].mean(),
                            "holdout_accuracy": accuracy_score(y_test.loc[test_mask], predicted),
                            "holdout_f1": f1_score(y_test.loc[test_mask], predicted, zero_division=0),
                            "holdout_rows": int(test_mask.sum())})
            print(records[-1])
    return pd.DataFrame(records)


def connectome(args):
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

    from .data import attach_targets, load_table, load_targets
    from .models import sex_pipeline

    frame = attach_targets(load_table(args.data_dir, "train", "CONNECTOME"), load_targets(args.data_dir))
    X, y = frame.drop(columns=[ID_COLUMN, *TARGET_COLUMNS]), frame["Sex_F"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    folds = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    records = []
    # Select the count using CV on training data, instead of maximizing test F1.
    for count in range(args.min_features, min(args.max_features, X.shape[1]) + 1, args.feature_step):
        scores = cross_validate(sex_pipeline(count), X_train, y_train, cv=folds, scoring="f1",
                                n_jobs=args.jobs, error_score="raise")
        records.append({"top_n": count, "cv_f1": scores["test_score"].mean()})
    if not records:
        raise ValueError("Feature count range is empty for this dataset")
    result = pd.DataFrame(records)
    best_n = int(result.loc[result["cv_f1"].idxmax(), "top_n"])
    pipeline = sex_pipeline(best_n).fit(X_train, y_train)
    score = f1_score(y_test, pipeline.predict(X_test), zero_division=0)
    print(f"CV selected top-{best_n}; untouched holdout F1: {score:.4f}")
    pd.DataFrame([{"top_n": best_n, "holdout_f1": score}]).to_csv(
        args.output_dir / "connectome_best.csv", index=False)
    return result


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--task", choices=["compare", "tune", "strategies", "connectome"], default="compare")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--cv", type=positive_int, default=5)
    parser.add_argument("--jobs", type=positive_int, default=1)
    parser.add_argument("--boosting", action="store_true", help="Include optional CatBoost, LightGBM and XGBoost models.")
    parser.add_argument("--min-features", type=positive_int, default=20)
    parser.add_argument("--max-features", type=positive_int, default=700)
    parser.add_argument("--feature-step", type=positive_int, default=1)
    args = parser.parse_args()
    if args.cv < 2:
        parser.error("--cv must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = connectome(args) if args.task == "connectome" else quantitative(args)
    destination = args.output_dir / f"{args.task}.csv"
    result.to_csv(destination, index=False)
    print(f"Saved experiment results to {destination}")


if __name__ == "__main__":
    main()
