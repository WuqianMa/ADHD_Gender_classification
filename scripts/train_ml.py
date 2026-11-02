"""Train the notebook's final classical models on CPU without running EDA."""

import json
from pathlib import Path

from .common import ADHD_FEATURES, ID_COLUMN, TARGET_COLUMNS, data_parser, positive_int


def train(args):
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    from .data import attach_targets, load_table, load_targets
    from .models import adhd_pipeline, sex_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_targets(args.data_dir)
    train_ids, validation_ids = train_test_split(
        labels[ID_COLUMN], test_size=0.2, random_state=args.seed,
        stratify=labels[TARGET_COLUMNS].astype(str).agg("_".join, axis=1))
    metrics = {}
    for target in (["adhd", "sex"] if args.target == "both" else [args.target]):
        kind = "QUANTITATIVE" if target == "adhd" else "CONNECTOME"
        frame = attach_targets(load_table(args.data_dir, "train", kind), labels)
        frame = frame.set_index(ID_COLUMN)
        column = "ADHD_Outcome" if target == "adhd" else "Sex_F"
        if target == "adhd":
            features = (ADHD_FEATURES if args.adhd_model == "logistic" else
                        [c for c in frame if c not in TARGET_COLUMNS + ["MRI_Track_Age_at_Scan"]])
            pipeline = adhd_pipeline(args.adhd_model, args.seed)
        else:
            features = [c for c in frame if c not in TARGET_COLUMNS]
            pipeline = sex_pipeline(args.top_n)
        X, y = frame[features], frame[column]
        # Every fitted preprocessing step sees training rows only during evaluation.
        validation_model = clone(pipeline).fit(X.loc[train_ids], y.loc[train_ids])
        predictions = validation_model.predict(X.loc[validation_ids])
        metrics[target] = {
            "validation_accuracy": float(accuracy_score(y.loc[validation_ids], predictions)),
            "validation_f1": float(f1_score(y.loc[validation_ids], predictions, zero_division=0)),
            "train_rows": len(train_ids), "validation_rows": len(validation_ids),
        }
        # Refit a fresh pipeline on all training rows for competition inference.
        pipeline.fit(X, y)
        artifact = {"pipeline": pipeline, "features": features, "target": column,
                    "seed": args.seed, "metrics": metrics[target]}
        destination = args.output_dir / f"{target}_model.joblib"
        joblib.dump(artifact, destination)
        if target == "sex":
            pipeline["selector"].correlations_.to_csv(
                args.output_dir / "sex_feature_correlations.csv", index=False)
        print(f"{target}: {metrics[target]} -> {destination}")
    # Separate files allow --target adhd and --target sex to run independently.
    for target, result in metrics.items():
        (args.output_dir / f"{target}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--target", choices=["both", "adhd", "sex"], default="both")
    parser.add_argument("--adhd-model", choices=["logistic", "svc"], default="logistic")
    parser.add_argument("--top-n", type=positive_int, default=463)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
