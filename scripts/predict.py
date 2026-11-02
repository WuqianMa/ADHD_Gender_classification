"""Generate a competition CSV from saved models; this script never trains."""

from pathlib import Path

from .common import ID_COLUMN, TARGET_COLUMNS, data_parser, positive_int


def predict(args):
    import joblib
    import pandas as pd

    from .data import load_table, read_table

    quantitative = load_table(args.data_dir, "test", "QUANTITATIVE")
    connectome = load_table(args.data_dir, "test", "CONNECTOME")
    if set(quantitative[ID_COLUMN]) != set(connectome[ID_COLUMN]):
        raise ValueError("Test quantitative/connectome participant IDs do not match")
    result = quantitative[[ID_COLUMN]].copy()
    adhd = joblib.load(args.model_dir / "adhd_model.joblib")
    result["ADHD_Outcome"] = adhd["pipeline"].predict(quantitative[adhd["features"]])
    if args.sex_model == "gnn":
        from .gnn import predict_gnn
        sex_predictions = predict_gnn(connectome, args.model_dir / "sex_gnn.pt",
                                      args.device, args.batch_size)
    else:
        sex = joblib.load(args.model_dir / "sex_model.joblib")
        sex_predictions = sex["pipeline"].predict(connectome[sex["features"]])
    sex_frame = connectome[[ID_COLUMN]].copy()
    sex_frame["Sex_F"] = sex_predictions
    result = result.merge(sex_frame, on=ID_COLUMN, validate="one_to_one")
    if args.sample_submission:
        sample = read_table(args.sample_submission)
        if set(sample[ID_COLUMN]) != set(result[ID_COLUMN]):
            raise ValueError("Sample submission IDs differ from the test IDs")
        result = sample[[ID_COLUMN]].merge(result, on=ID_COLUMN, how="left", validate="one_to_one")
    if result[TARGET_COLUMNS].isna().any().any():
        raise ValueError("Missing predictions; refusing to fill them with fake zero labels")
    result[TARGET_COLUMNS] = result[TARGET_COLUMNS].astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result[[ID_COLUMN, *TARGET_COLUMNS]].to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--sex-model", choices=["ml", "gnn"], default="ml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
