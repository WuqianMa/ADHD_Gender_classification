"""Competition file loading shared by EDA, training, and prediction.

Supports the notebook's Excel/CSV release and the competition's TSV release.
All joins use participant IDs rather than assuming matching row order.
"""

from pathlib import Path

import pandas as pd

from .common import ID_COLUMN, TARGET_COLUMNS


def find_table(data_dir, split, kind):
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Competition directory does not exist: {root}")
    candidates = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".xlsx", ".xls", ".csv", ".tsv"}:
            continue
        name = path.stem.upper()
        if split == "train" and not name.startswith("TRAIN"):
            continue
        if split == "test" and not name.startswith("TEST"):
            continue
        if kind.upper() in name:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No {split} {kind} table found under {root}")
    # Prefer the revised training data when the original release is also present.
    if split == "train":
        revised = [p for p in candidates if "NEW" in str(p.relative_to(root)).upper()]
        if revised:
            candidates = revised
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous {split} {kind} tables; keep one release/format: {candidates}")
    return candidates[0]


def read_table(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype={ID_COLUMN: str})
    else:
        frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                            dtype={ID_COLUMN: str})
    frame.columns = frame.columns.str.strip()
    if ID_COLUMN not in frame:
        raise ValueError(f"{path} has no {ID_COLUMN} column")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate participant IDs")
    return frame


def load_table(data_dir, split, kind):
    return read_table(find_table(data_dir, split, kind))


def load_targets(data_dir):
    targets = load_table(data_dir, "train", "SOLUTIONS")
    for column in TARGET_COLUMNS:
        if column not in targets or not targets[column].isin([0, 1]).all():
            raise ValueError(f"Solutions must contain binary, non-missing {column}")
    return targets[[ID_COLUMN, *TARGET_COLUMNS]]


def attach_targets(features, targets):
    if set(features[ID_COLUMN]) != set(targets[ID_COLUMN]):
        raise ValueError("Training feature and target participant IDs do not match")
    return features.merge(targets, on=ID_COLUMN, how="left", validate="one_to_one")


def load_dictionary(data_dir):
    paths = [p for p in Path(data_dir).expanduser().rglob("*.xlsx")
             if "dictionary" in p.stem.lower()]
    if len(paths) != 1:
        raise FileNotFoundError("EDA needs one Data Dictionary.xlsx under --data-dir")
    return pd.read_excel(paths[0], sheet_name="Dictionary")
