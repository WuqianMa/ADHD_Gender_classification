"""Shared configuration; importing this module never initializes a GPU."""

import argparse
from pathlib import Path

SEED = 8828
ID_COLUMN = "participant_id"
TARGET_COLUMNS = ["ADHD_Outcome", "Sex_F"]
TARGET_MAP = {
    "ADHD_Outcome": {0: "Non-ADHD", 1: "ADHD"},
    "Sex_F": {0: "Male", 1: "Female"},
}
ADHD_FEATURES = [
    "SDQ_SDQ_Conduct_Problems", "SDQ_SDQ_Difficulties_Total",
    "SDQ_SDQ_Externalizing", "SDQ_SDQ_Generating_Impact",
    "SDQ_SDQ_Hyperactivity", "SDQ_SDQ_Emotional_Problems",
    "SDQ_SDQ_Internalizing", "APQ_P_APQ_P_ID", "APQ_P_APQ_P_INV",
    "APQ_P_APQ_P_PM", "SDQ_SDQ_Peer_Problems", "SDQ_SDQ_Prosocial",
]


def data_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Extracted competition directory (searched recursively).")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value
