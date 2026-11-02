"""Run notebook EDA on CPU; never imports or trains a model."""

from pathlib import Path

from .common import data_parser


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--section", choices=["all", "categorical", "quantitative", "connectome"],
                        default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    import importlib
    import random
    import numpy as np

    from .data import load_dictionary, load_table, load_targets, attach_targets
    from .reporting import Report

    random.seed(args.seed)
    np.random.seed(args.seed)
    targets = load_targets(args.data_dir)
    sections = ["categorical", "quantitative", "connectome"] if args.section == "all" else [args.section]
    dictionary = load_dictionary(args.data_dir) if any(s != "connectome" for s in sections) else None
    for section in sections:
        print(f"Running {section} EDA...")
        frame = load_table(args.data_dir, "train", section.upper())
        attach_targets(frame, targets)  # Validate IDs before the original analysis joins.
        report = Report(args.output_dir / section, f"WiDS 2025 — {section} EDA")
        module = importlib.import_module(f"scripts.eda_{section}")
        module.run(frame, targets, dictionary, report)
        report.finish()


if __name__ == "__main__":
    main()
