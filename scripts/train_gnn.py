"""Train the notebook's Sex_F GNN independently of EDA and classical models."""

import json
from pathlib import Path

from .common import ID_COLUMN, TARGET_COLUMNS, data_parser, positive_int


def run_epoch(model, loader, criterion, device, optimizer=None):
    import numpy as np
    import torch

    model.train(optimizer is not None)
    total_loss, labels, probabilities = 0.0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.reshape(-1)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite GNN loss; check the input correlations")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.numel()
            labels.extend(targets.detach().cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
    if not labels:
        raise ValueError("Empty graph loader")
    return total_loss / len(labels), np.asarray(labels), np.asarray(probabilities)


def train(args):
    import random
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    from .data import attach_targets, load_table, load_targets
    from .gnn import BrainGNN, ConnectomeDataset, select_device

    device = select_device(args.device)
    print(f"Training GNN on {device}; batch size {args.batch_size}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = attach_targets(load_table(args.data_dir, "train", "CONNECTOME"), load_targets(args.data_dir))
    combined = frame[TARGET_COLUMNS].astype(str).agg("_".join, axis=1).to_numpy()
    indices = np.arange(len(frame))
    train_val, test = train_test_split(indices, test_size=0.15, random_state=args.seed, stratify=combined)
    train_indices, validation = train_test_split(train_val, test_size=0.15 / 0.85,
                                                 random_state=args.seed, stratify=combined[train_val])
    graph_config = {"num_nodes": args.num_nodes, "node_features": args.node_features,
                    "edges": args.edges, "edge_threshold": args.edge_threshold}
    config = {"graph": graph_config,
              "model": {"num_node_features": args.num_nodes if args.node_features == "identity" else 1,
                        "hidden_channels": args.hidden_channels}}
    dataset = ConnectomeDataset(frame, frame["Sex_F"].to_numpy(), **graph_config)

    def loader(selected, shuffle=False):
        return DataLoader(Subset(dataset, selected.tolist()), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")

    train_loader, val_loader = loader(train_indices, True), loader(validation)
    model = BrainGNN(**config["model"]).to(device)
    y_train = frame.iloc[train_indices]["Sex_F"].to_numpy()
    # Count Sex_F only; the notebook's earlier implementation counted both labels.
    if not (y_train == 1).any() or not (y_train == 0).any():
        raise ValueError("Both Sex_F classes are required in the training split")
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_val, val_prob = run_epoch(model, val_loader, criterion, device)
        val_f1 = float(f1_score(y_val, val_prob >= 0.5, zero_division=0))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1})
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            # Persist progress during long GPU runs, not only at the end.
            torch.save({"state_dict": best_state, "config": config, "threshold": 0.5,
                        "epoch": best_epoch, "seed": args.seed}, args.output_dir / "sex_gnn.pt")
    model.load_state_dict(best_state)
    _, y_val, val_prob = run_epoch(model, val_loader, criterion, device)
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [float(f1_score(y_val, val_prob >= t, zero_division=0)) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])
    # The untouched test partition is evaluated once, after validation selection.
    test_loss, y_test, test_prob = run_epoch(model, loader(test), criterion, device)
    metrics = {"best_epoch": best_epoch, "threshold": threshold,
               "validation_f1": max(scores), "test_loss": test_loss,
               "test_f1": float(f1_score(y_test, test_prob >= threshold, zero_division=0)),
               "test_accuracy": float(accuracy_score(y_test, test_prob >= threshold)),
               "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
               "train_rows": len(train_indices), "validation_rows": len(validation), "test_rows": len(test)}
    torch.save({"state_dict": best_state, "config": config, "threshold": threshold,
                "epoch": best_epoch, "seed": args.seed}, args.output_dir / "sex_gnn.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "gnn_history.csv", index=False)
    pd.DataFrame({ID_COLUMN: frame.iloc[test][ID_COLUMN].to_numpy(),
                  "Sex_F": y_test.astype(int), "probability": test_prob}).to_csv(
                      args.output_dir / "gnn_test_predictions.csv", index=False)
    (args.output_dir / "gnn_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def main():
    parser = data_parser(__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=positive_int, default=200)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--num-nodes", type=positive_int, default=200)
    parser.add_argument("--hidden-channels", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--node-features", choices=["identity", "strength"], default="identity")
    parser.add_argument("--edges", choices=["dense", "positive"], default="dense")
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    train(args)


if __name__ == "__main__":
    main()
