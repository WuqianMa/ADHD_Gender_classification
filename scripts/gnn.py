"""PyTorch-only graph helpers, imported only by GNN training/inference.

Architecture: notebook cell 246 (three GraphConv + BatchNorm blocks).
Graph variants: identity/dense from cell 241; strength/positive from cell 271.
"""

import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GraphConv, global_mean_pool

from .common import ID_COLUMN


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use a CUDA-enabled PyTorch/GPU or --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


class BrainGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super().__init__()
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels * 2)
        self.bn2 = BatchNorm(hidden_channels * 2)
        self.conv3 = GraphConv(hidden_channels * 2, hidden_channels * 4)
        self.bn3 = BatchNorm(hidden_channels * 4)
        self.lin1 = nn.Linear(hidden_channels * 4, hidden_channels * 2)
        self.lin2 = nn.Linear(hidden_channels * 2, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_attr, batch):
        weight = edge_attr.reshape(-1)
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=weight)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_weight=weight)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_weight=weight)))
        x = self.dropout(global_mean_pool(x, batch))
        x = self.dropout(F.relu(self.lin1(x)))
        return self.lin2(x).reshape(-1)


class ConnectomeDataset(Dataset):
    """Construct one graph at a time on CPU instead of retaining every dense graph.

    Indices are parsed from column names, so CSV column order cannot change the
    reconstructed brain matrix. Missing/non-finite correlations become zero.
    """

    def __init__(self, frame, labels=None, num_nodes=200, node_features="identity",
                 edges="dense", edge_threshold=0.0):
        self.ids = frame[ID_COLUMN].tolist()
        self.num_nodes = num_nodes
        self.node_features = node_features
        self.edges = edges
        self.edge_threshold = edge_threshold
        columns, rows, cols = [], [], []
        for name in frame.columns:
            match = re.fullmatch(r"(\d+)throw_(\d+)thcolumn", name)
            if match:
                row, col = map(int, match.groups())
                if not (0 <= row < num_nodes and 0 <= col < num_nodes):
                    raise ValueError(f"Out-of-bounds connectome column: {name}")
                if row == col:
                    raise ValueError(f"Expected off-diagonal connectome features: {name}")
                columns.append(name)
                rows.append(row)
                cols.append(col)
        pairs = {tuple(sorted(pair)) for pair in zip(rows, cols)}
        expected = num_nodes * (num_nodes - 1) // 2
        if len(pairs) != expected or len(columns) != expected:
            raise ValueError(f"Expected {expected} unique triangle features for {num_nodes} nodes; found {len(columns)}")
        self.columns = columns
        self.rows = np.asarray(rows)
        self.cols = np.asarray(cols)
        self.values = np.nan_to_num(frame[columns].to_numpy(dtype=np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        if self.labels is not None and len(self.labels) != len(self.ids):
            raise ValueError("Labels and connectome rows must have the same length")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        matrix[self.rows, self.cols] = self.values[index]
        matrix[self.cols, self.rows] = self.values[index]
        # Original identity variant includes a unit diagonal in reconstruction;
        # the hybrid strength variant sums off-diagonal correlations only.
        if self.node_features == "identity":
            x = torch.eye(self.num_nodes)
        else:
            x = torch.from_numpy(np.abs(matrix).sum(axis=1).reshape(-1, 1))
        if self.edges == "positive":
            mask = matrix > self.edge_threshold
        else:
            mask = np.ones_like(matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        row, col = np.where(mask)
        graph = Data(x=x, edge_index=torch.from_numpy(np.stack([row, col])).long(),
                     edge_attr=torch.from_numpy(matrix[row, col]), participant_id=self.ids[index])
        if self.labels is not None:
            graph.y = torch.tensor([self.labels[index]], dtype=torch.float32)
        return graph


def predict_gnn(frame, checkpoint_path, device="auto", batch_size=32):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = select_device(device)
    config = checkpoint["config"]
    dataset = ConnectomeDataset(frame, **config["graph"])
    model = BrainGNN(**config["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            probabilities = torch.sigmoid(model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
            predictions.extend((probabilities >= checkpoint["threshold"]).long().cpu().tolist())
    return np.asarray(predictions, dtype=int)
