"""Convert the released undirected graph artifacts into forward-time directed graphs.

The released preprocessing writes each NetworkX edge in both directions and puts
the trajectory index in x[:, 0].  This conversion:

* removes x[:, 0] from every node feature matrix;
* collapses each reciprocal pair into one edge;
* orients that edge from the earlier timestamp to the later timestamp;
* drops equal-timestamp edges because the manuscript requires t_i < t_j;
* recomputes same/upper/lower relation types in the forward-time direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs" / "graphs"
CSV_DIR = ROOT / "data" / "raw_csv"
OUTPUT_DIR = ROOT / "outputs" / "graphs-directed"


def read_times(csv_path: Path) -> np.ndarray:
    frame = pd.read_csv(csv_path, low_memory=False).sort_values("ID", kind="stable")
    numeric = pd.to_numeric(frame["time"], errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(np.float64)
    parsed = pd.to_datetime(frame["time"], errors="raise")
    return parsed.astype("int64").to_numpy(np.float64) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, default=SOURCE_DIR)
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    csv_by_stem = {path.stem: path for path in args.csv_dir.glob("*.csv")}
    graph_paths = sorted(args.graphs.rglob("*_gnn_input.pt"), key=lambda path: path.name)
    if len(graph_paths) != len(csv_by_stem):
        raise ValueError(f"Expected {len(csv_by_stem)} graphs, found {len(graph_paths)}")

    totals = {
        "graphs": 0,
        "nodes": 0,
        "undirected_pairs": 0,
        "directed_edges": 0,
        "equal_time_edges_dropped": 0,
        "source_feature_dim": None,
        "output_feature_dim": None,
        "relation_counts": {"same": 0, "upper": 0, "lower": 0},
    }
    relation_index = {"same": 0, "upper": 1, "lower": 2}

    for number, graph_path in enumerate(graph_paths, 1):
        stem = graph_path.stem.removesuffix("_gnn_input")
        csv_path = csv_by_stem[stem]
        loaded = torch.load(graph_path, map_location="cpu", weights_only=False)
        if loaded.x.ndim != 2 or loaded.x.shape[1] < 2:
            raise ValueError(f"Unexpected feature matrix in {graph_path}")
        times = read_times(csv_path)
        if len(times) != loaded.num_nodes:
            raise ValueError(f"CSV/graph node mismatch for {csv_path.name}")

        src = loaded.edge_index[0].cpu().numpy()
        dst = loaded.edge_index[1].cpu().numpy()
        attrs = loaded.edge_attr.cpu().numpy()
        unordered = {}
        for source, target, attr in zip(src, dst, attrs):
            if source == target:
                continue
            key = (int(min(source, target)), int(max(source, target)))
            unordered.setdefault(key, attr)

        directed_edges = []
        directed_attrs = []
        for (left, right), old_attr in unordered.items():
            if times[left] < times[right]:
                source, target = left, right
            elif times[right] < times[left]:
                source, target = right, left
            else:
                totals["equal_time_edges_dropped"] += 1
                continue

            source_layer = int(loaded.layer[source])
            target_layer = int(loaded.layer[target])
            if target_layer == source_layer:
                relation = "same"
            elif target_layer > source_layer:
                relation = "upper"
            else:
                relation = "lower"

            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[relation_index[relation]] = 1.0
            edge_attr = np.concatenate(
                [np.asarray([old_attr[0], times[target] - times[source]], dtype=np.float32), one_hot]
            )
            directed_edges.append((source, target))
            directed_attrs.append(edge_attr)
            totals["relation_counts"][relation] += 1

        if directed_edges:
            edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.asarray(directed_attrs), dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float32)

        result = Data(
            x=loaded.x[:, 1:].clone(),
            y=loaded.y.clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            layer=loaded.layer.clone(),
            node_time=torch.tensor(times, dtype=torch.float64),
            num_nodes=int(loaded.num_nodes),
        )
        torch.save(result, args.output / graph_path.name)

        totals["graphs"] += 1
        totals["nodes"] += int(result.num_nodes)
        totals["undirected_pairs"] += len(unordered)
        totals["directed_edges"] += int(edge_index.shape[1])
        totals["source_feature_dim"] = int(loaded.x.shape[1])
        totals["output_feature_dim"] = int(result.x.shape[1])
        if number % 100 == 0:
            print(f"converted {number}/{len(graph_paths)}", flush=True)

    # Global structural assertions.
    if totals["directed_edges"] + totals["equal_time_edges_dropped"] != totals["undirected_pairs"]:
        raise AssertionError("Each undirected pair must become one directed edge or be dropped")
    if totals["source_feature_dim"] - totals["output_feature_dim"] != 1:
        raise AssertionError("Exactly the trajectory-index feature must be removed")

    report_path = args.output / "directed_graph_conversion.json"
    report_path.write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
