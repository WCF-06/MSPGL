"""Reproducible supplementary experiments for MSPGL.

This module implements the reviewer-requested experiments that can be run with
the released labelled data:

* leakage-free, group-wise out-of-fold (OOF) Stage-II training;
* repeated random seeds and trajectory-clustered bootstrap confidence intervals;
* Stage-II feature-group ablations;
* Stage-I graph/relation ablations;
* imbalance-stratified evaluation; and
* one-factor-at-a-time hyperparameter sensitivity analysis.

The script deliberately does not evaluate annotation agreement or retune the
published baselines.  All paths are relative and all machine-readable results
are written below one user-selected output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import corrected_experiment as core


DEFAULT_SEEDS = (42, 7, 19, 73, 101)
DEFAULT_GRAPH_SEEDS = (42, 73, 101)
RELATION_NAMES = ("same", "upper", "lower")

RAW_FEATURE_NAMES = (
    "normlon",
    "normlat",
    "normlevel",
    "optype",
    "numlevel",
    "trandistance",
    "trandirection",
    "opdistance",
    "standtime",
    "density",
)
RAW_GROUP_NAMES = tuple(f"raw_{name}" for name in RAW_FEATURE_NAMES)


@dataclass(frozen=True)
class Stage1Config:
    hidden_dim: int = 128
    heads: int = 4
    layer_emb_dim: int = 32
    dropout: float = 0.20
    learning_rate: float = 5e-4
    weight_decay: float = 5e-5
    focal_gamma: float = 1.0
    patience: int = 20
    max_epochs: int = 200


@dataclass(frozen=True)
class FeatureLayout:
    """Column indices for the clean Stage-II matrix.

    Layout: raw attributes, GAT embedding, Stage-I probability, level
    embedding, and neighbourhood-consistency features.  It excludes the eight
    all-zero placeholders and duplicate raw layer value in the legacy script.
    """

    raw: tuple[int, ...]
    graph: tuple[int, ...]
    stage1_probability: tuple[int, ...]
    level_embedding: tuple[int, ...]
    neighbourhood: tuple[int, ...]
    names: tuple[str, ...]

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        return {
            "raw": self.raw,
            "graph": self.graph,
            "stage1_probability": self.stage1_probability,
            "level_embedding": self.level_embedding,
            "neighbourhood": self.neighbourhood,
        }


class NodeMLP(nn.Module):
    """Node-only control with no graph message passing."""

    def __init__(
        self,
        input_dim: int,
        layer_emb_dim: int,
        max_layer: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        width = hidden_dim * heads
        self.layer_embedding = nn.Embedding(max_layer + 1, layer_emb_dim)
        nn.init.xavier_uniform_(self.layer_embedding.weight)
        self.network = nn.Sequential(
            nn.Linear(input_dim + layer_emb_dim, width),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(width, 2)

    def get_embedding(self, data: Data) -> torch.Tensor:
        layer_emb = self.layer_embedding(data.layer)
        return self.network(torch.cat([data.x, layer_emb], dim=1))

    def get_layer_embedding(self, layer: torch.Tensor) -> torch.Tensor:
        return self.layer_embedding(layer)

    def forward(self, data: Data) -> torch.Tensor:
        return self.classifier(self.get_embedding(data))


class StableFocalLoss(nn.Module):
    """Numerically stable focal loss for integer class labels.

    Floating-point roundoff can make ``exp(-cross_entropy)`` infinitesimally
    larger than one.  Fractional gamma values would then raise a negative
    number to a fractional power and produce NaN.  Explicit clamping preserves
    the mathematical domain without changing representable valid values.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(inputs, targets, reduction="none")
        probability_true_class = torch.exp(-cross_entropy).clamp(min=0.0, max=1.0)
        if self.gamma == 0:
            modulation = torch.ones_like(probability_true_class)
        else:
            epsilon = torch.finfo(probability_true_class.dtype).eps
            modulation = (1.0 - probability_true_class).clamp_min(epsilon).pow(self.gamma)
        return (self.alpha[targets] * modulation * cross_entropy).mean()


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(
    variant: str,
    input_dim: int,
    max_layer: int,
    config: Stage1Config,
    device: torch.device,
) -> nn.Module:
    kwargs = dict(
        input_dim=input_dim,
        layer_emb_dim=config.layer_emb_dim,
        max_layer=max_layer,
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        dropout=config.dropout,
    )
    if variant == "node_mlp":
        model = NodeMLP(**kwargs)
    else:
        model = core.EdgeTypeAwareGAT(
            input_dim=input_dim,
            layer_emb_dim=config.layer_emb_dim,
            max_layer=max_layer,
            hidden_dim=config.hidden_dim,
            num_heads=config.heads,
            dropout=config.dropout,
        )
    return model.to(device)


def clone_record_with_graph_variant(record: core.Record, variant: str) -> core.Record:
    """Create a lightweight record with a controlled graph perturbation."""

    data = record.data
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    if variant == "homogeneous_gat":
        edge_attr = edge_attr.clone()
        edge_attr[:, 2:] = 0.0
    elif variant.startswith("drop_"):
        relation = variant.removeprefix("drop_")
        if relation not in RELATION_NAMES:
            raise ValueError(f"Unknown relation in variant: {variant}")
        relation_index = RELATION_NAMES.index(relation)
        keep = edge_attr[:, 2 + relation_index] < 0.5
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]
    elif variant not in {"full", "node_mlp"}:
        raise ValueError(f"Unknown graph variant: {variant}")

    cloned = Data(
        x=data.x,
        y=data.y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        layer=data.layer,
        num_nodes=int(data.num_nodes),
    )
    cloned.node_id = torch.arange(cloned.num_nodes)
    cloned.y_count = record.n_positive
    cloned.y_ratio = record.n_positive / max(record.n_nodes, 1)
    return core.Record(
        index=record.index,
        file=record.file,
        ip=record.ip,
        ip_session=record.ip_session,
        data=cloned,
        hgmm=record.hgmm,
        hgmm_rf=record.hgmm_rf,
    )


def apply_graph_variant(records: Sequence[core.Record], variant: str) -> list[core.Record]:
    if variant == "full":
        return list(records)
    return [clone_record_with_graph_variant(record, variant) for record in records]


@torch.no_grad()
def predict_records(
    model: nn.Module,
    records: Sequence[core.Record],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records:
        data = record.data.to(device)
        probabilities.append(F.softmax(model(data), dim=1)[:, 1].cpu().numpy())
        labels.append(data.y.cpu().numpy())
        record.data = data.cpu()
    return np.concatenate(probabilities), np.concatenate(labels)


def train_stage1(
    train_records: Sequence[core.Record],
    validation_records: Sequence[core.Record],
    input_dim: int,
    max_layer: int,
    device: torch.device,
    seed: int,
    config: Stage1Config,
    tag: str,
    variant: str = "full",
) -> tuple[nn.Module, dict[str, object]]:
    seed_everything(seed)
    model = build_model(variant, input_dim, max_layer, config, device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(3, config.patience // 2)
    )
    loss_fn = StableFocalLoss(
        core.label_alpha(list(train_records), device), gamma=config.focal_gamma
    )
    loader = core.make_loader(list(train_records), train=True, seed=seed)

    best_auc = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(data), data.y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in {tag} at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_probability, val_labels = predict_records(model, validation_records, device)
        if not np.isfinite(val_probability).all():
            raise FloatingPointError(
                f"Non-finite validation probability in {tag} at epoch {epoch}"
            )
        val_auc = average_precision_score(val_labels, val_probability)
        scheduler.step(val_auc)
        if val_auc > best_auc + 1e-8:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            log(f"[{tag}] epoch={epoch:03d} val_auc_pr={val_auc:.6f} best={best_auc:.6f}")
        if stale >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a valid model for {tag}")
    model.load_state_dict(best_state)
    info: dict[str, object] = {
        "tag": tag,
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_auc_pr": best_auc,
        "seconds": time.time() - started,
        "config": asdict(config),
    }
    log(f"[{tag}] completed at epoch {best_epoch}; validation AUC-PR={best_auc:.6f}")
    return model, info


def make_feature_layout(input_dim: int, graph_dim: int, layer_dim: int) -> FeatureLayout:
    if input_dim != len(RAW_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(RAW_FEATURE_NAMES)} clean raw features, got {input_dim}"
        )
    names: list[str] = list(RAW_FEATURE_NAMES)
    raw = tuple(range(len(names)))
    graph_start = len(names)
    names.extend(f"GATemb_{index}" for index in range(graph_dim))
    graph = tuple(range(graph_start, len(names)))
    probability = (len(names),)
    names.append("Stage1Prob")
    level_start = len(names)
    names.extend(f"levelemb_{index}" for index in range(layer_dim))
    level = tuple(range(level_start, len(names)))
    neighbourhood_start = len(names)
    names.extend(("posratio", "meandiff", "maxdiff"))
    neighbourhood = tuple(range(neighbourhood_start, len(names)))
    return FeatureLayout(raw, graph, probability, level, neighbourhood, tuple(names))


def neighbourhood_features(
    data: Data,
    probabilities: torch.Tensor,
    candidate_ids: np.ndarray,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Return posratio, node-minus-mean, and node-minus-maximum.

    The neighbourhood is the union of incoming and outgoing incident nodes.
    This definition is fixed for all experiments and is documented for exact
    reproduction of the Stage-II feature construction.
    """

    edges = data.edge_index.detach().cpu().numpy()
    probability = probabilities.detach().cpu().numpy()
    source, target = edges
    result = np.zeros((len(candidate_ids), 3), dtype=np.float32)
    for row, node_id in enumerate(candidate_ids):
        neighbours = np.unique(
            np.concatenate((target[source == node_id], source[target == node_id]))
        )
        if len(neighbours) == 0:
            continue
        neighbour_probability = probability[neighbours]
        result[row] = (
            np.mean(neighbour_probability >= confidence_threshold),
            probability[node_id] - np.mean(neighbour_probability),
            probability[node_id] - np.max(neighbour_probability),
        )
    return result


@torch.no_grad()
def extract_clean_features(
    model: nn.Module,
    records: Sequence[core.Record],
    device: torch.device,
    stage1_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], FeatureLayout]:
    model.eval()
    batches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    references: list[tuple[int, int]] = []
    layout: FeatureLayout | None = None
    for record in records:
        data = record.data.to(device)
        output = model(data)
        probability = F.softmax(output, dim=1)[:, 1]
        candidate_ids = torch.where(probability >= stage1_threshold)[0].cpu().numpy()
        if len(candidate_ids):
            raw = data.x[candidate_ids].cpu().numpy().astype(np.float32)
            graph = model.get_embedding(data)[candidate_ids].cpu().numpy().astype(np.float32)
            stage1_probability = probability[candidate_ids].cpu().numpy().astype(np.float32)[:, None]
            level = model.get_layer_embedding(data.layer)[candidate_ids].cpu().numpy().astype(np.float32)
            neighbourhood = neighbourhood_features(data, probability, candidate_ids)
            fused = np.concatenate(
                (raw, graph, stage1_probability, level, neighbourhood), axis=1
            ).astype(np.float32, copy=False)
            current_layout = make_feature_layout(raw.shape[1], graph.shape[1], level.shape[1])
            if layout is None:
                layout = current_layout
            elif layout.names != current_layout.names:
                raise AssertionError("Feature layout changed between trajectories")
            batches.append(fused)
            labels.append(data.y[candidate_ids].cpu().numpy().astype(np.int8))
            references.extend((record.index, int(node_id)) for node_id in candidate_ids)
        record.data = data.cpu()
    if not batches or layout is None:
        raise RuntimeError("Stage I retained no candidate nodes")
    return np.concatenate(batches), np.concatenate(labels), references, layout


def select_columns(layout: FeatureLayout, included_groups: Iterable[str]) -> np.ndarray:
    columns: list[int] = []
    for group in included_groups:
        if group.startswith("raw_"):
            name = group.removeprefix("raw_")
            if name not in RAW_FEATURE_NAMES:
                raise KeyError(f"Unknown raw feature group: {group}")
            columns.append(layout.raw[RAW_FEATURE_NAMES.index(name)])
        else:
            if group not in layout.groups:
                raise KeyError(f"Unknown feature group: {group}")
            columns.extend(layout.groups[group])
    if not columns:
        raise ValueError("At least one feature group must be selected")
    return np.asarray(sorted(set(columns)), dtype=np.int64)


def stage2_ablation_definitions() -> dict[str, tuple[str, ...]]:
    all_groups = ("stage1_probability", "raw", "neighbourhood", "level_embedding", "graph")
    definitions = {
        "probability_only": ("stage1_probability",),
        "probability_plus_raw": ("stage1_probability", "raw"),
        "probability_plus_neighbourhood": ("stage1_probability", "neighbourhood"),
        "probability_plus_level": ("stage1_probability", "level_embedding"),
        "probability_plus_graph": ("stage1_probability", "graph"),
        "full": all_groups,
        "full_minus_probability": tuple(x for x in all_groups if x != "stage1_probability"),
        "full_minus_raw": tuple(x for x in all_groups if x != "raw"),
        "full_minus_neighbourhood": tuple(x for x in all_groups if x != "neighbourhood"),
        "full_minus_level": tuple(x for x in all_groups if x != "level_embedding"),
        "full_minus_graph": tuple(x for x in all_groups if x != "graph"),
    }
    non_raw_groups = tuple(x for x in all_groups if x != "raw")
    for name in RAW_FEATURE_NAMES:
        kept_raw = tuple(f"raw_{other}" for other in RAW_FEATURE_NAMES if other != name)
        definitions[f"full_minus_raw_{name}"] = non_raw_groups + kept_raw
    return definitions


def fit_stage2(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    seed: int,
) -> tuple[xgb.XGBClassifier, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x).astype(np.float32)
    validation_scaled = scaler.transform(validation_x).astype(np.float32)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=float(np.sum(train_y == 0) / max(np.sum(train_y == 1), 1)),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        tree_method="hist",
        # Stage-II matrices are NumPy arrays on host memory. CPU histogram
        # training avoids implicit device transfers and is fast at this scale.
        device="cpu",
        early_stopping_rounds=20,
        n_jobs=1,
    )
    model.fit(train_scaled, train_y, eval_set=[(validation_scaled, validation_y)], verbose=False)
    return model, scaler


def metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "auc_pr": float(average_precision_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(labels)),
    }
    if len(np.unique(labels)) == 2:
        result["auc_roc"] = float(roc_auc_score(labels, probabilities))
    return result


def inner_group_split(records: Sequence[core.Record], seed: int) -> tuple[list[core.Record], list[core.Record]]:
    indices = np.arange(len(records))
    groups = np.asarray([record.ip for record in records])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_ids, validation_ids = next(splitter.split(indices, groups=groups))
    train_records = [records[int(index)] for index in train_ids]
    validation_records = [records[int(index)] for index in validation_ids]
    if {record.ip for record in train_records} & {record.ip for record in validation_records}:
        raise AssertionError("Inner train/validation IP groups overlap")
    return train_records, validation_records


def make_oof_features(
    train_records: Sequence[core.Record],
    input_dim: int,
    max_layer: int,
    device: torch.device,
    seed: int,
    folds: int,
    config: Stage1Config,
    variant: str = "full",
) -> tuple[np.ndarray, np.ndarray, FeatureLayout, list[dict[str, object]]]:
    indices = np.arange(len(train_records))
    groups = np.asarray([record.ip for record in train_records])
    try:
        splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    except TypeError:
        splitter = GroupKFold(n_splits=folds)

    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    layout: FeatureLayout | None = None
    for fold, (fit_ids, holdout_ids) in enumerate(splitter.split(indices, groups=groups), 1):
        fit_pool = [train_records[int(index)] for index in fit_ids]
        holdout = [train_records[int(index)] for index in holdout_ids]
        inner_train, inner_validation = inner_group_split(fit_pool, seed + 10_000 + fold)
        if {record.ip for record in fit_pool} & {record.ip for record in holdout}:
            raise AssertionError("OOF fitting and holdout IP groups overlap")
        model, train_report = train_stage1(
            inner_train,
            inner_validation,
            input_dim,
            max_layer,
            device,
            seed + fold,
            config,
            tag=f"seed-{seed}-oof-{fold}",
            variant=variant,
        )
        inner_probability, inner_labels = predict_records(model, inner_validation, device)
        threshold = core.select_threshold(inner_probability, inner_labels, core.TARGET_RECALL)
        features, labels, _, current_layout = extract_clean_features(
            model, holdout, device, threshold
        )
        if layout is None:
            layout = current_layout
        elif layout.names != current_layout.names:
            raise AssertionError("OOF feature layouts do not match")
        feature_batches.append(features)
        label_batches.append(labels)
        fold_reports.append(
            {
                **train_report,
                "inner_train_trajectories": len(inner_train),
                "inner_validation_trajectories": len(inner_validation),
                "holdout_trajectories": len(holdout),
                "candidate_threshold": threshold,
                "candidate_nodes": int(len(labels)),
                "candidate_positives": int(labels.sum()),
            }
        )
        del model, features, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if layout is None:
        raise RuntimeError("No OOF features were created")
    return np.concatenate(feature_batches), np.concatenate(label_batches), layout, fold_reports


def all_node_probabilities(
    records: Sequence[core.Record],
    references: Sequence[tuple[int, int]],
    candidate_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return core.all_node_stage2_prob(list(records), list(references), candidate_probabilities)


def trajectory_slices(records: Sequence[core.Record]) -> list[slice]:
    slices: list[slice] = []
    start = 0
    for record in records:
        stop = start + record.n_nodes
        slices.append(slice(start, stop))
        start = stop
    return slices


def run_seed(
    seed: int,
    records: Sequence[core.Record],
    split: dict[str, np.ndarray],
    device: torch.device,
    config: Stage1Config,
    folds: int,
    output_dir: Path,
) -> dict[str, object]:
    seed_dir = output_dir / f"seed_{seed}"
    result_path = seed_dir / "result.json"
    if result_path.exists():
        log(f"[seed-{seed}] reusing {result_path}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    seed_dir.mkdir(parents=True, exist_ok=True)
    split_records = {
        name: [records[int(index)] for index in indices] for name, indices in split.items()
    }
    input_dim = int(records[0].data.x.shape[1])
    max_layer = max(int(record.data.layer.max().item()) for record in records)

    final_model, stage1_training = train_stage1(
        split_records["train"],
        split_records["val"],
        input_dim,
        max_layer,
        device,
        seed,
        config,
        tag=f"seed-{seed}-final",
    )
    validation_probability, validation_labels = predict_records(
        final_model, split_records["val"], device
    )
    stage1_threshold = core.select_threshold(
        validation_probability, validation_labels, core.TARGET_RECALL
    )
    test_probability, test_labels = predict_records(final_model, split_records["test"], device)
    stage1_result = metric_values(test_labels, test_probability, stage1_threshold)

    oof_x, oof_y, oof_layout, fold_reports = make_oof_features(
        split_records["train"], input_dim, max_layer, device, seed, folds, config
    )
    validation_x, validation_y, validation_refs, validation_layout = extract_clean_features(
        final_model, split_records["val"], device, stage1_threshold
    )
    test_x, test_y, test_refs, test_layout = extract_clean_features(
        final_model, split_records["test"], device, stage1_threshold
    )
    if not (oof_layout.names == validation_layout.names == test_layout.names):
        raise AssertionError("Train/validation/test feature layouts differ")

    ablation_results: dict[str, object] = {}
    full_all_node_probability: np.ndarray | None = None
    full_threshold: float | None = None
    for name, groups in stage2_ablation_definitions().items():
        columns = select_columns(oof_layout, groups)
        model, scaler = fit_stage2(
            oof_x[:, columns],
            oof_y,
            validation_x[:, columns],
            validation_y,
            seed,
        )
        validation_candidate_probability = model.predict_proba(
            scaler.transform(validation_x[:, columns]).astype(np.float32)
        )[:, 1]
        test_candidate_probability = model.predict_proba(
            scaler.transform(test_x[:, columns]).astype(np.float32)
        )[:, 1]
        validation_all, validation_all_labels = all_node_probabilities(
            split_records["val"], validation_refs, validation_candidate_probability
        )
        test_all, test_all_labels = all_node_probabilities(
            split_records["test"], test_refs, test_candidate_probability
        )
        threshold = core.select_threshold(validation_all, validation_all_labels)
        ablation_results[name] = {
            "included_groups": list(groups),
            "n_features": int(len(columns)),
            "best_iteration": int(model.best_iteration),
            "validation_threshold": threshold,
            "test": metric_values(test_all_labels, test_all, threshold),
        }
        if name == "full":
            full_all_node_probability = test_all
            full_threshold = threshold
            joblib.dump(model, seed_dir / "stage2_full.joblib")
            joblib.dump(scaler, seed_dir / "stage2_full_scaler.joblib")

    if full_all_node_probability is None or full_threshold is None:
        raise AssertionError("Full Stage-II model was not evaluated")
    torch.save(final_model.state_dict(), seed_dir / "stage1_final.pt")
    np.savez_compressed(
        seed_dir / "test_predictions.npz",
        labels=test_labels.astype(np.int8),
        stage1_probability=test_probability.astype(np.float32),
        stage2_probability=full_all_node_probability.astype(np.float32),
        stage1_threshold=np.float32(stage1_threshold),
        stage2_threshold=np.float32(full_threshold),
        trajectory_lengths=np.asarray([r.n_nodes for r in split_records["test"]], dtype=np.int32),
    )
    write_json(
        seed_dir / "feature_layout.json",
        {"names": list(oof_layout.names), "groups": {k: list(v) for k, v in oof_layout.groups.items()}},
    )
    result: dict[str, object] = {
        "seed": seed,
        "stage1_training": stage1_training,
        "oof_training": fold_reports,
        "stage1": stage1_result,
        "stage2_ablations": ablation_results,
        "candidate_counts": {
            "oof_train": int(len(oof_y)),
            "validation": int(len(validation_y)),
            "test": int(len(test_y)),
        },
    }
    write_json(result_path, result)
    return result


def summarise_seed_results(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        seed = int(result["seed"])
        stage1 = result["stage1"]
        rows.append({"seed": seed, "experiment": "stage1", **stage1})
        for name, ablation in result["stage2_ablations"].items():
            rows.append({"seed": seed, "experiment": f"stage2_{name}", **ablation["test"]})
    return pd.DataFrame(rows)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ("precision", "recall", "f1", "auc_pr")
    rows: list[dict[str, object]] = []
    for experiment, group in frame.groupby("experiment", sort=False):
        row: dict[str, object] = {"experiment": experiment, "n_seeds": int(len(group))}
        for metric in metric_columns:
            values = group[metric].astype(float)
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            row[f"{metric}_cv"] = standard_deviation / mean if mean else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_across_trajectories(
    output_dir: Path,
    seeds: Sequence[int],
    replicates: int,
    random_seed: int = 20260915,
) -> pd.DataFrame:
    payloads = []
    for seed in seeds:
        stored = np.load(output_dir / f"seed_{seed}" / "test_predictions.npz")
        payloads.append({name: stored[name] for name in stored.files})
    lengths = payloads[0]["trajectory_lengths"].astype(int)
    if any(not np.array_equal(item["trajectory_lengths"], lengths) for item in payloads):
        raise AssertionError("Trajectory layouts differ across seeds")
    slices: list[slice] = []
    start = 0
    for length in lengths:
        slices.append(slice(start, start + int(length)))
        start += int(length)

    rng = np.random.default_rng(random_seed)
    metric_names = ("precision", "recall", "f1", "auc_pr")
    distributions = {"stage1": {m: [] for m in metric_names}, "stage2_full": {m: [] for m in metric_names}}
    for _ in range(replicates):
        chosen = rng.integers(0, len(slices), size=len(slices))
        for experiment, probability_key, threshold_key in (
            ("stage1", "stage1_probability", "stage1_threshold"),
            ("stage2_full", "stage2_probability", "stage2_threshold"),
        ):
            per_seed = []
            for item in payloads:
                labels = np.concatenate([item["labels"][slices[index]] for index in chosen])
                probabilities = np.concatenate(
                    [item[probability_key][slices[index]] for index in chosen]
                )
                per_seed.append(metric_values(labels, probabilities, float(item[threshold_key])))
            for metric in metric_names:
                distributions[experiment][metric].append(
                    float(np.mean([result[metric] for result in per_seed]))
                )

    rows = []
    for experiment, metrics_by_name in distributions.items():
        for metric, values in metrics_by_name.items():
            array = np.asarray(values)
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "bootstrap_replicates": replicates,
                    "mean": float(array.mean()),
                    "ci_2.5%": float(np.quantile(array, 0.025)),
                    "ci_97.5%": float(np.quantile(array, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def imbalance_analysis(
    records: Sequence[core.Record],
    split: dict[str, np.ndarray],
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_records = [records[int(index)] for index in split["test"]]
    stored = np.load(output_dir / f"seed_{seed}" / "test_predictions.npz")
    probabilities = stored["stage2_probability"]
    labels = stored["labels"]
    threshold = float(stored["stage2_threshold"])
    slices = trajectory_slices(test_records)
    rows: list[dict[str, object]] = []
    for record, current_slice in zip(test_records, slices):
        current_labels = labels[current_slice]
        current_probability = probabilities[current_slice]
        positive = int(current_labels.sum())
        negative = int(len(current_labels) - positive)
        imbalance = 1.0 - (2.0 * min(positive, negative) / max(positive + negative, 1))
        item: dict[str, object] = {
            "file": record.file,
            "IP": record.ip,
            "nodes": int(len(current_labels)),
            "positives": positive,
            "negatives": negative,
            "IMB": imbalance,
        }
        if positive > 0:
            item.update(metric_values(current_labels, current_probability, threshold))
        rows.append(item)
    detail = pd.DataFrame(rows)
    detail["IMB_group"], boundaries = pd.qcut(
        detail["IMB"], q=3, labels=("Low", "Mid", "High"), retbins=True, duplicates="drop"
    )
    summaries = []
    for group_name, group in detail.groupby("IMB_group", observed=True):
        summaries.append(
            {
                "IMB_group": str(group_name),
                "trajectories": int(len(group)),
                "nodes": int(group["nodes"].sum()),
                "positives": int(group["positives"].sum()),
                "IMB_min": float(group["IMB"].min()),
                "IMB_max": float(group["IMB"].max()),
                "F1_median": float(group["f1"].median()),
                "F1_IQR": float(group["f1"].quantile(0.75) - group["f1"].quantile(0.25)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.attrs["quantile_boundaries"] = boundaries.tolist()
    return detail, summary


def run_graph_ablations(
    records: Sequence[core.Record],
    split: dict[str, np.ndarray],
    device: torch.device,
    config: Stage1Config,
    seeds: Sequence[int],
    output_dir: Path,
) -> pd.DataFrame:
    result_path = output_dir / "graph_ablation_runs.csv"
    existing = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    completed = set(zip(existing.get("variant", []), existing.get("seed", [])))
    rows = existing.to_dict("records")
    input_dim = int(records[0].data.x.shape[1])
    max_layer = max(int(record.data.layer.max().item()) for record in records)
    variants = ("full", "homogeneous_gat", "node_mlp", "drop_same", "drop_upper", "drop_lower")
    for variant in variants:
        variant_records = apply_graph_variant(records, variant)
        split_records = {
            name: [variant_records[int(index)] for index in indices] for name, indices in split.items()
        }
        for seed in seeds:
            if (variant, seed) in completed:
                continue
            if variant == "full":
                core_result_path = output_dir / f"seed_{seed}" / "result.json"
                if core_result_path.exists():
                    core_result = json.loads(core_result_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "best_epoch": core_result["stage1_training"]["best_epoch"],
                            **core_result["stage1"],
                        }
                    )
                    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                    continue
            model, training = train_stage1(
                split_records["train"],
                split_records["val"],
                input_dim,
                max_layer,
                device,
                seed,
                config,
                tag=f"graph-{variant}-{seed}",
                variant="node_mlp" if variant == "node_mlp" else "full",
            )
            validation_probability, validation_labels = predict_records(
                model, split_records["val"], device
            )
            threshold = core.select_threshold(
                validation_probability, validation_labels, core.TARGET_RECALL
            )
            test_probability, test_labels = predict_records(model, split_records["test"], device)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "best_epoch": training["best_epoch"],
                    **metric_values(test_labels, test_probability, threshold),
                }
            )
            pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def sensitivity_grid(base: Stage1Config) -> dict[str, tuple[object, ...]]:
    return {
        "hidden_dim": (64, 128, 256),
        "dropout": (0.0, 0.1, 0.2, 0.3, 0.5),
        "learning_rate": (1e-4, 3e-4, 5e-4, 1e-3, 2e-3),
        "weight_decay": (0.0, 1e-5, 5e-5, 1e-4, 5e-4),
        "heads": (1, 2, 4, 8),
        "layer_emb_dim": (8, 16, 32, 64),
        "focal_gamma": (0.0, 0.5, 1.0, 2.0),
        "patience": (5, 10, 20, 30),
    }


def canonical_parameter_value(value: object) -> str:
    """Normalize CSV-loaded numeric values for reliable checkpoint matching."""

    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def run_hyperparameter_sensitivity(
    records: Sequence[core.Record],
    split: dict[str, np.ndarray],
    device: torch.device,
    base_config: Stage1Config,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    result_path = output_dir / "hyperparameter_sensitivity_runs.csv"
    existing = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    completed = (
        {
            (str(row.hyperparameter), canonical_parameter_value(row.value))
            for row in existing.itertuples(index=False)
        }
        if len(existing)
        else set()
    )
    rows = existing.to_dict("records")
    split_records = {
        name: [records[int(index)] for index in indices] for name, indices in split.items()
    }
    input_dim = int(records[0].data.x.shape[1])
    max_layer = max(int(record.data.layer.max().item()) for record in records)
    for parameter, values in sensitivity_grid(base_config).items():
        for value in values:
            key = (parameter, canonical_parameter_value(value))
            if key in completed:
                continue
            config = replace(base_config, **{parameter: value})
            if config == base_config:
                core_result_path = output_dir / f"seed_{seed}" / "result.json"
                if core_result_path.exists():
                    core_result = json.loads(core_result_path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "hyperparameter": parameter,
                            "value": value,
                            "seed": seed,
                            "best_epoch": core_result["stage1_training"]["best_epoch"],
                            "best_validation_auc_pr": core_result["stage1_training"][
                                "best_validation_auc_pr"
                            ],
                            **core_result["stage1"],
                        }
                    )
                    pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
                    continue
            model, training = train_stage1(
                split_records["train"],
                split_records["val"],
                input_dim,
                max_layer,
                device,
                seed,
                config,
                tag=f"sensitivity-{parameter}-{value}",
            )
            validation_probability, validation_labels = predict_records(
                model, split_records["val"], device
            )
            threshold = core.select_threshold(
                validation_probability, validation_labels, core.TARGET_RECALL
            )
            test_probability, test_labels = predict_records(model, split_records["test"], device)
            rows.append(
                {
                    "hyperparameter": parameter,
                    "value": value,
                    "seed": seed,
                    "best_epoch": training["best_epoch"],
                    "best_validation_auc_pr": training["best_validation_auc_pr"],
                    **metric_values(test_labels, test_probability, threshold),
                }
            )
            pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarise_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for parameter, group in frame.groupby("hyperparameter", sort=False):
        for metric in ("precision", "recall", "f1", "auc_pr"):
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rows.append(
                {
                    "hyperparameter": parameter,
                    "metric": metric,
                    "n_configurations": int(len(values)),
                    "mean": mean,
                    "standard_deviation": std,
                    "coefficient_of_variation": std / mean if mean else math.nan,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def save_split_manifest(
    records: Sequence[core.Record], split: dict[str, np.ndarray], output_dir: Path
) -> None:
    rows = []
    for name, indices in split.items():
        for index in indices:
            record = records[int(index)]
            rows.append(
                {
                    "file": record.file,
                    "anonymized_group": record.ip,
                    "session": record.ip_session,
                    "split": name,
                }
            )
    pd.DataFrame(rows).sort_values(["split", "file"]).to_csv(
        output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("core", "graph", "hyperparameter"),
        default=("core",),
        help="Experiment families to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--graph-seeds", nargs="+", type=int, default=DEFAULT_GRAPH_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/supplementary_experiments"))
    parser.add_argument("--source", type=Path, default=core.SOURCE)
    parser.add_argument("--data-dir", type=Path, default=core.DATA_DIR)
    parser.add_argument("--graph-dir", type=Path, default=core.GRAPH_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    core.SOURCE = args.source
    core.DATA_DIR = args.data_dir
    core.GRAPH_DIR = args.graph_dir
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}; gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    records = core.load_records()
    split = core.choose_group_split(records)
    save_split_manifest(records, split, args.output)
    split_stats = {name: core.subset_stats(records, indices) for name, indices in split.items()}
    config = Stage1Config(max_epochs=args.max_epochs)
    write_json(
        args.output / "run_configuration.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "experiments": args.experiments,
            "seeds": args.seeds,
            "graph_seeds": args.graph_seeds,
            "folds": args.folds,
            "bootstrap_replicates": args.bootstrap_replicates,
            "stage1": asdict(config),
            "split": split_stats,
        },
    )

    if "core" in args.experiments:
        results = [
            run_seed(seed, records, split, device, config, args.folds, args.output)
            for seed in args.seeds
        ]
        runs = summarise_seed_results(results)
        runs.to_csv(args.output / "seed_runs.csv", index=False, encoding="utf-8-sig")
        aggregate_metrics(runs).to_csv(
            args.output / "seed_summary.csv", index=False, encoding="utf-8-sig"
        )
        bootstrap_across_trajectories(
            args.output, args.seeds, args.bootstrap_replicates
        ).to_csv(args.output / "trajectory_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
        detail, summary = imbalance_analysis(records, split, args.output, args.seeds[0])
        detail.to_csv(args.output / "imbalance_trajectory_detail.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(args.output / "imbalance_group_summary.csv", index=False, encoding="utf-8-sig")
        write_json(
            args.output / "imbalance_group_boundaries.json",
            {"quantile_boundaries": summary.attrs["quantile_boundaries"]},
        )

    if "graph" in args.experiments:
        graph_runs = run_graph_ablations(
            records, split, device, config, args.graph_seeds, args.output
        )
        graph_summary = aggregate_metrics(
            graph_runs.rename(columns={"variant": "experiment"})
        )
        graph_summary.to_csv(
            args.output / "graph_ablation_summary.csv", index=False, encoding="utf-8-sig"
        )

    if "hyperparameter" in args.experiments:
        sensitivity = run_hyperparameter_sensitivity(
            records, split, device, config, args.seeds[0], args.output
        )
        summarise_sensitivity(sensitivity).to_csv(
            args.output / "hyperparameter_sensitivity_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    log(f"All requested experiments completed. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
