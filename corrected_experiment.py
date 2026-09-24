"""Leakage-corrected MSPGL experiment.

Changes relative to the released training notebook:
1. train/validation/test are disjoint by anonymized IP;
2. Stage-II training features are generated out-of-fold at the IP/trajectory level;
3. the trajectory-index/ID feature is removed from Stage I and Stage II;
4. graph edges are single, forward-time directed edges rather than reciprocal pairs;
5. Stage-I and Stage-II thresholds/model selection use validation only;
6. the untouched test split is evaluated once after all choices are fixed.

The Stage-I architecture, remaining feature layout, focal loss, sampling weights,
and main hyperparameters remain aligned with the released code.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

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
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MSPGL.py"
DATA_DIR = ROOT / "data" / "raw_csv"
GRAPH_DIR = ROOT / "outputs" / "graphs-directed"
OUTPUT_DIR = ROOT / "outputs" / "corrected_directed_noindex"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TARGET_RECALL = 0.80
HIDDEN_DIM = 128
HEADS = 4
LAYER_EMB_DIM = 32
DROPOUT = 0.20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 20
MAX_EPOCHS = 200


@dataclass
class Record:
    index: int
    file: str
    ip: str
    ip_session: str
    data: object
    hgmm: np.ndarray
    hgmm_rf: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.data.num_nodes)

    @property
    def n_positive(self) -> int:
        return int((self.data.y == 1).sum().item())


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * (1 - pt) ** self.gamma * ce).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_released_model_classes() -> tuple[type, type]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"EdgeTypeAwareGATConv", "EdgeTypeAwareGAT"}
        and node.lineno < 1800
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "MessagePassing": MessagePassing,
        "BatchNorm": BatchNorm,
        "glorot": glorot,
        "zeros": zeros,
        "softmax": softmax,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["EdgeTypeAwareGATConv"], namespace["EdgeTypeAwareGAT"]


_, EdgeTypeAwareGAT = load_released_model_classes()


def read_csv_metadata(path: Path) -> tuple[str, str, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) == 0:
        raise ValueError(f"Empty trajectory: {path}")
    frame = frame.sort_values("ID", kind="stable")
    ip_values = frame["IP"].astype(str).str.strip().unique()
    session_values = frame["IP_session"].astype(str).str.strip().unique()
    if len(ip_values) != 1 or len(session_values) != 1:
        raise ValueError(f"Expected one IP and IP_session in {path.name}")
    hgmm_col = "leaves" if "leaves" in frame else "HGMM"
    hgmm_rf_col = "leaves_rf" if "leaves_rf" in frame else "HGMM_RF"
    hgmm = (frame[hgmm_col].astype(str).str.upper() == "Y").to_numpy(np.int8)
    hgmm_rf = (frame[hgmm_rf_col].astype(str).str.upper() == "Y").to_numpy(np.int8)
    return ip_values[0], session_values[0], hgmm, hgmm_rf


def load_records() -> list[Record]:
    csv_by_stem = {path.stem: path for path in DATA_DIR.glob("*.csv")}
    graph_paths = sorted(GRAPH_DIR.rglob("*_gnn_input.pt"), key=lambda p: p.name)
    if len(graph_paths) != len(csv_by_stem):
        raise ValueError(f"Expected {len(csv_by_stem)} graphs, found {len(graph_paths)}")
    records = []
    seen = set()
    for graph_path in graph_paths:
        stem = graph_path.stem.removesuffix("_gnn_input")
        csv_path = csv_by_stem.get(stem)
        if csv_path is None:
            raise KeyError(f"No CSV corresponds to {graph_path.name}")
        loaded = torch.load(graph_path, map_location="cpu", weights_only=False)
        data = Data(
            x=loaded.x,
            y=loaded.y,
            edge_index=loaded.edge_index,
            edge_attr=loaded.edge_attr,
            layer=loaded.layer,
            num_nodes=int(loaded.num_nodes),
        )
        data.node_id = torch.arange(data.num_nodes)
        data.y_count = int((data.y == 1).sum().item())
        data.y_ratio = data.y_count / max(int(data.num_nodes), 1)
        ip, ip_session, hgmm, hgmm_rf = read_csv_metadata(csv_path)
        if len(hgmm) != data.num_nodes:
            raise ValueError(f"CSV/graph node mismatch for {csv_path.name}")
        records.append(Record(len(records), csv_path.name, ip, ip_session, data, hgmm, hgmm_rf))
        seen.add(stem)
    missing = set(csv_by_stem) - seen
    if missing:
        raise ValueError(f"Missing graphs for {len(missing)} trajectories")
    return records


def subset_stats(records: list[Record], ids: np.ndarray | list[int]) -> dict:
    chosen = [records[int(i)] for i in ids]
    positives = sum(r.n_positive for r in chosen)
    nodes = sum(r.n_nodes for r in chosen)
    return {
        "trajectories": len(chosen),
        "ips": len({r.ip for r in chosen}),
        "nodes": nodes,
        "positives": positives,
        "negatives": nodes - positives,
        "positive_rate": positives / nodes,
    }


def choose_group_split(records: list[Record], attempts: int = 3000) -> dict[str, np.ndarray]:
    indices = np.arange(len(records))
    groups = np.array([r.ip for r in records])
    overall_rate = sum(r.n_positive for r in records) / sum(r.n_nodes for r in records)
    best = None
    best_score = math.inf
    outer = GroupShuffleSplit(n_splits=attempts, test_size=TEST_RATIO, random_state=SEED)
    for attempt, (dev_ids, test_ids) in enumerate(outer.split(indices, groups=groups)):
        dev_groups = groups[dev_ids]
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=SEED + 1000 + attempt,
        )
        train_rel, val_rel = next(inner.split(dev_ids, groups=dev_groups))
        train_ids, val_ids = dev_ids[train_rel], dev_ids[val_rel]
        candidate = {"train": train_ids, "val": val_ids, "test": test_ids}
        stats = {name: subset_stats(records, ids) for name, ids in candidate.items()}
        ratios = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
        score = 0.0
        for name in candidate:
            score += 8 * abs(stats[name]["trajectories"] / len(records) - ratios[name])
            score += 3 * abs(stats[name]["nodes"] / sum(r.n_nodes for r in records) - ratios[name])
            score += 30 * abs(stats[name]["positive_rate"] - overall_rate)
        if score < best_score:
            best_score, best = score, candidate
    assert best is not None
    split_ips = {name: {records[int(i)].ip for i in ids} for name, ids in best.items()}
    assert not (split_ips["train"] & split_ips["val"])
    assert not (split_ips["train"] & split_ips["test"])
    assert not (split_ips["val"] & split_ips["test"])
    return best


def make_loader(records: list[Record], train: bool, seed: int) -> DataLoader:
    dataset = [r.data for r in records]
    if not train:
        return DataLoader(dataset, batch_size=1, shuffle=False)
    weights = []
    for record in records:
        weights.append(record.n_positive * 12 + (record.n_positive / record.n_nodes) * 8 + 1.0)
    weights = torch.tensor(weights, dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=1, sampler=sampler)


def label_alpha(records: list[Record], device: torch.device) -> torch.Tensor:
    positives = sum(r.n_positive for r in records)
    negatives = sum(r.n_nodes - r.n_positive for r in records)
    alpha = torch.tensor([1 / negatives, 1 / positives], dtype=torch.float32, device=device)
    return alpha / alpha.sum() * 2


@torch.no_grad()
def predict_records(model: nn.Module, records: list[Record], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for record in records:
        data = record.data.to(device)
        prob = F.softmax(model(data), dim=1)[:, 1]
        probs.append(prob.cpu().numpy())
        labels.append(data.y.cpu().numpy())
        record.data = data.cpu()
    return np.concatenate(probs), np.concatenate(labels)


def train_stage1(
    train_records: list[Record],
    val_records: list[Record],
    input_dim: int,
    max_layer: int,
    device: torch.device,
    seed: int,
    max_epochs: int,
    tag: str,
) -> tuple[nn.Module, dict]:
    seed_everything(seed)
    model = EdgeTypeAwareGAT(
        input_dim=input_dim,
        layer_emb_dim=LAYER_EMB_DIM,
        max_layer=max_layer,
        hidden_dim=HIDDEN_DIM,
        num_heads=HEADS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    loss_fn = FocalLoss(label_alpha(train_records, device), gamma=1.0)
    train_loader = make_loader(train_records, train=True, seed=seed)
    best_auc = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_probs, val_labels = predict_records(model, val_records, device)
        val_auc = average_precision_score(val_labels, val_probs)
        scheduler.step(val_auc)
        if val_auc > best_auc + 1e-8:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{tag}] epoch={epoch:03d} val_auc_pr={val_auc:.6f} best={best_auc:.6f}", flush=True)
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"No valid model state for {tag}")
    model.load_state_dict(best_state)
    info = {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_val_auc_pr": best_auc,
        "epochs_run": epoch,
        "seconds": time.time() - started,
    }
    print(f"[{tag}] done: {json.dumps(info)}", flush=True)
    return model, info


def select_threshold(probs: np.ndarray, labels: np.ndarray, target_recall: float | None = None) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]
    valid = np.ones(len(thresholds), dtype=bool)
    if target_recall is not None:
        valid &= recall >= target_recall
    if not valid.any():
        valid = recall >= recall.max()
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    score = np.where(valid, f1, -1)
    return float(thresholds[int(np.argmax(score))])


def metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "auc_roc": float(roc_auc_score(labels, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(labels)),
    }


def node_veto_features(data, probs: torch.Tensor, candidate_ids: np.ndarray) -> np.ndarray:
    edge_index = data.edge_index.detach().cpu().numpy()
    p = probs.detach().cpu().numpy()
    src, dst = edge_index
    result = np.zeros((len(candidate_ids), 3), dtype=np.float32)
    for row, node_id in enumerate(candidate_ids):
        nbrs = np.unique(np.concatenate([dst[src == node_id], src[dst == node_id]]))
        if len(nbrs) == 0:
            continue
        nbr_probs = p[nbrs]
        result[row] = [
            np.mean(nbr_probs > 0.5),
            p[node_id] - np.mean(nbr_probs),
            p[node_id] - np.max(nbr_probs),
        ]
    return result


@torch.no_grad()
def extract_features(
    model: nn.Module,
    records: list[Record],
    device: torch.device,
    stage1_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    model.eval()
    feature_batches, label_batches, refs = [], [], []
    for record in records:
        data = record.data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=1)[:, 1]
        candidate_ids = torch.where(probs >= stage1_threshold)[0].cpu().numpy()
        if len(candidate_ids):
            embedding = model.get_embedding(data)[candidate_ids].cpu().numpy().astype(np.float32)
            layer_embedding = model.get_layer_embedding(data.layer)[candidate_ids].cpu().numpy().astype(np.float32)
            raw = data.x[candidate_ids].cpu().numpy().astype(np.float32)
            trajectory_placeholders = np.zeros((len(candidate_ids), 8), dtype=np.float32)
            stage1_prob = probs[candidate_ids].cpu().numpy().astype(np.float32)[:, None]
            raw_layer = data.layer[candidate_ids].cpu().numpy().astype(np.float32)[:, None]
            veto = node_veto_features(data, probs, candidate_ids)
            fused = np.concatenate(
                [raw, embedding, trajectory_placeholders, stage1_prob, raw_layer, layer_embedding, veto], axis=1
            ).astype(np.float32, copy=False)
            feature_batches.append(fused)
            label_batches.append(data.y[candidate_ids].cpu().numpy().astype(np.int8))
            refs.extend((record.index, int(node_id)) for node_id in candidate_ids)
        record.data = data.cpu()
    if not feature_batches:
        raise RuntimeError("Stage I retained no candidate nodes")
    return np.concatenate(feature_batches), np.concatenate(label_batches), refs


def fit_stage2(train_x, train_y, val_x, val_y, seed: int, tag: str):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x).astype(np.float32)
    val_scaled = scaler.transform(val_x).astype(np.float32)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=float(np.sum(train_y == 0) / max(np.sum(train_y == 1), 1)),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_rounds=20,
    )
    model.fit(train_scaled, train_y, eval_set=[(val_scaled, val_y)], verbose=False)
    print(f"[{tag}] XGBoost best_iteration={model.best_iteration}", flush=True)
    return model, scaler


def all_node_stage2_prob(
    records: list[Record], refs: list[tuple[int, int]], candidate_probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    by_record = {r.index: np.zeros(r.n_nodes, dtype=np.float32) for r in records}
    for (record_index, node_index), prob in zip(refs, candidate_probs):
        by_record[record_index][node_index] = prob
    probs = np.concatenate([by_record[r.index] for r in records])
    labels = np.concatenate([r.data.y.numpy() for r in records])
    return probs, labels


def baseline_metrics(records: list[Record]) -> dict:
    labels = np.concatenate([r.data.y.numpy() for r in records])
    hgmm = np.concatenate([r.hgmm for r in records]).astype(np.float32)
    hgmm_rf = np.concatenate([r.hgmm_rf for r in records]).astype(np.float32)
    return {"HGMM": metrics(labels, hgmm, 0.5), "HGMM_RF": metrics(labels, hgmm_rf, 0.5)}


def main() -> None:
    global SOURCE, DATA_DIR, GRAPH_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    SOURCE, DATA_DIR, GRAPH_DIR, OUTPUT_DIR = (
        args.source,
        args.data_dir,
        args.graph_dir,
        args.output,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)

    records = load_records()
    split = choose_group_split(records)
    split_records = {name: [records[int(i)] for i in ids] for name, ids in split.items()}
    split_stats = {name: subset_stats(records, ids) for name, ids in split.items()}
    print("group split=" + json.dumps(split_stats, ensure_ascii=False), flush=True)
    input_dim = int(records[0].data.x.shape[1])
    max_layer = max(int(r.data.layer.max().item()) for r in records)

    manifest = []
    for split_name, ids in split.items():
        for index in ids:
            record = records[int(index)]
            manifest.append({"file": record.file, "IP": record.ip, "IP_session": record.ip_session, "split": split_name})
    pd.DataFrame(manifest).sort_values(["split", "file"]).to_csv(
        OUTPUT_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )

    # Final Stage I: full grouped training set, validation-only early stopping/threshold selection.
    final_model, final_info = train_stage1(
        split_records["train"], split_records["val"], input_dim, max_layer, device,
        SEED, args.max_epochs, "stage1-final",
    )
    val_prob, val_y_all = predict_records(final_model, split_records["val"], device)
    stage1_threshold = select_threshold(val_prob, val_y_all, TARGET_RECALL)
    test_prob, test_y_all = predict_records(final_model, split_records["test"], device)
    stage1_metrics = metrics(test_y_all, test_prob, stage1_threshold)
    print(f"stage1_threshold={stage1_threshold:.8f} test={json.dumps(stage1_metrics)}", flush=True)
    torch.save(final_model.state_dict(), OUTPUT_DIR / "stage1_final.pt")

    # Validation/test features are generated only by the final Stage-I model.
    val_x, val_y, val_refs = extract_features(final_model, split_records["val"], device, stage1_threshold)
    test_x, test_y, test_refs = extract_features(final_model, split_records["test"], device, stage1_threshold)

    # Diagnostic leaky Stage II on the same grouped split, isolating only the OOF correction.
    leaky_x, leaky_y, _ = extract_features(final_model, split_records["train"], device, stage1_threshold)
    leaky_model, leaky_scaler = fit_stage2(leaky_x, leaky_y, val_x, val_y, SEED, "stage2-leaky-control")
    leaky_val_candidate = leaky_model.predict_proba(leaky_scaler.transform(val_x).astype(np.float32))[:, 1]
    leaky_test_candidate = leaky_model.predict_proba(leaky_scaler.transform(test_x).astype(np.float32))[:, 1]
    leaky_val_all, leaky_val_labels = all_node_stage2_prob(split_records["val"], val_refs, leaky_val_candidate)
    leaky_test_all, leaky_test_labels = all_node_stage2_prob(split_records["test"], test_refs, leaky_test_candidate)
    leaky_threshold = select_threshold(leaky_val_all, leaky_val_labels)
    leaky_metrics = {
        "fixed_0.5": metrics(leaky_test_labels, leaky_test_all, 0.5),
        "validation_selected": metrics(leaky_test_labels, leaky_test_all, leaky_threshold),
    }
    del leaky_x

    # OOF Stage-II training features: every held-out IP/trajectory is predicted by a Stage-I
    # model that never saw that IP or trajectory during fitting.
    train_records = split_records["train"]
    groups = np.array([r.ip for r in train_records])
    indices = np.arange(len(train_records))
    try:
        fold_splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    except TypeError:
        fold_splitter = GroupKFold(n_splits=args.folds)
    oof_x_batches, oof_y_batches = [], []
    fold_infos = []
    for fold, (fit_ids, hold_ids) in enumerate(fold_splitter.split(indices, groups=groups), 1):
        fit_records = [train_records[int(i)] for i in fit_ids]
        hold_records = [train_records[int(i)] for i in hold_ids]
        assert not ({r.ip for r in fit_records} & {r.ip for r in hold_records})
        fold_model, fold_info = train_stage1(
            fit_records, split_records["val"], input_dim, max_layer, device,
            SEED + fold, args.max_epochs, f"stage1-oof-{fold}",
        )
        fold_x, fold_y, _ = extract_features(fold_model, hold_records, device, stage1_threshold)
        oof_x_batches.append(fold_x)
        oof_y_batches.append(fold_y)
        fold_info.update({"fit_trajectories": len(fit_records), "holdout_trajectories": len(hold_records)})
        fold_infos.append(fold_info)
        del fold_model, fold_x, fold_y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    oof_x = np.concatenate(oof_x_batches)
    oof_y = np.concatenate(oof_y_batches)

    corrected_model, corrected_scaler = fit_stage2(oof_x, oof_y, val_x, val_y, SEED, "stage2-oof")
    corrected_val_candidate = corrected_model.predict_proba(
        corrected_scaler.transform(val_x).astype(np.float32)
    )[:, 1]
    corrected_test_candidate = corrected_model.predict_proba(
        corrected_scaler.transform(test_x).astype(np.float32)
    )[:, 1]
    corrected_val_all, corrected_val_labels = all_node_stage2_prob(
        split_records["val"], val_refs, corrected_val_candidate
    )
    corrected_test_all, corrected_test_labels = all_node_stage2_prob(
        split_records["test"], test_refs, corrected_test_candidate
    )
    corrected_threshold = select_threshold(corrected_val_all, corrected_val_labels)
    corrected_metrics = {
        "fixed_0.5": metrics(corrected_test_labels, corrected_test_all, 0.5),
        "validation_selected": metrics(corrected_test_labels, corrected_test_all, corrected_threshold),
    }

    joblib.dump(corrected_model, OUTPUT_DIR / "stage2_oof_xgb.joblib")
    joblib.dump(corrected_scaler, OUTPUT_DIR / "stage2_oof_scaler.joblib")
    np.save(OUTPUT_DIR / "test_prob_stage1.npy", test_prob)
    np.save(OUTPUT_DIR / "test_prob_stage2_oof.npy", corrected_test_all)
    np.save(OUTPUT_DIR / "test_labels.npy", corrected_test_labels)

    report = {
        "configuration": {
            "seed": SEED,
            "folds": args.folds,
            "max_epochs": args.max_epochs,
            "patience": PATIENCE,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "stage1_threshold_validation": stage1_threshold,
            "stage2_threshold_validation_leaky": leaky_threshold,
            "stage2_threshold_validation_oof": corrected_threshold,
        },
        "split": split_stats,
        "stage1_training": final_info,
        "oof_training": fold_infos,
        "candidate_counts": {
            "leaky_train": int(len(leaky_y)),
            "oof_train": int(len(oof_y)),
            "validation": int(len(val_y)),
            "test": int(len(test_y)),
        },
        "grouped_test_baselines": baseline_metrics(split_records["test"]),
        "stage1_grouped_test": stage1_metrics,
        "stage2_grouped_leaky_control": leaky_metrics,
        "stage2_grouped_oof_corrected": corrected_metrics,
        "paper_reported": {
            "HGMM": {"precision": 0.1168, "recall": 0.6713, "f1": 0.1990, "auc_pr": 0.1042},
            "HGMM_RF": {"precision": 0.1109, "recall": 0.8292, "f1": 0.1956, "auc_pr": 0.1053},
            "MSPGL_Stage_I": {"precision": 0.3490, "recall": 0.8471, "f1": 0.4944, "auc_pr": 0.5237},
            "MSPGL_Full": {"precision": 0.6166, "recall": 0.6929, "f1": 0.6525, "auc_pr": 0.6838},
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for name, item in {
        "paper_MSPGL_full": report["paper_reported"]["MSPGL_Full"],
        "grouped_leaky_fixed_0.5": leaky_metrics["fixed_0.5"],
        "grouped_leaky_val_threshold": leaky_metrics["validation_selected"],
        "grouped_oof_fixed_0.5": corrected_metrics["fixed_0.5"],
        "grouped_oof_val_threshold": corrected_metrics["validation_selected"],
    }.items():
        rows.append({"experiment": name, **{k: item.get(k) for k in ("precision", "recall", "f1", "auc_pr")}})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    print("FINAL_REPORT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
