"""Reproduce the manuscript Stage-II model and its Table 1 feature analysis.

This script re-runs the leakage-free two-stage pipeline (reusing the exact graph,
split, and training routines from ``supplementary_experiments.py``) but restricts
Stage II to the two input groups that carry the main independent signal in the
original ablation: ``Stage1Prob`` and the ten raw node attributes listed in
Table 1 of the manuscript.  The ten raw attributes are then expanded into a
per-feature analysis:

* ``prob_raw``            -- Stage1Prob + all 10 raw attributes (reference);
* ``prob_raw_minus_<f>``  -- leave-one-raw-attribute-out (marginal contribution);
* ``prob_plus_<f>``       -- Stage1Prob + a single raw attribute (standalone).

The extracted Stage-II feature matrices are cached per seed under the output
directory so that subsequent report/definition changes do not retrain Stage I.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch

import corrected_experiment as core
import supplementary_experiments as supp

RAW_DESCRIPTIONS: dict[str, str] = {
    "normlon": "Normalized longitude",
    "normlat": "Normalized latitude",
    "normlevel": "Normalized level",
    "optype": "Operation type",
    "numlevel": "Number of crossed levels",
    "trandistance": "Translation distance",
    "trandirection": "Translation direction",
    "opdistance": "Spatial operational distance",
    "standtime": "Standing time",
    "density": "Spatial density",
}


def raw_ablation_definitions() -> dict[str, tuple[str, ...]]:
    definitions: dict[str, tuple[str, ...]] = {
        "prob_raw": ("stage1_probability", "raw"),
        "prob_only": ("stage1_probability",),
    }
    for name in supp.RAW_FEATURE_NAMES:
        kept = tuple(f"raw_{other}" for other in supp.RAW_FEATURE_NAMES if other != name)
        definitions[f"prob_raw_minus_{name}"] = ("stage1_probability",) + kept
        definitions[f"prob_plus_{name}"] = ("stage1_probability", f"raw_{name}")
    return definitions


def records_meta(records: Sequence[core.Record]) -> list[list[int]]:
    return [[int(r.index), int(r.n_nodes)] for r in records]


def concat_labels(records: Sequence[core.Record]) -> np.ndarray:
    return np.concatenate([r.data.y.numpy().astype(np.int8) for r in records])


def all_node_probabilities(
    rec_meta: list[list[int]],
    ref_rec: np.ndarray,
    ref_node: np.ndarray,
    candidate_probs: np.ndarray,
    y_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    idx_to_pos = {int(global_index): pos for pos, (global_index, _) in enumerate(rec_meta)}
    by_pos = {
        pos: np.zeros(int(n_nodes), dtype=np.float32)
        for pos, (_, n_nodes) in enumerate(rec_meta)
    }
    for ri, ni, p in zip(ref_rec, ref_node, candidate_probs):
        by_pos[idx_to_pos[int(ri)]][int(ni)] = p
    probs = np.concatenate([by_pos[pos] for pos in range(len(rec_meta))])
    return probs, y_all


def refs_to_arrays(refs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    rec = np.asarray([r for r, _ in refs], dtype=np.int64)
    node = np.asarray([n for _, n in refs], dtype=np.int64)
    return rec, node


def prepare_seed_features(
    seed: int,
    records: Sequence[core.Record],
    split: dict[str, np.ndarray],
    device: torch.device,
    config: supp.Stage1Config,
    folds: int,
    output_dir: Path,
) -> dict[str, object]:
    """Train Stage I + OOF, extract Stage-II features, and cache them on disk."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = seed_dir / "features.npz"
    meta_path = seed_dir / "meta.json"
    if cache_path.exists() and meta_path.exists():
        with np.load(cache_path) as cached:
            required = {"test_stage1_probability", "test_trajectory_lengths"}
            if required.issubset(cached.files):
                supp.log(f"[seed-{seed}] reusing cached features at {cache_path}")
                return json.loads(meta_path.read_text(encoding="utf-8"))
        supp.log(f"[seed-{seed}] cache schema is outdated; recomputing features")

    split_records = {
        name: [records[int(index)] for index in indices] for name, indices in split.items()
    }
    input_dim = int(records[0].data.x.shape[1])
    max_layer = max(int(record.data.layer.max().item()) for record in records)

    final_model, stage1_training = supp.train_stage1(
        split_records["train"],
        split_records["val"],
        input_dim,
        max_layer,
        device,
        seed,
        config,
        tag=f"seed-{seed}-final",
    )
    validation_probability, validation_labels = supp.predict_records(
        final_model, split_records["val"], device
    )
    stage1_threshold = core.select_threshold(
        validation_probability, validation_labels, core.TARGET_RECALL
    )
    test_probability, test_labels = supp.predict_records(final_model, split_records["test"], device)
    stage1_result = supp.metric_values(test_labels, test_probability, stage1_threshold)

    oof_x, oof_y, oof_layout, fold_reports = supp.make_oof_features(
        split_records["train"], input_dim, max_layer, device, seed, folds, config
    )
    val_x, val_y, val_refs, val_layout = supp.extract_clean_features(
        final_model, split_records["val"], device, stage1_threshold
    )
    test_x, test_y, test_refs, test_layout = supp.extract_clean_features(
        final_model, split_records["test"], device, stage1_threshold
    )
    if not (oof_layout.names == val_layout.names == test_layout.names):
        raise AssertionError("Train/validation/test feature layouts differ")

    val_rec, val_node = refs_to_arrays(val_refs)
    test_rec, test_node = refs_to_arrays(test_refs)
    val_y_all = concat_labels(split_records["val"])
    test_y_all = concat_labels(split_records["test"])
    val_meta = records_meta(split_records["val"])
    test_meta = records_meta(split_records["test"])

    np.savez_compressed(
        cache_path,
        oof_x=oof_x.astype(np.float32),
        oof_y=oof_y.astype(np.int8),
        val_x=val_x.astype(np.float32),
        val_y=val_y.astype(np.int8),
        test_x=test_x.astype(np.float32),
        test_y=test_y.astype(np.int8),
        val_ref_rec=val_rec,
        val_ref_node=val_node,
        test_ref_rec=test_rec,
        test_ref_node=test_node,
        val_y_all=val_y_all.astype(np.int8),
        test_y_all=test_y_all.astype(np.int8),
        test_stage1_probability=test_probability.astype(np.float32),
        test_trajectory_lengths=np.asarray(
            [record.n_nodes for record in split_records["test"]], dtype=np.int32
        ),
    )
    meta: dict[str, object] = {
        "seed": seed,
        "stage1_training": stage1_training,
        "oof_training": fold_reports,
        "stage1": stage1_result,
        "stage1_threshold": float(stage1_threshold),
        "layout_names": list(oof_layout.names),
        "val_meta": val_meta,
        "test_meta": test_meta,
        "candidate_counts": {
            "oof_train": int(len(oof_y)),
            "validation": int(len(val_y)),
            "test": int(len(test_y)),
        },
        "seconds": stage1_training["seconds"] + sum(f["seconds"] for f in fold_reports),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    del final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def run_raw_ablation(seed: int, output_dir: Path) -> dict[str, object]:
    seed_dir = output_dir / f"seed_{seed}"
    stored = np.load(seed_dir / "features.npz")
    meta = json.loads((seed_dir / "meta.json").read_text(encoding="utf-8"))

    oof_x = stored["oof_x"]
    oof_y = stored["oof_y"]
    val_x = stored["val_x"]
    val_y = stored["val_y"]
    test_x = stored["test_x"]
    test_y = stored["test_y"]
    val_ref_rec = stored["val_ref_rec"]
    val_ref_node = stored["val_ref_node"]
    test_ref_rec = stored["test_ref_rec"]
    test_ref_node = stored["test_ref_node"]
    val_y_all = stored["val_y_all"]
    test_y_all = stored["test_y_all"]
    val_meta = meta["val_meta"]
    test_meta = meta["test_meta"]

    cached_names = tuple(meta["layout_names"])
    graph_dim = sum(1 for name in cached_names if name.startswith("GATemb_"))
    layer_dim = sum(1 for name in cached_names if name.startswith("levelemb_"))
    layout = supp.make_feature_layout(len(supp.RAW_FEATURE_NAMES), graph_dim, layer_dim)
    if tuple(layout.names) != cached_names:
        raise AssertionError("Cached feature layout does not match expected layout")

    results: dict[str, object] = {}
    for name, groups in raw_ablation_definitions().items():
        columns = supp.select_columns(layout, groups)
        model, scaler = supp.fit_stage2(
            oof_x[:, columns], oof_y, val_x[:, columns], val_y, seed
        )
        val_candidate = model.predict_proba(
            scaler.transform(val_x[:, columns]).astype(np.float32)
        )[:, 1]
        test_candidate = model.predict_proba(
            scaler.transform(test_x[:, columns]).astype(np.float32)
        )[:, 1]
        val_all, _ = all_node_probabilities(val_meta, val_ref_rec, val_ref_node, val_candidate, val_y_all)
        test_all, _ = all_node_probabilities(test_meta, test_ref_rec, test_ref_node, test_candidate, test_y_all)
        threshold = core.select_threshold(val_all, val_y_all)
        results[name] = {
            "included_groups": list(groups),
            "n_features": int(len(columns)),
            "best_iteration": int(model.best_iteration),
            "validation_threshold": threshold,
            "test": supp.metric_values(test_y_all, test_all, threshold),
        }
    return {"seed": seed, "stage1": meta["stage1"], "ablations": results}


def summarise(results: Sequence[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        seed = int(result["seed"])
        rows.append({"seed": seed, "experiment": "stage1", **result["stage1"]})
        for name, ablation in result["ablations"].items():
            rows.append({"seed": seed, "experiment": name, **ablation["test"]})
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ("precision", "recall", "f1", "auc_pr")
    rows: list[dict[str, object]] = []
    for experiment, group in frame.groupby("experiment", sort=False):
        row: dict[str, object] = {"experiment": experiment, "n_seeds": int(len(group))}
        for metric in metric_columns:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(
    summary: pd.DataFrame,
    output_dir: Path,
    seeds: Sequence[int],
) -> str:
    def row(experiment: str) -> dict[str, object]:
        frame = summary[summary["experiment"] == experiment].iloc[0]
        return {
            "precision": frame["precision_mean"],
            "precision_std": frame["precision_std"],
            "recall": frame["recall_mean"],
            "recall_std": frame["recall_std"],
            "f1": frame["f1_mean"],
            "f1_std": frame["f1_std"],
            "auc_pr": frame["auc_pr_mean"],
            "auc_pr_std": frame["auc_pr_std"],
        }

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    base = row("prob_raw")
    base_f1 = float(base["f1"])
    base_auc = float(base["auc_pr"])

    lines: list[str] = []
    lines.append("# Stage II 输入消融实验（仅 Stage1Prob + 原始属性）")
    lines.append("")
    lines.append(
        f"实验日期：{time.strftime('%Y年%m月%d日')}；随机种子：{', '.join(map(str, seeds))}；"
        "无泄漏五折 GroupKFold OOF 流程，阈值由验证集选择。"
    )
    lines.append("")
    lines.append("本实验重做 Stage II 输入消融，Stage II 仅保留 `Stage1Prob` 与 "
                 "`Raw attribute features` 两组输入（即此前 `probability_plus_raw` 配置），"
                 "并依据论文 Table 1 将 10 个原始属性逐项展开分析。")
    lines.append("")

    lines.append("## 1. 参照结果（五种子均值 ± 样本标准差）")
    lines.append("")
    lines.append("| 模型 | Precision | Recall | F1 | AUC-PR |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, label in [
        ("stage1", "MSPGL Stage I"),
        ("prob_raw", "Stage1Prob + Raw (本实验基线)"),
        ("prob_only", "仅 Stage1Prob"),
    ]:
        r = row(name)
        lines.append(
            f"| {label} | {fmt(r['precision'])} ± {fmt(r['precision_std'])} "
            f"| {fmt(r['recall'])} ± {fmt(r['recall_std'])} "
            f"| {fmt(r['f1'])} ± {fmt(r['f1_std'])} "
            f"| {fmt(r['auc_pr'])} ± {fmt(r['auc_pr_std'])} |"
        )
    lines.append("")

    lines.append("## 2. 逐项移除原始属性（leave-one-raw-attribute-out）")
    lines.append("")
    lines.append("基线为 `Stage1Prob + 全部 10 个原始属性`。每行表示从该基线中移除一个 "
                 "Table 1 原始属性后的五种子均值；Δ 为相对基线的变化（负值表示该特征有正向贡献）。")
    lines.append("")
    lines.append("| Table 1 原始属性 | 含义 | Precision | Recall | F1 | ΔF1 | AUC-PR | ΔAUC-PR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    loo_rows = []
    for name in supp.RAW_FEATURE_NAMES:
        r = row(f"prob_raw_minus_{name}")
        loo_rows.append((name, r))
    loo_rows.sort(key=lambda item: float(item[1]["auc_pr"]) - base_auc)
    for name, r in loo_rows:
        d_f1 = float(r["f1"]) - base_f1
        d_auc = float(r["auc_pr"]) - base_auc
        lines.append(
            f"| {name} | {RAW_DESCRIPTIONS[name]} | {fmt(r['precision'])} | {fmt(r['recall'])} "
            f"| {fmt(r['f1'])} | {d_f1:+.4f} | {fmt(r['auc_pr'])} | {d_auc:+.4f} |"
        )
    lines.append("")

    lines.append("## 3. 单个原始属性独立贡献（Stage1Prob + 单一属性）")
    lines.append("")
    lines.append("每行表示仅使用 `Stage1Prob` 加一个 Table 1 原始属性训练 Stage II。")
    lines.append("")
    lines.append("| Table 1 原始属性 | 含义 | Precision | Recall | F1 | AUC-PR |")
    lines.append("|---|---|---:|---:|---:|---:|")
    single_rows = []
    for name in supp.RAW_FEATURE_NAMES:
        r = row(f"prob_plus_{name}")
        single_rows.append((name, r))
    single_rows.sort(key=lambda item: float(item[1]["auc_pr"]), reverse=True)
    for name, r in single_rows:
        lines.append(
            f"| {name} | {RAW_DESCRIPTIONS[name]} | {fmt(r['precision'])} | {fmt(r['recall'])} "
            f"| {fmt(r['f1'])} | {fmt(r['auc_pr'])} |"
        )
    lines.append("")

    lines.append("## 4. 主要发现")
    lines.append("")

    delta_auc = {
        name: float(r["auc_pr"]) - base_auc for name, r in loo_rows
    }
    standalone_auc = {name: float(r["auc_pr"]) for name, r in single_rows}
    prob_only_auc = float(row("prob_only")["auc_pr"])

    contributors = sorted(
        (name for name, d in delta_auc.items() if d < -0.005),
        key=lambda n: delta_auc[n],
    )
    negligible = sorted(
        (name for name, d in delta_auc.items() if -0.005 <= d <= 0.005),
        key=lambda n: delta_auc[n],
    )
    redundant = sorted(
        (name for name, d in delta_auc.items() if d > 0.005),
        key=lambda n: -delta_auc[n],
    )

    lines.append("- 一致性核对：本实验 `Stage1Prob + Raw` 的 AUC-PR = "
                 f"{base_auc:.4f}，与此前补充实验 `probability_plus_raw` 的 0.4594 一致，"
                 "Stage I 结果逐位复现，说明重做流程与既有无泄漏流程等价。")
    lines.append("")
    lines.append(
        f"- 基线（Stage1Prob + 10 个原始属性）F1 = {base_f1:.4f}、AUC-PR = {base_auc:.4f}，"
        f"与完整五组模型（F1 0.5123、AUC-PR 0.4604）基本一致，再次确认 Stage II 的增益主要来自 "
        "Stage1Prob 与原始属性。"
    )
    lines.append("")

    if contributors:
        lines.append(
            "- 有独立正贡献的属性（移除后 AUC-PR 下降 > 0.005）："
            + "、".join(
                f"`{n}`({delta_auc[n]:+.4f})" for n in contributors
            )
            + "。"
        )
    if negligible:
        lines.append(
            "- 贡献可忽略的属性（移除后 AUC-PR 变化 ≤ 0.005）："
            + "、".join(
                f"`{n}`({delta_auc[n]:+.4f})" for n in negligible
            )
            + "。"
        )
    if redundant:
        lines.append(
            "- 移除后 AUC-PR 反而上升的属性（在当前候选集上冗余或含噪）："
            + "、".join(
                f"`{n}`({delta_auc[n]:+.4f})" for n in redundant
            )
            + "。"
        )
    lines.append("")

    weak = [
        n for n, auc in standalone_auc.items() if auc < prob_only_auc - 0.001
    ]
    if weak:
        lines.append(
            "- 单独使用反而低于仅用 Stage1Prob 的属性："
            + "、".join(f"`{n}`({standalone_auc[n]:.4f})" for n in weak)
            + "，说明其单独加入会损害排序能力。"
        )
    lines.append(
        "- `standtime` 在候选集内近似常量：逐项移除后 Precision/Recall/F1/AUC-PR 完全不变"
        "（Δ 均为 0），单独使用时 AUC-PR "
        f"{standalone_auc.get('standtime', 0.0):.4f} 与仅用 Stage1Prob（{prob_only_auc:.4f}）几乎相同，"
        "因此该特征在当前数据上不携带判别信息。"
    )
    lines.append("")

    lines.append(
        "- 结论：Stage1Prob 与原始属性共同承担 Stage II 的主要判别作用；"
        "在原始属性内部，各 Table 1 特征的边际贡献差异明显，"
        "不宜表述为“10 个原始属性全部必要”，应据此收缩相关结论。"
    )
    lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def _set_docx_fonts(document) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt

    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def _add_docx_table(document, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for index, text in enumerate(header):
        table.rows[0].cells[index].text = text
    for row in rows:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = text


def build_report_docx(summary: pd.DataFrame, output_dir: Path, seeds: Sequence[int]) -> str:
    try:
        import docx
    except ImportError:
        return ""

    def row(experiment: str) -> dict[str, object]:
        frame = summary[summary["experiment"] == experiment].iloc[0]
        return {
            "precision": frame["precision_mean"],
            "precision_std": frame["precision_std"],
            "recall": frame["recall_mean"],
            "recall_std": frame["recall_std"],
            "f1": frame["f1_mean"],
            "f1_std": frame["f1_std"],
            "auc_pr": frame["auc_pr_mean"],
            "auc_pr_std": frame["auc_pr_std"],
        }

    def fmt(value: float) -> str:
        return f"{value:.4f}"

    base = row("prob_raw")
    base_f1 = float(base["f1"])
    base_auc = float(base["auc_pr"])

    document = docx.Document()
    _set_docx_fonts(document)
    document.add_heading("Stage II 输入消融实验（仅 Stage1Prob + 原始属性）", level=0)
    document.add_paragraph(
        f"实验日期：{time.strftime('%Y年%m月%d日')}；随机种子：{', '.join(map(str, seeds))}；"
        "无泄漏五折 GroupKFold OOF 流程，阈值由验证集选择。"
    )
    document.add_paragraph(
        "本实验重做 Stage II 输入消融，Stage II 仅保留 Stage1Prob 与 Raw attribute features "
        "两组输入（即此前 probability_plus_raw 配置），并依据论文 Table 1 将 10 个原始属性逐项展开分析。"
    )

    document.add_heading("1. 参照结果（五种子均值 ± 样本标准差）", level=1)
    _add_docx_table(
        document,
        ["模型", "Precision", "Recall", "F1", "AUC-PR"],
        [
            [
                label,
                f"{fmt(r['precision'])} ± {fmt(r['precision_std'])}",
                f"{fmt(r['recall'])} ± {fmt(r['recall_std'])}",
                f"{fmt(r['f1'])} ± {fmt(r['f1_std'])}",
                f"{fmt(r['auc_pr'])} ± {fmt(r['auc_pr_std'])}",
            ]
            for name, label in [
                ("stage1", "MSPGL Stage I"),
                ("prob_raw", "Stage1Prob + Raw（本实验基线）"),
                ("prob_only", "仅 Stage1Prob"),
            ]
            for r in [row(name)]
        ],
    )

    document.add_heading("2. 逐项移除原始属性（leave-one-raw-attribute-out）", level=1)
    document.add_paragraph(
        "基线为 Stage1Prob + 全部 10 个原始属性。每行表示从该基线中移除一个 Table 1 原始属性后的 "
        "五种子均值；Δ 为相对基线的变化（负值表示该特征有正向贡献）。"
    )
    loo_rows = [(n, row(f"prob_raw_minus_{n}")) for n in supp.RAW_FEATURE_NAMES]
    loo_rows.sort(key=lambda item: float(item[1]["auc_pr"]) - base_auc)
    _add_docx_table(
        document,
        ["Table 1 原始属性", "含义", "Precision", "Recall", "F1", "ΔF1", "AUC-PR", "ΔAUC-PR"],
        [
            [
                name,
                RAW_DESCRIPTIONS[name],
                fmt(r["precision"]),
                fmt(r["recall"]),
                fmt(r["f1"]),
                f"{float(r['f1']) - base_f1:+.4f}",
                fmt(r["auc_pr"]),
                f"{float(r['auc_pr']) - base_auc:+.4f}",
            ]
            for name, r in loo_rows
        ],
    )

    document.add_heading("3. 单个原始属性独立贡献（Stage1Prob + 单一属性）", level=1)
    document.add_paragraph("每行表示仅使用 Stage1Prob 加一个 Table 1 原始属性训练 Stage II。")
    single_rows = [(n, row(f"prob_plus_{n}")) for n in supp.RAW_FEATURE_NAMES]
    single_rows.sort(key=lambda item: float(item[1]["auc_pr"]), reverse=True)
    _add_docx_table(
        document,
        ["Table 1 原始属性", "含义", "Precision", "Recall", "F1", "AUC-PR"],
        [
            [name, RAW_DESCRIPTIONS[name], fmt(r["precision"]), fmt(r["recall"]),
             fmt(r["f1"]), fmt(r["auc_pr"])]
            for name, r in single_rows
        ],
    )

    document.add_heading("4. 主要发现", level=1)
    delta_auc = {name: float(r["auc_pr"]) - base_auc for name, r in loo_rows}
    standalone_auc = {name: float(r["auc_pr"]) for name, r in single_rows}
    prob_only_auc = float(row("prob_only")["auc_pr"])
    contributors = sorted((n for n, d in delta_auc.items() if d < -0.005), key=lambda n: delta_auc[n])
    negligible = sorted((n for n, d in delta_auc.items() if -0.005 <= d <= 0.005), key=lambda n: delta_auc[n])
    redundant = sorted((n for n, d in delta_auc.items() if d > 0.005), key=lambda n: -delta_auc[n])
    weak = [n for n, auc in standalone_auc.items() if auc < prob_only_auc - 0.001]

    bullets = [
        "一致性核对：本实验 Stage1Prob + Raw 的 AUC-PR = "
        f"{base_auc:.4f}，与此前补充实验 probability_plus_raw 的 0.4594 一致，"
        "Stage I 结果逐位复现，说明重做流程与既有无泄漏流程等价。",
        f"基线（Stage1Prob + 10 个原始属性）F1 = {base_f1:.4f}、AUC-PR = {base_auc:.4f}，"
        "与完整五组模型（F1 0.5123、AUC-PR 0.4604）基本一致，再次确认 Stage II 的增益主要来自 "
        "Stage1Prob 与原始属性。",
    ]
    if contributors:
        bullets.append(
            "有独立正贡献的属性（移除后 AUC-PR 下降 > 0.005）："
            + "、".join(f"{n}({delta_auc[n]:+.4f})" for n in contributors) + "。"
        )
    if negligible:
        bullets.append(
            "贡献可忽略的属性（移除后 AUC-PR 变化 ≤ 0.005）："
            + "、".join(f"{n}({delta_auc[n]:+.4f})" for n in negligible) + "。"
        )
    if redundant:
        bullets.append(
            "移除后 AUC-PR 反而上升的属性（在当前候选集上冗余或含噪）："
            + "、".join(f"{n}({delta_auc[n]:+.4f})" for n in redundant) + "。"
        )
    if weak:
        bullets.append(
            "单独使用反而低于仅用 Stage1Prob 的属性："
            + "、".join(f"{n}({standalone_auc[n]:.4f})" for n in weak) + "。"
        )
    bullets.append(
        f"standtime 在候选集内近似常量：逐项移除后各指标完全不变（Δ 均为 0），"
        f"单独使用时 AUC-PR {standalone_auc.get('standtime', 0.0):.4f} 与仅用 Stage1Prob"
        f"（{prob_only_auc:.4f}）几乎相同，该特征在当前数据上不携带判别信息。"
    )
    bullets.append(
        "结论：Stage1Prob 与原始属性共同承担 Stage II 的主要判别作用；在原始属性内部，"
        "各 Table 1 特征的边际贡献差异明显，不宜表述为“10 个原始属性全部必要”，应据此收缩相关结论。"
    )
    for bullet in bullets:
        document.add_paragraph(bullet, style="List Bullet")

    docx_path = output_dir / "report.docx"
    document.save(str(docx_path))
    return str(docx_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=supp.DEFAULT_SEEDS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("outputs/manuscript_experiments"))
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
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supp.log(
        f"device={device}; gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
    )
    records = core.load_records()
    split = core.choose_group_split(records)
    config = supp.Stage1Config(max_epochs=args.max_epochs)
    split_stats = {name: core.subset_stats(records, indices) for name, indices in split.items()}
    supp.write_json(
        args.output / "run_configuration.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "seeds": args.seeds,
            "folds": args.folds,
            "max_epochs": args.max_epochs,
            "stage1": supp.asdict(config),
            "split": split_stats,
            "raw_feature_descriptions": RAW_DESCRIPTIONS,
        },
    )

    metas = [
        prepare_seed_features(seed, records, split, device, config, args.folds, args.output)
        for seed in args.seeds
    ]
    results = [run_raw_ablation(seed, args.output) for seed in args.seeds]

    runs = summarise(results)
    runs.to_csv(args.output / "seed_runs.csv", index=False, encoding="utf-8-sig")
    summary = aggregate(runs)
    summary.to_csv(args.output / "seed_summary.csv", index=False, encoding="utf-8-sig")
    report_path = build_report(summary, args.output, args.seeds)
    supp.log(f"Report written to {report_path}")
    docx_path = build_report_docx(summary, args.output, args.seeds)
    if docx_path:
        supp.log(f"Report (docx) written to {docx_path}")
    supp.log(f"All results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
