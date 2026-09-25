from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import corrected_experiment as core
import manuscript_experiments as raw_ablation
import supplementary_experiments as supp


FEATURE_SOURCE = ROOT / "outputs" / "manuscript_experiments"
STAGE1_SOURCE = FEATURE_SOURCE
OUTPUT = ROOT / "outputs" / "manuscript_results"
SEEDS = (42, 7, 19, 73, 101)
FEATURE_NAMES = ("Stage1Prob",) + supp.RAW_FEATURE_NAMES


def fit_one_seed(seed: int) -> dict[str, object]:
    source_dir = FEATURE_SOURCE / f"seed_{seed}"
    stored = np.load(source_dir / "features.npz")
    meta = json.loads((source_dir / "meta.json").read_text(encoding="utf-8"))
    cached_names = tuple(meta["layout_names"])
    graph_dim = sum(name.startswith("GATemb_") for name in cached_names)
    layer_dim = sum(name.startswith("levelemb_") for name in cached_names)
    layout = supp.make_feature_layout(len(supp.RAW_FEATURE_NAMES), graph_dim, layer_dim)
    columns = np.asarray(layout.stage1_probability + layout.raw, dtype=np.int64)

    oof_x = stored["oof_x"][:, columns]
    oof_y = stored["oof_y"]
    val_x = stored["val_x"][:, columns]
    val_y = stored["val_y"]
    test_x = stored["test_x"][:, columns]
    val_y_all = stored["val_y_all"]
    test_y_all = stored["test_y_all"]

    model, scaler = supp.fit_stage2(oof_x, oof_y, val_x, val_y, seed)
    val_candidate = model.predict_proba(scaler.transform(val_x).astype(np.float32))[:, 1]
    test_candidate = model.predict_proba(scaler.transform(test_x).astype(np.float32))[:, 1]
    val_all, _ = raw_ablation.all_node_probabilities(
        meta["val_meta"], stored["val_ref_rec"], stored["val_ref_node"], val_candidate, val_y_all
    )
    test_all, _ = raw_ablation.all_node_probabilities(
        meta["test_meta"], stored["test_ref_rec"], stored["test_ref_node"], test_candidate, test_y_all
    )
    threshold = core.select_threshold(val_all, val_y_all)
    metrics = supp.metric_values(test_y_all, test_all, threshold)

    target = OUTPUT / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / "stage2_simplified_11feature.joblib")
    joblib.dump(scaler, target / "stage2_simplified_11feature_scaler.joblib")

    trajectory_lengths = np.asarray([int(length) for _, length in meta["test_meta"]], dtype=np.int32)
    if not np.array_equal(stored["test_trajectory_lengths"], trajectory_lengths):
        raise AssertionError(f"Trajectory layout mismatch for seed {seed}")
    np.savez_compressed(
        target / "test_predictions.npz",
        labels=test_y_all.astype(np.int8),
        stage1_probability=stored["test_stage1_probability"].astype(np.float32),
        stage2_probability=test_all.astype(np.float32),
        stage1_threshold=np.float32(meta["stage1_threshold"]),
        stage2_threshold=np.float32(threshold),
        trajectory_lengths=trajectory_lengths,
        candidate_feature_values=test_x.astype(np.float32),
        candidate_probabilities=test_candidate.astype(np.float32),
        candidate_record_indices=stored["test_ref_rec"].astype(np.int64),
        candidate_node_indices=stored["test_ref_node"].astype(np.int64),
    )
    (target / "result.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "stage2_definition": "Stage1Prob plus the 10 Table 1 raw attributes",
                "feature_names": FEATURE_NAMES,
                "n_features": 11,
                "best_iteration": int(model.best_iteration),
                "validation_threshold": float(threshold),
                "test": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"seed": seed, "experiment": "stage2_simplified", **metrics}


def summarise(seed_runs: pd.DataFrame) -> pd.DataFrame:
    row: dict[str, object] = {"model": "MSPGL Full", "stage2_features": "11 simplified Table 1 features", "n_seeds": len(seed_runs)}
    for metric in ("precision", "recall", "f1", "auc_pr"):
        values = seed_runs[metric].astype(float)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1))
        row[f"{metric}_cv"] = float(values.std(ddof=1) / values.mean())
    return pd.DataFrame([row])


def draw_main_figure(summary: pd.DataFrame) -> None:
    full = summary.iloc[0]
    models = ["HGMM", "HGMM-RF", "MSPGL Stage I", "MSPGL Full"]
    means = np.array(
        [
            [0.1157, 0.7422, 0.2001, 0.1060],
            [0.1066, 0.7359, 0.1862, 0.0991],
            [0.3435, 0.6765, 0.4556, 0.4866],
            [full.precision_mean, full.recall_mean, full.f1_mean, full.auc_pr_mean],
        ],
        dtype=float,
    )
    errors = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [0.0059, 0.0083, 0.0063, 0.0056],
            [full.precision_std, full.recall_std, full.f1_std, full.auc_pr_std],
        ],
        dtype=float,
    )
    metrics = ["Precision", "Recall", "F1", "AUC-PR"]
    colors = ["#4C78A8", "#F58518", "#E3B505", "#54A24B"]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    x = np.arange(len(models))
    width = 0.17
    for index, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            means[:, index],
            width,
            color=color,
            label=metric,
            yerr=np.ma.masked_invalid(errors[:, index]),
            capsize=6,
            error_kw={"elinewidth": 1.8, "capthick": 1.8},
        )
        for model_index, (bar, value) in enumerate(zip(bars, means[:, index])):
            extra = 0.0 if np.isnan(errors[model_index, index]) else errors[model_index, index]
            ax.text(bar.get_x() + bar.get_width() / 2, value + extra + 0.012, f"{value:.3f}", ha="center", fontsize=15)
    ax.set_title("Performance comparison on the IP-disjoint test set", fontsize=25, fontweight="bold", pad=24)
    ax.set_ylabel("Metric value", fontsize=21, labelpad=14)
    ax.set_xticks(x, models, fontsize=18)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", color="#D7DEE8", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=17)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.22)
    fig.savefig(OUTPUT / "figure5_main_performance_simplified.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure5_main_performance_simplified.svg", bbox_inches="tight")
    plt.close(fig)


def draw_bootstrap(bootstrap: pd.DataFrame) -> None:
    metrics = ["precision", "recall", "f1", "auc_pr"]
    labels = ["Precision", "Recall", "F1", "AUC-PR"]
    colors = {"stage1": "#9E9E9E", "stage2_full": "#2C7FB8"}
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    y = np.arange(len(metrics))
    for offset, (experiment, label) in zip((-0.12, 0.12), (("stage1", "MSPGL Stage I"), ("stage2_full", "MSPGL Full"))):
        frame = bootstrap[bootstrap.experiment == experiment].set_index("metric").loc[metrics]
        mean = frame["mean"].to_numpy(float)
        low = mean - frame["ci_2.5%"].to_numpy(float)
        high = frame["ci_97.5%"].to_numpy(float) - mean
        ax.errorbar(mean, y + offset, xerr=np.vstack([low, high]), fmt="o", markersize=8, capsize=5, linewidth=2.0, color=colors[experiment], label=label)
    ax.set_yticks(y, labels, fontsize=17)
    ax.set_xlabel("Metric value with trajectory-level 95% CI", fontsize=19)
    ax.tick_params(axis="x", labelsize=16)
    # Keep the complete Stage I recall interval visible (upper bound is 0.760).
    ax.set_xlim(0.25, 0.78)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    ax.legend(fontsize=16, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figureS4_bootstrap_simplified.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "figureS4_bootstrap_simplified.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [fit_one_seed(seed) for seed in SEEDS]
    seed_runs = pd.DataFrame(rows)
    summary = summarise(seed_runs)
    seed_runs.to_csv(OUTPUT / "seed_runs_simplified.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "seed_summary_simplified.csv", index=False, encoding="utf-8-sig")

    bootstrap = supp.bootstrap_across_trajectories(OUTPUT, SEEDS, 1000)
    bootstrap.to_csv(OUTPUT / "trajectory_bootstrap_ci_simplified.csv", index=False, encoding="utf-8-sig")

    stage1 = pd.read_csv(STAGE1_SOURCE / "seed_summary.csv", encoding="utf-8-sig")
    stage1 = stage1.loc[stage1.experiment == "stage1"].iloc[0]
    table4 = pd.DataFrame(
        [
            {"Model": "HGMM", "Precision": "0.1157", "Recall": "0.7422", "F1": "0.2001", "AUC-PR": "0.1060"},
            {"Model": "HGMM-RF", "Precision": "0.1066", "Recall": "0.7359", "F1": "0.1862", "AUC-PR": "0.0991"},
            {"Model": "MSPGL Stage I", "Precision": f"{stage1.precision_mean:.4f} ± {stage1.precision_std:.4f}", "Recall": f"{stage1.recall_mean:.4f} ± {stage1.recall_std:.4f}", "F1": f"{stage1.f1_mean:.4f} ± {stage1.f1_std:.4f}", "AUC-PR": f"{stage1.auc_pr_mean:.4f} ± {stage1.auc_pr_std:.4f}"},
            {"Model": "MSPGL Full", "Precision": f"{summary.iloc[0].precision_mean:.4f} ± {summary.iloc[0].precision_std:.4f}", "Recall": f"{summary.iloc[0].recall_mean:.4f} ± {summary.iloc[0].recall_std:.4f}", "F1": f"{summary.iloc[0].f1_mean:.4f} ± {summary.iloc[0].f1_std:.4f}", "AUC-PR": f"{summary.iloc[0].auc_pr_mean:.4f} ± {summary.iloc[0].auc_pr_std:.4f}"},
        ]
    )
    table4.to_csv(OUTPUT / "table4_main_performance_simplified.csv", index=False, encoding="utf-8-sig")

    table6_rows = []
    for model, values in (("MSPGL Stage I", stage1), ("MSPGL Full", summary.iloc[0])):
        for metric, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("AUC-PR", "auc_pr")):
            mean = float(values[f"{key}_mean"])
            std = float(values[f"{key}_std"])
            table6_rows.append({"Model": model, "Metric": metric, "Mean": mean, "Standard deviation": std, "CV": std / mean})
    pd.DataFrame(table6_rows).to_csv(OUTPUT / "table6_seed_stability_simplified.csv", index=False, encoding="utf-8-sig")

    draw_main_figure(summary)
    draw_bootstrap(bootstrap)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
