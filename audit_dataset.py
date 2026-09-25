from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw_csv"
OUTPUT = ROOT / "outputs" / "dataset_audit.json"
SEED = 42


def norm(value: str) -> str:
    return (value or "").strip()


def label01(value: str) -> int:
    return int(norm(value).upper() == "Y")


def digest(parts) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit trajectory labels, grouping, and duplicates.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    files = list(args.data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir.resolve()}")

    trajectories = []
    exact_hash_to_files = defaultdict(list)
    window_hash_to_files = defaultdict(set)
    label_counts = Counter()
    baseline_agreement = Counter()
    baseline_positive_overlap = Counter()

    for file_index, path in enumerate(files):
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue

        ips = {norm(row.get("IP", "")) for row in rows}
        ip_sessions = {norm(row.get("IP_session", "")) for row in rows}
        sessions = {norm(row.get("session", "")) for row in rows}
        sequence = [
            (
                norm(row.get("time", "")),
                norm(row.get("layer", row.get("level", ""))),
                norm(row.get("row", "")),
                norm(row.get("col", "")),
                norm(row.get("lon", "")),
                norm(row.get("lat", "")),
            )
            for row in rows
        ]
        labels = [label01(row.get("label", row.get("MSPGL_label", ""))) for row in rows]
        hgmm = [label01(row.get("leaves", row.get("HGMM", ""))) for row in rows]
        hgmm_rf = [label01(row.get("leaves_rf", row.get("HGMM_RF", ""))) for row in rows]

        label_counts.update(labels)
        baseline_agreement["hgmm_equal"] += sum(a == b for a, b in zip(labels, hgmm))
        baseline_agreement["hgmm_rf_equal"] += sum(a == b for a, b in zip(labels, hgmm_rf))
        baseline_agreement["total"] += len(labels)
        baseline_positive_overlap["manual_positive"] += sum(labels)
        baseline_positive_overlap["manual_and_hgmm"] += sum(a and b for a, b in zip(labels, hgmm))
        baseline_positive_overlap["manual_and_hgmm_rf"] += sum(a and b for a, b in zip(labels, hgmm_rf))

        seq_hash = digest(sequence)
        exact_hash_to_files[seq_hash].append(path.name)
        for window_size in (3, 5, 10):
            if len(sequence) >= window_size:
                for start in range(len(sequence) - window_size + 1):
                    window_hash_to_files[(window_size, digest(sequence[start:start + window_size]))].add(file_index)

        trajectories.append(
            {
                "file": path.name,
                "n_rows": len(rows),
                "ips": sorted(ips),
                "ip_sessions": sorted(ip_sessions),
                "sessions": sorted(sessions),
                "sequence_hash": seq_hash,
                "positive": sum(labels),
                "negative": len(labels) - sum(labels),
            }
        )

    # Reproduce the released code's trajectory-level shuffle as closely as possible.
    indices = list(range(len(trajectories)))
    random.Random(SEED).shuffle(indices)
    n_total = len(indices)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    file_to_split = {index: split_name for split_name, ids in split_indices.items() for index in ids}

    ip_to_splits = defaultdict(set)
    ip_to_files = defaultdict(set)
    ip_session_to_splits = defaultdict(set)
    ip_session_to_files = defaultdict(set)
    for index, trajectory in enumerate(trajectories):
        split_name = file_to_split[index]
        for ip in trajectory["ips"]:
            ip_to_splits[ip].add(split_name)
            ip_to_files[ip].add(index)
        for key in trajectory["ip_sessions"]:
            ip_session_to_splits[key].add(split_name)
            ip_session_to_files[key].add(index)

    shared_windows = Counter()
    shared_window_examples = defaultdict(list)
    for (window_size, window_hash), file_ids in window_hash_to_files.items():
        splits = {file_to_split[index] for index in file_ids}
        if len(splits) > 1:
            shared_windows[window_size] += 1
            if len(shared_window_examples[window_size]) < 10:
                shared_window_examples[window_size].append(
                    {
                        "hash": window_hash,
                        "files": [trajectories[index]["file"] for index in sorted(file_ids)],
                        "splits": sorted(splits),
                    }
                )

    trajectories_per_ip = Counter(len(file_ids) for file_ids in ip_to_files.values())
    report = {
        "dataset": {
            "trajectory_files": len(trajectories),
            "total_nodes": sum(t["n_rows"] for t in trajectories),
            "positive_nodes": label_counts[1],
            "negative_nodes": label_counts[0],
            "unique_ips": len(ip_to_files),
            "unique_ip_sessions": len(ip_session_to_files),
            "files_with_multiple_ips": sum(len(t["ips"]) != 1 for t in trajectories),
            "files_with_multiple_ip_sessions": sum(len(t["ip_sessions"]) != 1 for t in trajectories),
            "trajectories_per_ip_histogram": dict(sorted(trajectories_per_ip.items())),
            "max_trajectories_per_ip": max(map(len, ip_to_files.values())),
        },
        "original_random_split": {
            "sizes": {name: len(ids) for name, ids in split_indices.items()},
            "ips_crossing_splits": sum(len(splits) > 1 for splits in ip_to_splits.values()),
            "ip_sessions_crossing_splits": sum(len(splits) > 1 for splits in ip_session_to_splits.values()),
            "trajectories_belonging_to_cross_split_ips": sum(
                1 for t in trajectories if any(len(ip_to_splits[ip]) > 1 for ip in t["ips"])
            ),
            "shared_exact_windows": {str(k): shared_windows[k] for k in (3, 5, 10)},
            "shared_exact_window_examples": {str(k): shared_window_examples[k] for k in (3, 5, 10)},
        },
        "duplicates": {
            "exact_duplicate_trajectory_groups": [files for files in exact_hash_to_files.values() if len(files) > 1],
            "reused_ip_session_groups": [
                [trajectories[index]["file"] for index in sorted(file_ids)]
                for file_ids in ip_session_to_files.values()
                if len(file_ids) > 1
            ],
        },
        "label_audit": {
            "hgmm_agreement": baseline_agreement["hgmm_equal"] / baseline_agreement["total"],
            "hgmm_rf_agreement": baseline_agreement["hgmm_rf_equal"] / baseline_agreement["total"],
            "manual_positive_count": baseline_positive_overlap["manual_positive"],
            "manual_positive_also_hgmm": baseline_positive_overlap["manual_and_hgmm"],
            "manual_positive_also_hgmm_rf": baseline_positive_overlap["manual_and_hgmm_rf"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
