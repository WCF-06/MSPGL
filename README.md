# MSPGL

Multi-Stage Pyramid Graph Learning (MSPGL) for point-level browsing target identification in virtual trajectories.

This repository provides the reviewer-facing Jupyter implementation for the manuscript:

**MSPGL: Multi-Stage Pyramid Graph Learning for Point-level Browsing Target Identification in Virtual Trajectories**  


The method represents each virtual trajectory as a directed multi-relational graph that encodes spatial adjacency, temporal continuity, and cross-level scale relations. The pipeline then uses a two-stage design:

1. **Stage I:** relation-aware graph attention learning for high-recall candidate generation.
2. **Stage II:** multi-source feature fusion and XGBoost-based refinement for higher-precision target identification.

The submitted manuscript reports an AUC-PR of 0.6838 on real Public Map Service Platform logs. In the two-stage setting, Stage I reaches Recall 0.8471 and Precision 0.3490, while Stage II improves Precision to 0.6166 and keeps Recall at 0.6929.

## Repository Contents

```text
.
|-- MSPGL.ipynb              # Main Jupyter notebook for reproduction and review
|-- MSPGL.py                 # Python export of the notebook
|-- requirements.txt         # Python package requirements
|-- MSPGL model/             # Released trained model parameters and Stage-II feature artifacts
|-- data/
|   |-- README.md            # Input data schema and path notes
|   `-- raw_csv/             # Anonymized sample trajectory CSV
`-- outputs/                 # Generated at runtime and ignored by Git
```

## Environment

The experiments were run on the following device configuration:

| Item | Value |
| --- | --- |
| Main device | CUDA |
| GPU | NVIDIA GeForce RTX 4080 |
| GPU count | 1 |
| GPU memory | 16.84 GB |
| CPU cores | 28 |
| Total memory | 33.45 GB |
| PyTorch | 2.4.0+cu124 |
| CUDA | 12.4 |
| Platform | Linux-6.8.0-49-generic-x86_64-with-glibc2.35 |

Python 3.10 or 3.11 is recommended.

## Installation

Create and activate a clean environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA acceleration, install the PyTorch build that matches your local CUDA driver first. The reported environment used PyTorch `2.4.0+cu124`.

PyTorch Geometric may require version-specific wheels depending on the installed PyTorch and CUDA versions. See the official PyTorch Geometric installation guide if `torch-geometric` or its optional extensions fail to install.

## Data

The original training logs contain sensitive Public Map Service Platform user interaction records and are not released in this repository.

An anonymized sample CSV is provided only to show the expected input format:

```text
data/raw_csv/sample trajectory.csv
```

Users who have authorized trajectory data can place raw CSV files in `data/raw_csv/` and run the graph-construction and training sections. Without access to the private training data, reviewers can use the released model artifacts for inference and interpretability checks.

## Released Model Artifacts

Because the training data cannot be publicly shared, this repository includes trained MSPGL model artifacts in:

```text
MSPGL model/
```

The folder contains:

| File | Purpose |
| --- | --- |
| `best_model_layer_aware.pt` | Stage-I relation-aware graph attention model weights |
| `xgb_phase2.model` | Stage-II XGBoost refinement model |
| `scaler.pkl` | Feature scaler used by Stage II |
| `model_config.pt` | Saved model and feature configuration |
| `train_X_phase2.npy` | Saved Stage-II training feature matrix for SHAP background/reference use |
| `val_X_phase2.npy` | Saved Stage-II validation feature matrix |
| `test_X_phase2.npy` | Saved Stage-II test feature matrix |
| `test_y_phase2.npy` | Saved Stage-II test labels |

The notebook's original training pipeline writes model artifacts to `./outputs/model`. To use the released model artifacts directly, either copy the contents of `MSPGL model/` to `outputs/model/`, or update the notebook path variables `MODEL_DIR` and `MODEL_INPUT_DIR` to `./MSPGL model`.

## Running the Notebook

Open and run:

```text
MSPGL.ipynb
```

The notebook is organized into the following parts:

1. Build graph-structured data and visualize it.
2. Train the MSPGL two-stage model.
3. Extract target points using the trained MSPGL model.
4. Calculate class-imbalance statistics.
5. Run SHAP-based model interpretability analysis.

If the private training data are unavailable, skip the full training section and use the released files in `MSPGL model/` for model loading, inference, and SHAP analysis.

Default input and output paths:

| Purpose | Path |
| --- | --- |
| Raw CSV input | `./data/raw_csv` |
| PyTorch Geometric graphs | `./data/pyg_graphs` |
| Graph visualization output | `./outputs/graph_visualization` |
| Trained model artifacts | `./outputs/model` |
| Released model artifacts | `./MSPGL model` |
| Inference results | `./outputs/inference` |
| SHAP results | `./outputs/shap` |
| Imbalance statistics | `./outputs/imbalance` |

Generated runtime outputs are ignored by Git. Released model files under `MSPGL model/` are intended to be tracked with Git LFS because some files exceed normal GitHub file-size limits.

## Notes for GitHub Submission

- Submit `MSPGL.ipynb` as the primary reproducible artifact.
- Keep `MSPGL.py` as an optional plain-Python export for reviewers who prefer scripts.
- Use Git LFS for `MSPGL model/` before pushing to GitHub.
- Do not upload private or non-anonymized PMSP logs.

## Citation

If you use this code, please cite the associated manuscript once it is accepted or publicly available.
