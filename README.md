# MSPGL

Official implementation of **Multi-Stage Pyramid Graph Learning for Point-level Browsing Target Identification in Virtual Trajectories**.

MSPGL represents each virtual trajectory as a directed multi-relational graph. Stage I uses relation-aware graph attention to generate a recall-oriented candidate set. Stage II refines those candidates with XGBoost using the out-of-fold Stage I probability and the ten node attributes listed in Table 1 of the manuscript.

## What is included

```text
.
|-- MSPGL.ipynb                   # Original notebook and inference workflow
|-- MSPGL.py                      # Python export used by the reproducibility scripts
|-- prepare_graphs.py             # Build the released undirected graph artifacts
|-- make_directed_graphs.py       # Convert them to forward-time directed graphs
|-- corrected_experiment.py       # Single-seed leakage-control diagnostic
|-- supplementary_experiments.py # Multi-seed Stage-I and ablation utilities
|-- manuscript_experiments.py    # Manuscript Stage II and Table 1 analysis
|-- summarize_manuscript_results.py # Final tables and bootstrap intervals
|-- audit_dataset.py              # Group, duplicate, and label audit
|-- data/raw_csv/                 # De-identified schema example
`-- MSPGL model/                  # Previously released model artifacts (Git LFS)
```

The notebook and previously released model files remain available for compatibility. The command-line scripts implement the revised leakage-controlled protocol.

## Revised evaluation protocol

- Train, validation, and test trajectories are disjoint by pseudonymized `IP` group.
- Stage II training uses five-fold group-wise out-of-fold Stage I probabilities.
- Graph edges point forward in time; equal-time pairs are removed.
- Graph construction uses a spatial threshold of `55.33`, a temporal threshold of 3 seconds, and the three manuscript relation types: same-level, upward, and downward.
- The trajectory-index feature is excluded from model inputs.
- Validation data select early stopping and decision thresholds; the test split is used only for final evaluation.
- The complete procedure is repeated with seeds `42`, `7`, `19`, `73`, and `101`.
- Stage II uses exactly 11 features: `Stage1Prob` plus the ten raw attributes described in Table 1.

The reported five-seed test means are F1 `0.4556` for MSPGL Stage I and F1 `0.5048` for MSPGL Full. The scripts regenerate summary tables locally under `outputs/`; generated results and predictions are not tracked.

## Environment

Use Python 3.10 or 3.11. PyTorch 2.4.0 does not support Python 3.13.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CUDA driver before the remaining dependencies if GPU acceleration is required. CPU execution is supported but the full five-seed experiment is computationally expensive.

## Data

The server-log dataset is restricted and is not distributed. `data/raw_csv/sample trajectory.csv` is a de-identified schema example, not a replacement for the experimental dataset. Authorized users should place one trajectory per CSV in `data/raw_csv/`. Full training additionally requires pseudonymized `IP` and `IP_session` columns so group-disjoint splitting can be reproduced.

No raw IP address, access token, or direct user identifier should be committed. See [`data/README.md`](data/README.md) for the accepted column names.

## Reproduce the revised experiment

From the repository root:

```bash
python audit_dataset.py
python prepare_graphs.py
python make_directed_graphs.py
python manuscript_experiments.py
python summarize_manuscript_results.py
```

The first graph-building step preserves the released preprocessing logic. The second step enforces the manuscript's forward-time directed graph and removes the trajectory-index input. `manuscript_experiments.py` trains the Stage II specification used in the revision and also reports per-feature Table 1 analyses. `summarize_manuscript_results.py` refits the final 11-feature model from the cached OOF matrices and produces the multi-seed tables and trajectory-clustered bootstrap intervals.

`supplementary_experiments.py` remains available for the broader feature-group, graph, and hyperparameter diagnostics used during revision.

A small unit-test suite checks focal-loss stability and Stage II feature selection:

```bash
python -m unittest test_supplementary_experiments.py
```

For a short pipeline check on authorized data:

```bash
python supplementary_experiments.py --experiments core --seeds 42 --folds 2 --max-epochs 2 --bootstrap-replicates 100 --output outputs/smoke
python manuscript_experiments.py --seeds 42 --folds 2 --max-epochs 2 --output outputs/manuscript_smoke
```

Generated graphs, checkpoints, predictions, and reports are written below `outputs/` and ignored by Git.

## Released model artifacts

`MSPGL model/` contains the model files from the earlier public release and is tracked with Git LFS. These files support the original notebook workflow. They are not claimed to be the five independently trained models used to calculate the revised multi-seed summary.

## Result interpretation

Stage II is a precision-oriented candidate refiner. At validation-selected thresholds it raises precision and F1 relative to Stage I, while recall and AUC-PR decrease. The two stages therefore provide different operating characteristics rather than a uniform improvement across every metric.
