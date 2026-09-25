# Data layout

Place authorized input trajectories in `data/raw_csv/`, with one CSV file per trajectory. The public sample illustrates the non-identifying feature and label columns only; it is not the study dataset and is not sufficient for grouped training.

## Required columns

Graph construction accepts the released schema, including:

- `ID`: within-trajectory node order;
- `time`: interaction timestamp;
- `layer` (or the corresponding released level field);
- `row`, `col`, `lon`, and `lat`;
- manual label: `label` or `MSPGL_label`, encoded as `Y`/`N`;
- baseline labels: `leaves`/`HGMM` and `leaves_rf`/`HGMM_RF`, encoded as `Y`/`N`.

The leakage-controlled experiments also require:

- `IP`: a pseudonymized grouping identifier;
- `IP_session`: a pseudonymized session identifier.

Every trajectory associated with the same `IP` is kept in a single train, validation, or test subset. Do not use raw IP addresses or direct user identifiers.

## Annotation fields

The experimental CSVs contain one final manual-label column. They do not contain a second-annotator column, so independent inter-annotator agreement cannot be recomputed from the released files. The dataset audit reports manual-versus-baseline comparisons separately and does not present them as inter-annotator agreement.

## Generated files

`prepare_graphs.py` writes intermediate graphs to `outputs/graphs/`. `make_directed_graphs.py` converts them to the forward-time representation in `outputs/graphs-directed/`, which is consumed by the revised experiments.
