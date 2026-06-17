# Data Notes

Raw virtual trajectory CSV files should be placed in:

```text
data/raw_csv/
```

Each CSV corresponds to one virtual trajectory or one processing unit used by the MSPGL notebook. The sample file follows the expected schema used by `MSPGL.ipynb`.

## Important Columns

| Column | Description |
| --- | --- |
| `ID` | Point identifier within the trajectory file |
| `time` | Timestamp or time-order variable |
| `level` | Map zoom level |
| `layer` | Pyramid layer used for graph construction |
| `row`, `col` | Tile-grid row and column coordinates |
| `lon`, `lat` | Geographic coordinates |
| `HGMM`, `HGMM_RF` | Baseline method labels included for comparison |
| `MSPGL_label` | Point-level target label used by MSPGL |

The notebook expects binary MSPGL labels represented as `Y` and `N` in `MSPGL_label`. The code also keeps compatibility with older CSV files that use a `label` column.

