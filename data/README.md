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
| `x`, `y` | Screen or tile-grid coordinates |
| `lon`, `lat` | Geographic coordinates |
| `time` | Timestamp or time-order variable |
| `level` / `layer` | Map zoom or pyramid level |
| `session`, `ipsession`, `IP_session` | Session-related identifiers |
| `label` | Point-level target label used for supervised learning |

The notebook expects binary target labels represented as `Y` and `N` in the relevant label fields.

## Privacy

Before uploading data to GitHub, remove or anonymize any sensitive user, session, or platform identifiers. If the complete PMSP logs cannot be shared, keep only a small anonymized sample and document how qualified reviewers can request the full data.

