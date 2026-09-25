"""Build the graph artifacts with the repository's released preprocessing logic.

The wrapper executes only the graph-construction definitions from MSPGL.py.  It
deliberately skips the notebook's later training/inference top-level code and the
HTML visualization step, which is irrelevant to the corrected experiment.
"""

from __future__ import annotations

import ast
import argparse
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MSPGL.py"
INPUT = ROOT / "data" / "raw_csv"
OUTPUT = ROOT / "outputs" / "graphs"


def load_preprocessing_namespace() -> dict:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected = []
    for node in tree.body:
        if getattr(node, "lineno", 10**9) >= 1346:
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name for alias in node.names]
            if any(name == "plotly" or name.startswith("plotly.") for name in imported):
                continue
        selected.append(node)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"__name__": "mspgl_preprocessing"}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    namespace = load_preprocessing_namespace()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    namespace["visualize_graph"] = lambda *args, **kwargs: None
    # The released script logs an undefined local name after a successful save.
    # Supplying a harmless global keeps the original computation intact.
    namespace["pyg_path"] = str(args.output)
    namespace["vis_path"] = "visualization skipped"
    namespace["process_and_save_files"](
        input_dir=str(args.input),
        output_vis_dir="outputs/graph_visualization_skipped",
        output_pyg_dir=str(args.output),
        threshold=55.33,
        time_threshold=3,
        samelayer_penalty=1.0,
        window=3,
        max_same_layer=8,
        max_upper_layer=4,
        max_lower_layer=4,
        base_degree_threshold=16,
        layer_degree_bonus=2,
        knn_k=10,
    )


if __name__ == "__main__":
    main()
