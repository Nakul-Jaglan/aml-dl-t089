"""CI validates portable structure without downloading data or retraining."""
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
for path in (root / "notebooks").glob("*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    if errors:
        raise ValueError(f"Saved notebook has execution errors: {path}")
    print(f"Valid notebook: {path.name} ({len(notebook.cells)} cells)")
