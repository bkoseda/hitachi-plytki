import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
results = []
for path in sorted((root / "notebooks").glob("*.ipynb")):
    print("RUN", path.name, flush=True)
    start = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=1200, kernel_name="pinglab", resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(nb, path)
    results.append(
        dict(
            notebook=path.name,
            seconds=round(time.perf_counter() - start, 2),
            status="passed",
            code_cells=sum(c.cell_type == "code" for c in nb.cells),
        )
    )
    print(results[-1], flush=True)
(root / "reports/notebook_validation.json").write_text(json.dumps(results, indent=2))
