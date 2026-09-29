"""Execute the client review notebook in a fresh kernel; keep private outputs in reports/runs."""

import copy
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from plotly.io import from_json, to_html

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "notebooks/client/01_wstepna_analiza.ipynb"
    out = ROOT / "reports/runs/client-20260929"
    out.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(source, 4)
    start = time.time()
    try:
        NotebookClient(
            nb, timeout=900, kernel_name="pinglab", resources={"metadata": {"path": str(source.parent)}}
        ).execute()
    except Exception:
        nbformat.write(nb, out / "analiza_FAILED.ipynb")
        raise
    nbformat.write(nb, out / "analiza.ipynb")
    rendered = copy.deepcopy(nb)
    for cell in rendered.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            mime = "application/vnd.plotly.v1+json"
            if mime in data:
                data["text/html"] = to_html(
                    from_json(json.dumps(data.pop(mime))), include_plotlyjs=True, full_html=False
                )
    html, _ = HTMLExporter().from_notebook_node(rendered)
    (out / "analiza.html").write_text(html)
    result = dict(
        status="passed",
        code_cells=sum(c.cell_type == "code" for c in nb.cells),
        seconds=round(time.time() - start, 1),
        fresh_kernel=True,
        outputs_have_errors=False,
    )
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
