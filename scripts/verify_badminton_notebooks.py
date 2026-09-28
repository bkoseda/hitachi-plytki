"""Execute every badminton notebook in a fresh kernel and export presentation HTML."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from plotly.io import from_json, to_html

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/runs/badminton-notebooks"
REPORT.mkdir(parents=True, exist_ok=True)


def html_ready(nb):
    rendered = copy.deepcopy(nb)
    for cell in rendered.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            mime = "application/vnd.plotly.v1+json"
            if mime in data:
                figure = from_json(json.dumps(data[mime]))
                data["text/html"] = to_html(figure, include_plotlyjs=True, full_html=False)
                del data[mime]
    return rendered


def run(path):
    nb = nbformat.read(path, 4)
    try:
        NotebookClient(
            nb, timeout=600, kernel_name="pinglab", resources={"metadata": {"path": str(path.parent)}}
        ).execute()
        nbformat.write(nb, path)
        html, _ = HTMLExporter().from_notebook_node(html_ready(nb))
        (REPORT / (path.stem + ".html")).write_text(html)
        n = sum(c.cell_type == "code" for c in nb.cells)
        result = dict(
            notebook=path.name,
            status="passed",
            code_cells=n,
            png_outputs=sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", [])),
            plotly_outputs=sum(
                "application/vnd.plotly.v1+json" in o.get("data", {})
                for c in nb.cells
                for o in c.get("outputs", [])
            ),
        )
    except Exception as exc:
        nbformat.write(nb, REPORT / (path.stem + "_failed.ipynb"))
        result = dict(notebook=path.name, status="failed", error=str(exc)[-5000:])
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, sorted((ROOT / "notebooks/badminton").glob("*.ipynb"))))
    (REPORT / "validation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)
