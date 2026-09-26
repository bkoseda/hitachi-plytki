"""Run the delivered exploration from a different directory, without the original repo's data."""

import json
import tempfile
import zipfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="pinglab-portable-") as folder:
    with zipfile.ZipFile(root / "deliverables/Ping_Lab_start.zip") as archive:
        archive.extractall(folder)
    project = Path(folder) / "Ping_Lab"
    assert not (project / "data/demo/SYN_001_ok.wav").exists()
    nb = nbformat.read(project / "notebooks/00_eksploracja_do_przegladu.ipynb", as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell('import os; os.environ.pop("PINGLAB_SETTINGS",None)'))
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import pinglab; assert Path(pinglab.__file__).is_relative_to(ROOT), pinglab.__file__"
        )
    )
    NotebookClient(
        nb, timeout=1200, kernel_name="pinglab", resources={"metadata": {"path": str(project)}}
    ).execute()
    assert (project / "reports/runs/demo-session/exploration/leaderboard.json").exists()
    count = sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", []))
    assert count >= 9
    report = dict(
        status="passed",
        dataset_generated_from_scratch=True,
        png_figures=count,
        imports_from_unpacked_bundle=True,
        environment="existing Python dependencies; relocated project and fresh demo",
    )
    (root / "reports/portable_bundle_validation.json").write_text(json.dumps(report, indent=2))
    print(report)
