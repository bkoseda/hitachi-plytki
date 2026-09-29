"""Execute both notebooks in fresh local kernels and export static offline HTML."""

import copy
import hashlib
import json
import sys
import time

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from pinglab.clientstudy import OUT_REL, find_root

ROOT = find_root()
OUT = ROOT / OUT_REL


def data_checks():
    archive = json.loads((ROOT / "config/client_archive_manifest.json").read_text())
    for row in archive:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    m = pd.read_csv(OUT / "manifest.csv")
    f = pd.read_csv(OUT / "features.csv")
    assert len(m) == 45 and m.recording_id.is_unique
    assert f.event_id.is_unique and len(f) == 3613
    assert not set(f.recording_id) & {"R16", "G11", "G12", "G16"}
    assert (abs((f.end_s - f.start_s) - 0.17) < 1 / 48000).all()
    reps = pd.read_pickle(OUT / "representations.pkl")
    assert len(reps) == 410 and reps.event_id.is_unique
    assert np.isfinite(reps.filter(regex="^ast_").to_numpy()).all()
    for split in json.loads((OUT / "split_audit.json").read_text()):
        assert not set(split["train_groups"]) & set(split["test_groups"])
        assert not set(split["train_recordings"]) & set(split["test_recordings"])
    scores = pd.read_csv(OUT / "model_comparison.csv")
    assert (scores.query("level == 'assumed_plate'").n == 5).all()
    assert (scores.query("level == 'recording'").n == 27).all()
    noise = pd.read_csv(OUT / "noise_metrics.csv")
    assert len(noise) == 32 and noise.model.nunique() == 8
    assert not noise.duplicated(["model", "variant"]).any()
    snr = pd.read_csv(OUT / "noise_snr_audit.csv")
    assert len(snr) == 540 and snr.event_id.nunique() == 270
    np.testing.assert_allclose(snr.requested_snr_db, snr.measured_snr_db, atol=1e-8)
    noisy_predictions = pd.read_csv(OUT / "noise_recording_predictions.csv")
    assert len(noisy_predictions) == 8 * 3 * 27
    assert not noisy_predictions.duplicated(["model", "variant", "recording_id"]).any()
    for model in noise.model.unique():
        g = noise[noise.model == model].set_index("variant")
        np.testing.assert_allclose(
            g.loc["original", ["accuracy", "auc"]].to_numpy(float),
            g.loc["gain -12 dB", ["accuracy", "auc"]].to_numpy(float),
        )
    return dict(
        archive_files_verified=len(archive),
        audio_files=len(m),
        events=len(f),
        embedding_events=len(reps),
        groups_disjoint=True,
        unchanged_by_gain=True,
        noise_models=8,
        controlled_snr_windows=540,
    )


def execute():
    verification = {"python": sys.executable, "notebooks": []}
    for name in ["02_jakosc_obu_paczek", "03_modele_obu_paczek"]:
        path = ROOT / "notebooks/client" / f"{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        t = time.monotonic()
        try:
            NotebookClient(
                nb, timeout=300, kernel_name="pinglab", store_widget_state=True,
                on_cell_start=lambda cell, cell_index: print(f"{name}: cell {cell_index}", flush=True),
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        except BaseException:
            nbformat.write(nb, OUT / f"{name}_FAILED.ipynb")
            raise
        errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
        assert not errors, errors
        assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
        nbformat.validate(nb)
        nbformat.write(nb, path)
        printable = copy.deepcopy(nb)
        printable.metadata.pop("widgets", None)
        for cell in printable.cells:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "application/vnd.jupyter.widget-view+json" in data:
                    data.clear()
                    data["text/plain"] = (
                        "Interaktywny wybór nagrania jest dostępny w notebooku IPYNB w VS Code."
                    )
        exporter = HTMLExporter(template_name="lab")
        exporter.exclude_input_prompt = True
        body, _ = exporter.from_notebook_node(printable)
        (OUT / f"{name}.html").write_text(body)
        row = dict(
            name=name,
            seconds=round(time.monotonic() - t, 2),
            code_cells=sum(c.cell_type == "code" for c in nb.cells),
            errors=0,
        )
        verification["notebooks"].append(row)
        print(row, flush=True)
    verification["data_checks"] = data_checks()
    (OUT / "validation.json").write_text(json.dumps(verification, indent=2))
    print(verification["data_checks"], flush=True)


if __name__ == "__main__":
    execute()
