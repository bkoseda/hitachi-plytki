"""Execute every notebook with real external unlabeled WAV+MP3 files; never alter project.json."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import nbformat
import pandas as pd
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
folder = root / "reports/runs/client-switch-check"
folder.mkdir(parents=True, exist_ok=True)
source = root / "data/external/realimpact"
if not (source / "deconvolved0.wav").exists():
    raise SystemExit("Najpierw scripts/download_realimpact.py — ten test wymaga zewnętrznego audio.")
shutil.copyfile(source / "deconvolved0.wav", folder / "sample_a.wav")
subprocess.run(
    [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-i",
        str(source / "deconvolved1.wav"),
        str(folder / "sample_b.mp3"),
    ],
    check=True,
)
pd.DataFrame(
    [
        dict(
            path=name,
            plate_id="REAL_OBJECT",
            label="unknown",
            session_id="EXTERNAL",
            plate_type="unknown",
            temperature="",
        )
        for name in ["sample_a.wav", "sample_b.mp3"]
    ]
).to_csv(folder / "manifest.csv", index=False)
spec = dict(
    source_kind="external",
    manifest=str(folder / "manifest.csv"),
    annotations=str(folder / "review.json"),
    output=str(folder / "outputs"),
    fmax=20000,
)
settings = folder / "settings.json"
settings.write_text(json.dumps(spec))
os.environ["PINGLAB_SETTINGS"] = str(settings)
results = []
for path in sorted((root / "notebooks").glob("*.ipynb")):
    if path.name.startswith("07_"):
        continue  # recorder has an explicit separate audio source; never activate it in batch tests
    nb = nbformat.read(path, as_version=4)
    NotebookClient(
        nb, timeout=1200, kernel_name="pinglab", resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(nb, folder / path.name)
    text = json.dumps(nb, ensure_ascii=False)
    assert "GOTOWE" in text or all(c.get("execution_count") for c in nb.cells if c.cell_type == "code")
    assert "SYN_001" not in text, path.name
    results.append(
        dict(
            notebook=path.name,
            status="passed",
            source="external WAV + MP3, unknown labels, one physical object",
        )
    )
    print(path.name, "OK", flush=True)
(root / "reports/client_switch_validation.json").write_text(json.dumps(results, indent=2))
