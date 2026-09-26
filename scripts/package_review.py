"""Portable demo-only share bundle. No client audio, weights or private local paths."""

import html
import json
import zipfile
from pathlib import Path

import mistune
import nbformat

root = Path(__file__).resolve().parents[1]
out = root / "deliverables"
out.mkdir(exist_ok=True)
nb = nbformat.read(root / "notebooks/00_eksploracja_do_przegladu.ipynb", as_version=4)
streams = "\n".join(str(o.get("text", "")) for c in nb.cells for o in c.get("outputs", []))
if "GOTOWE · eksploracja · źródło: synthetic" not in streams:
    raise ValueError("Paczka domyślnie tylko demo. Uruchom 00 na synthetic przed eksportem.")
parts = [
    """<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Ping Lab — przegląd demo</title><style>
body{max-width:1120px;margin:40px auto;padding:0 22px;color:#23303d;background:#faf9f6;font:16px/1.6 system-ui}h1{font-size:34px}h2{margin-top:48px;border-top:1px solid #cdd7dc;padding-top:24px}table{border-collapse:collapse;font-size:13px;background:white}td,th{padding:8px 12px;border:1px solid #dce2e6;text-align:left}img{max-width:100%;height:auto}pre{overflow:auto;background:#eef1f2;padding:14px;font-size:12px}.output{overflow-x:auto;margin:14px 0}summary{cursor:pointer;padding:8px;background:#eef1f2}details{margin:6px 0}audio{max-width:100%}.banner{padding:16px;border-left:4px solid #bf732b;background:#fff1d8}</style><body><p class="banner">DANE SYNTETYCZNE · demonstracja przepływu. Wyniki nie potwierdzają wykrywania pęknięć u klienta.</p>"""
]
for cell in nb.cells:
    if cell.cell_type == "markdown":
        parts.append(mistune.html(cell.source))
        continue
    parts.append(
        "<details><summary>Kod analizy</summary><pre>" + html.escape(cell.source) + "</pre></details>"
    )
    for o in cell.get("outputs", []):
        data = o.get("data", {})
        if o.output_type == "error":
            raise ValueError("Notebook zawiera błąd.")
        if "image/png" in data:
            rendered = (
                '<img alt="Wykres analizy audio" src="data:image/png;base64,' + data["image/png"] + '">'
            )
        elif "text/html" in data:
            rendered = data["text/html"]
        elif "text/markdown" in data:
            rendered = mistune.html(data["text/markdown"])
        else:
            rendered = "<pre>" + html.escape(o.get("text", data.get("text/plain", ""))) + "</pre>"
        parts.append('<div class="output">' + rendered + "</div>")
parts.append("</body></html>")
(out / "Ping_Lab_przeglad.html").write_text("\n".join(parts))
# A shareable notebook is a snapshot; the zip provides imports and reproducible demo generation.
nbformat.write(nb, out / "Ping_Lab_eksploracja.ipynb")
paths = list((root / "reports").glob("*.json"))
for pattern in [
    "src/pinglab/*.py",
    "notebooks/*.ipynb",
    "docs/*.md",
    "config/*.json",
    ".vscode/*.json",
    "tests/*.py",
    "assets/*",
]:
    paths.extend(root.glob(pattern))
for name in [
    "README.md",
    "pyproject.toml",
    "project.json",
    "requirements-macos-py312.lock",
    "index.html",
    "Start.command",
    "data/manifest.template.csv",
    "data/demo/manifest.csv",
    "data/demo/annotations.json",
    "scripts/setup.sh",
    "scripts/warm_model.py",
    "scripts/make_notebooks.py",
    "scripts/make_recording_notebook.py",
    "scripts/package_review.py",
    "scripts/verify_notebooks.py",
    "scripts/download_realimpact.py",
    "scripts/verify_client_switch.py",
    "scripts/verify_portable_bundle.py",
]:
    paths.append(root / name)
with zipfile.ZipFile(out / "Ping_Lab_start.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        # Remove machine-specific paths from saved output snapshots, never executable source.
        content = path.read_bytes()
        if path.suffix in (".ipynb", ".json"):
            content = content.replace(str(root).encode(), b".")
        archive.writestr("Ping_Lab/" + str(path.relative_to(root)), content)
    archive.write(out / "Ping_Lab_przeglad.html", "Ping_Lab/Ping_Lab_przeglad.html")
print(json.dumps({p.name: p.stat().st_size for p in out.iterdir()}, indent=2))
