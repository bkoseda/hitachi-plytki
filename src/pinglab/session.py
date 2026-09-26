"""One explicit dataset configuration for every notebook; no client-to-demo fallback."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .audio import Config, load_audio
from .dataset import read_manifest
from .demo import generate_demo


@dataclass
class Session:
    root: Path
    manifest: Path
    annotations: Path
    output: Path
    config: Config
    fmax: float
    source_kind: str
    records: object
    background: Path | None

    def pair(self):
        rows = self.records[self.records.label != "background"]
        if rows.empty:
            raise ValueError("Manifest zawiera tylko tło. Dodaj co najmniej jeden plik płytki.")
        a = rows.iloc[0]
        other = rows[rows.label != a.label]
        b = other.iloc[0] if len(other) else rows.iloc[min(1, len(rows) - 1)]
        return a, b


def load_session(root, settings=None):
    root = Path(root).resolve()
    settings = Path(settings or os.environ.get("PINGLAB_SETTINGS", root / "project.json"))
    spec = json.loads(settings.read_text())
    kind = spec["source_kind"]
    if kind not in ("synthetic", "client", "external"):
        raise ValueError("source_kind: synthetic, client lub external.")
    manifest = (root / Path(spec["manifest"]).expanduser()).resolve()
    if kind == "synthetic" and manifest == root / "data/demo/manifest.csv":
        if not (manifest.parent / "SYN_001_ok.wav").exists():
            generate_demo(manifest.parent)
    records = read_manifest(manifest)  # A missing client file is an error, never silently a demo.
    if kind == "client" and records.environment.eq("synthetic").any():
        raise ValueError("Konfiguracja client wskazuje dane synthetic. Popraw manifest.")
    config = Config(**spec.get("audio", {}))
    # Inspect native bandwidth once. Upsampling cannot recreate missing frequencies.
    rates = [
        load_audio(r.resolved_path, config, channel=int(r.channel) if r.channel else None).native_sr
        for r in records.itertuples()
    ]
    fmax = min(float(spec.get("fmax", 20000)), min(rates) / 2)
    if fmax < 500:
        raise ValueError("Za wąskie wspólne pasmo źródeł.")
    annotations = root / spec["annotations"]
    output = root / spec["output"]
    output.mkdir(parents=True, exist_ok=True)
    bg = records[records.label == "background"]
    background = Path(bg.iloc[0].resolved_path) if len(bg) else None
    return Session(root, manifest, annotations, output, config, fmax, kind, records, background)
