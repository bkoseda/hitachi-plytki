"""Fetch only five public demonstration recordings; no defect labels are invented."""

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

out = Path(__file__).resolve().parents[1] / "data/external/realimpact"
out.mkdir(parents=True, exist_ok=True)
records = []
provenance = []
for i in range(5):
    name = f"deconvolved{i}.wav"
    url = f"https://samuelpclarke.com/realimpact/soundfield/{name}"
    dst = out / name
    if not dst.exists():
        with urllib.request.urlopen(url, timeout=45) as r:
            payload = r.read(25 * 1024 * 1024)
        if len(payload) >= 25 * 1024 * 1024:
            raise ValueError("Unexpectedly large demo file")
        dst.write_bytes(payload)
    provenance.append(dict(file=name, url=url, sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    # Demo is the same object observed from different microphone locations, not five independent plates.
    records.append(
        dict(
            path=name,
            plate_id="REALIMPACT_DEMO_OBJECT",
            label="unknown",
            session_id="RI_DEMO",
            position=f"microphone_{i}",
            environment="controlled",
            notes="RealImpact public sound-field demo; no health-state labels",
        )
    )
pd.DataFrame(records).to_csv(out / "manifest.csv", index=False)
(out / "provenance.json").write_text(json.dumps(provenance, indent=2))
(out / "README.txt").write_text(
    "RealImpact, Clarke et al., CVPR 2023.\n"
    "https://samuelpclarke.com/realimpact/\n"
    "Public demonstration audio, deconvolved impact responses from different microphone positions.\n"
    "Not a cracked/intact dataset. Do not assign ok/nok labels.\n"
    "Local research/demo use; external audio is excluded from Git.\n"
    "Repository has MIT software license; verify recording redistribution terms before sharing externally.\n"
)
print(out)
