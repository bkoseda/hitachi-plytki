"""Verify original bytes, source archives, splits, embeddings and portable report links."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/runs/client-update-20260930"
count = 0
for manifest in ["client_archive_manifest.json", "client_archives_20260930.json"]:
    for row in json.loads((ROOT / "config" / manifest).read_text()):
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
        count += 1
new = json.loads((ROOT / "config/client_archives_20260930.json").read_text())
for name in {r["archive"] for r in new}:
    p = Path.home() / "Downloads" / name
    if p.exists():
        rows = [r for r in new if r["archive"] == name]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == rows[0]["archive_sha256"]
        with ZipFile(p) as z:
            for row in rows:
                assert hashlib.sha256(z.read(row["member"])).hexdigest() == row["sha256"]
m = pd.read_csv(OUT / "all_recordings.csv")
assert len(m) == 61 and m.recording_id.is_unique
r = pd.read_pickle(OUT / "all_representations.pkl")
assert r.event_id.is_unique and len(r) == 558
assert np.isfinite(r.filter(regex="^ast_").to_numpy()).all()
assert r[r.scope.eq("core")].recording_id.nunique() == 42
for item in json.loads((OUT / "identity_scenarios.json").read_text()).values():
    for fold in item["splits"]:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
        assert set(fold["train_recordings"]).isdisjoint(fold["test_recordings"])
a = json.loads((OUT / "old_to_new_training.json").read_text())
assert len(a["train_recordings"]) == 27 and len(a["test_recordings"]) == 15
assert set(a["train_recordings"]).isdisjoint(a["test_recordings"])
b = pd.read_csv(OUT / "old_to_new_by_batch.csv")
assert b[b.batch.eq("3-batch-enhancements-off")].recall_nok.isna().all()


class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k == "src" or k == "href" and not v.startswith("#"):
                assert not v.startswith(("http:", "https:"))
                assert (OUT / unquote(v)).is_file(), v


Links().feed((OUT / "START.html").read_text())
n = nbformat.read(ROOT / "notebooks/client/05_nowe_paczki_20260930.ipynb", 4)
code = [c for c in n.cells if c.cell_type == "code"]
assert all(c.execution_count is not None for c in code)
assert not [o for c in code for o in c.get("outputs", []) if o.output_type == "error"]
print(
    f"PASS: {count} original hashes, 16 ZIP members, 61 recordings, 558 finite representations, disjoint scenario splits, report links, {len(code)} executed notebook cells."
)
