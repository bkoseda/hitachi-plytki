import json
import time
from pathlib import Path

from pinglab.audio import cut_hits, detect_hits, load_audio
from pinglab.embeddings import ASTEncoder

root = Path(__file__).resolve().parents[1]
t = time.perf_counter()
encoder = ASTEncoder(root / "models/ast")
a = load_audio(root / "data/demo/SYN_001_ok.wav")
y = cut_hits(a.y, a.sr, detect_hits(a.y, a.sr))[0]["y"]
v = encoder.encode(y, a.sr)
report = dict(shape=list(v.shape), revision=encoder.revision, seconds=time.perf_counter() - t)
(root / "reports/ast_smoke.json").write_text(json.dumps(report, indent=2))
print(report)
