"""Deterministic software stress check; all inputs are explicitly synthetic."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from pinglab.dataset import event_audio
from pinglab.features import extract_features
from pinglab.learning import evaluate, make_classifier, permutation_check, plate_table
from pinglab.robustness import mix_noise

root = Path(__file__).resolve().parents[1]
events = pd.read_csv(root / "reports/runs/demo/features.csv")
base, cols = plate_table(events)
_, _, folds = evaluate(events, bootstrap=0)
rng = np.random.default_rng(812)
variants = {"gain_0.25": [], "white_noise_20db": [], "white_noise_10db": [], "white_noise_0db": []}
for row in events.to_dict("records"):
    y, sr = event_audio(row)
    noise = rng.normal(size=len(y))
    signals = {
        "gain_0.25": y * 0.25,
        **{f"white_noise_{snr}db": mix_noise(y, noise, snr) for snr in [20, 10, 0]},
    }
    for name, signal in signals.items():
        variants[name].append({**row, **extract_features(signal, sr, row["onset_offset_s"])})
result = []
for name, rows in variants.items():
    perturbed, _ = plate_table(pd.DataFrame(rows))
    scores = np.zeros(len(base))
    for fold in folds:
        train = base.plate_id.isin(fold["train_plates"])
        test = base.plate_id.isin(fold["test_plates"])
        model = make_classifier().fit(base.loc[train, cols], (base.loc[train, "label"] == "nok").astype(int))
        scores[test] = model.predict_proba(perturbed.loc[test, cols])[:, 1]
    result.append(
        {
            "variant": name,
            "balanced_accuracy": balanced_accuracy_score(base.label == "nok", scores >= 0.5),
            "n_plates": len(base),
        }
    )
permutation = permutation_check(events, repeats=39)
report = {
    "source": "synthetic only; no claim of ceramic defect performance",
    "protocol": "train original plates; perturb only held-out plate audio inside each same fold",
    "stress": result,
    "permutation": permutation,
}
(root / "reports/stress_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
