"""Save an exploratory model and its preprocessing contract for later local use."""

import json
from pathlib import Path

import joblib

from .learning import make_classifier, plate_table


def train_final(events, folder, family="physics", processing=None):
    table, columns = plate_table(events, family)
    table = table[table.label.isin(["ok", "nok"])]
    counts = table.label.value_counts()
    if counts.get("ok", 0) < 3 or counts.get("nok", 0) < 3:
        raise ValueError("Model wymaga >=3 płytek każdej klasy; najpierw wykonaj ocenę OOF.")
    model = make_classifier(family).fit(table[columns], (table.label == "nok").astype(int))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "model.joblib").exists():
        raise FileExistsError("Model już istnieje; wybierz nowy katalog wersji.")
    joblib.dump(model, folder / "model.joblib")
    metadata = dict(
        family=family,
        columns=columns,
        training_ids=table.plate_id.tolist(),
        aggregation="median",
        processing=processing or {},
        threshold=0.5,
        warning="Model eksploracyjny. Score nie jest skalibrowanym prawdopodobieństwem.",
    )
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return folder


def predict_new(events, folder, processing=None):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["processing"] != (processing or {}):
        raise ValueError("Konfiguracja przetwarzania różni się od treningowej.")
    table, columns = plate_table(events, metadata["family"])
    if set(table.plate_id) & set(metadata["training_ids"]):
        raise ValueError(
            "To płytki treningowe. Użyj nowych ID dla nowych fizycznych płytek; nie zmieniaj ID starych."
        )
    if columns != metadata["columns"]:
        raise ValueError("Niezgodne kolumny cech.")
    model = joblib.load(folder / "model.joblib")  # only load your own trusted local model
    score = model.predict_proba(table[columns])[:, 1]
    result = table[["plate_id", "n_hits"]].copy()
    result["score_nok"] = score
    result["prediction"] = ["nok" if s >= 0.5 else "ok" for s in score]
    return result
