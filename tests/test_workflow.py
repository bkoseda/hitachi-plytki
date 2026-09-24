import json
from pathlib import Path

import numpy as np
import pytest

from pinglab.dataset import analyze_manifest
from pinglab.demo import generate_demo
from pinglab.learning import evaluate, wilson_interval
from pinglab.model import predict_new, train_final


def test_save_predict_contract(tmp_path):
    manifest = generate_demo(tmp_path / "data", n_plates=15)
    events, _ = analyze_manifest(manifest, tmp_path / "run")
    test_ids = ["SYN_001", "SYN_015"]
    train = events[~events.plate_id.isin(test_ids)]
    test = events[events.plate_id.isin(test_ids)]
    folder = train_final(train, tmp_path / "model", processing={"fmax": 20000})
    result = predict_new(test, folder, {"fmax": 20000})
    assert set(result.plate_id) == set(test_ids)
    assert result.score_nok.between(0, 1).all()
    with pytest.raises(ValueError, match="Konfiguracja"):
        predict_new(test, folder, {"fmax": 7500})
    with pytest.raises(ValueError, match="treningowe"):
        predict_new(train, folder, {"fmax": 20000})
    with pytest.raises(FileExistsError):
        train_final(train, folder)


def test_perfect_small_sample_not_certain():
    lower, upper = wilson_interval(10, 10)
    assert 0.70 < lower < 0.75 and upper == pytest.approx(1)
    lower, upper = wilson_interval(0, 20)
    assert lower < 0.001 and 0.15 < upper < 0.18


def test_annotation_selection_reset(tmp_path, monkeypatch):
    import pinglab.widgets as widgets

    manifest = generate_demo(tmp_path / "data", n_plates=6)
    annotations = tmp_path / "review.json"
    monkeypatch.setattr(widgets, "display", lambda *a, **k: None)
    monkeypatch.setattr(widgets, "inspection", lambda *a, **k: None)
    controls = widgets.review_manifest(manifest, annotations)
    assert controls["save"].disabled
    controls["inspect"].click()
    assert not controls["save"].disabled
    controls["times"].value = "0.4, 1.9, 3.4"
    controls["save"].click()
    saved = json.loads(annotations.read_text())
    assert saved[controls["select"].value] == [0.4, 1.9, 3.4]
    controls["select"].value = controls["select"].options[1]
    assert controls["save"].disabled and controls["times"].value == ""
    controls["save"].click()  # programmatic calls cannot bypass the selected-file guard
    assert json.loads(annotations.read_text()) == saved


def test_session_confound_rejected(tmp_path):
    manifest = generate_demo(tmp_path / "data", n_plates=12)
    events, _ = analyze_manifest(manifest, tmp_path / "run")
    events["session_id"] = events.label
    with pytest.raises(ValueError, match="brakuje klasy"):
        evaluate(events, by_session=True)


@pytest.mark.integration
def test_real_ast_offline_when_prepared():
    from pinglab.demo import resonator
    from pinglab.embeddings import ASTEncoder

    root = Path(__file__).resolve().parents[1]
    cache = root / "models/ast"
    if not cache.exists():
        pytest.skip("Run scripts/warm_model.py to prepare optional weights")
    encoder = ASTEncoder(cache, offline=True)
    first = encoder.encode(resonator(), 48000)
    second = encoder.encode(resonator() * 0.3, 48000)
    assert first.shape == (768,) and np.isfinite(first).all()
    assert np.dot(first, second) > 0.999


def test_noise_robustness_keeps_plate_count(tmp_path):
    from pinglab.robustness import noise_robustness

    manifest = generate_demo(tmp_path / "data", n_plates=12)
    events, _ = analyze_manifest(manifest, tmp_path / "run")
    result = noise_robustness(events, snrs=(20,))
    assert len(result) == 4 and (result.n_plates == 12).all()
    assert np.isfinite(result.balanced_accuracy).all()


def test_channel_contract_in_batch(tmp_path):
    import pandas as pd
    import soundfile as sf

    from pinglab.dataset import event_audio
    from pinglab.demo import resonator

    good = resonator()
    sf.write(tmp_path / "stereo.wav", np.stack([good * 0.1, good], axis=1), 48000)
    pd.DataFrame([dict(path="stereo.wav", plate_id="P1", label="ok", session_id="S1", channel="0")]).to_csv(
        tmp_path / "m.csv", index=False
    )
    events, qc = analyze_manifest(tmp_path / "m.csv", tmp_path / "run")
    assert (events.selected_channel == 0).all() and qc.selected_channel.iloc[0] == 0
    y, _ = event_audio(events.iloc[0].to_dict())
    assert np.max(np.abs(y)) < 0.15
