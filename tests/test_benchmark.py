import numpy as np
import pandas as pd
import pytest

from pinglab.benchmark import benchmark, tap_count_curve
from pinglab.learning import NormalReference


def events(n=12):
    rng = np.random.default_rng(23)
    rows = []
    for i in range(n):
        label = "ok" if i < n // 2 else "nok"
        for hit in range(10):
            rows.append(
                dict(
                    plate_id=f"P{i:02}",
                    label=label,
                    session_id="s",
                    hit_id=f"{i}_{hit}",
                    peak_hz=rng.normal(1500, 60),
                    centroid_hz=rng.normal(2000, 80),
                    decay10_s=rng.uniform(0.1, 0.5),
                    mfcc_01_mean=rng.normal(),
                    mfcc_01_std=rng.uniform(),
                )
            )
    return pd.DataFrame(rows)


def test_nested_threshold_and_normal_fit_exclude_outer_test(monkeypatch):
    data = events(20)
    fitted = []
    original = NormalReference.fit

    def spy(self, x):
        fitted.append(np.asarray(x))
        return original(self, x)

    monkeypatch.setattr(NormalReference, "fit", spy)
    metrics, pred, folds, taps = benchmark(data, bootstrap=5)
    assert len(pred) == 60
    assert (pred.groupby(["method", "plate_id"]).size() == 1).all()
    for fold in folds:
        tr, te = set(fold["train_plates"]), set(fold["test_plates"])
        assert tr.isdisjoint(te)
        for inner in fold["inner"]:
            fit, cal = set(inner["fit_plates"]), set(inner["calibration_plates"])
            assert fit.isdisjoint(cal) and (fit | cal) <= tr and (fit | cal).isdisjoint(te)
    # First coordinate = mean peak_hz; every one-class training vector belongs to an OK plate.
    good = data[data.label.eq("ok")].groupby("plate_id").peak_hz.mean().to_numpy()
    assert fitted and all(np.isin(x[:, 0], good).all() for x in fitted)
    curve = tap_count_curve(taps, repeats=4)
    assert set(curve.k) == {1, 3, 5, 10}
    assert curve.n_plates.eq(20).all()
    assert all(len(v) == 2 for v in metrics.recall_cp95)


def test_tap_counts_do_not_invent_taps_and_small_sets_decline():
    data = events(8)
    _, pred, _, taps = benchmark(data, bootstrap=0)
    assert pred.threshold_policy.eq("fixed").all()
    short = taps.groupby("plate_id").head(3)
    assert set(tap_count_curve(short, repeats=2).k) == {1, 3}
    with pytest.raises(ValueError, match=">=4"):
        benchmark(events(6))
