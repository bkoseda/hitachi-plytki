import numpy as np
import pandas as pd
import pytest

from pinglab.series import evaluate_series, paired_folds


def events():
    rows = []
    for r in [1, 2]:
        for label in ["central", "offcenter"]:
            for i in range(5):
                rows.append(
                    dict(
                        event_id=f"{r}_{label}_{i}",
                        series_id=f"{r}_{label}",
                        round=r,
                        label=label,
                        peak_dbfs=(-30 if label == "central" else -10) + i * 0.1,
                    )
                )
    return pd.DataFrame(rows)


def test_whole_series_out_and_test_only_changes_do_not_change_training():
    e = events()
    for tr, te in paired_folds(e):
        assert not set(e.iloc[tr].series_id) & set(e.iloc[te].series_id)
    base, p, a = evaluate_series(e, "loudness")
    changed = e.copy()
    changed.loc[changed["round"] == 2, "peak_dbfs"] += 100
    other, q, _ = evaluate_series(e, "loudness", test_features=changed)
    assert np.allclose(p[p.fold == 1].score, q[q.fold == 1].score)
    assert (base.balanced_accuracy == 1).all()
    assert len(p) == len(e)


def test_invalid_groups_and_variant_alignment_rejected():
    e = events()
    e.loc[0, "round"] = 2
    with pytest.raises(ValueError):
        paired_folds(e)
    e = events()
    with pytest.raises(ValueError):
        evaluate_series(e, "loudness", test_features=e.iloc[::-1])


def test_distance_training_threshold_ignores_test_variants():
    e = events()
    m, _, _ = evaluate_series(e, "loudness", "distance")
    v = e.copy()
    v.peak_dbfs += 100
    n, _, _ = evaluate_series(e, "loudness", "distance", v)
    assert np.allclose(m.threshold, n.threshold)
