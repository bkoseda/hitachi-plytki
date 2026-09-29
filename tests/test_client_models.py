import numpy as np
import pandas as pd

from pinglab.clientmodels import balanced_group_weights, fit_predict, group_folds
from pinglab.features import feature_columns


def example():
    rng = np.random.default_rng(11)
    rows = []
    names = ["Bad_1", "Bad_2", "OK_1", "OK_2", "OK_3"]
    for i, group in enumerate(names):
        for j in range(3 + i):
            row = dict(group=group, label=int(i < 2), recording_id=f"file-{i}", event_id=f"{i}-{j}")
            row.update({k: float(rng.normal()) for k in feature_columns({}, "physics")})
            rows.append(row)
    return pd.DataFrame(rows)


def test_splits_keep_objects_and_recordings_whole():
    data = example()
    splits = list(group_folds(data))
    assert len(splits) == 6
    for _, tr, te in splits:
        train, test = data.iloc[tr], data.iloc[te]
        assert set(train.group).isdisjoint(test.group)
        assert set(train.recording_id).isdisjoint(test.recording_id)
        assert set(train.label) == set(test.label) == {0, 1}


def test_weights_do_not_reward_longer_recordings_or_majority_class():
    data = example()
    data["weight"] = balanced_group_weights(data)
    totals = data.groupby("label").weight.sum()
    np.testing.assert_allclose(totals.iloc[0], totals.iloc[1])
    good = data[data.label == 0].groupby("group").weight.sum()
    np.testing.assert_allclose(good, good.iloc[0])


def test_held_out_labels_and_other_test_rows_do_not_change_prediction():
    data = example()
    _, tr, te = next(group_folds(data))
    train, test = data.iloc[tr].copy(), data.iloc[te].copy()
    original = fit_predict(train, test, "physics")
    test["label"] = 1 - test.label
    cols = feature_columns(test, "physics")
    test.loc[test.index[1:], cols] = 1e8
    changed = fit_predict(train, test, "physics")
    np.testing.assert_allclose(original[0], changed[0], rtol=1e-10, atol=1e-10)


def test_combined_features_are_union_without_metadata():
    data = example()
    for i in range(1, 14):
        for stat in ["mean", "std"]:
            data[f"mfcc_{i:02d}_{stat}"] = 0.0
    combined = feature_columns(data, "mfcc_physics")
    assert len(combined) == len(set(combined)) == 41
    assert set(combined) == set(feature_columns(data, "physics")) | set(feature_columns(data, "mfcc"))
    assert not set(combined) & {"label", "group", "recording_id", "event_id"}
