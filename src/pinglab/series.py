"""Exploratory hit classification with entire recording series held out.

For repeated measurements of one object; deliberately separate from plate-level
validation. Series labels describe impact location, not an object's defect.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .features import feature_columns
from .learning import make_classifier


def paired_folds(events):
    """Each round contains both classes; never split hits from the same series."""
    required = {"event_id", "series_id", "round", "label"}
    if not required <= set(events):
        raise ValueError("Missing series metadata")
    if events.event_id.duplicated().any():
        raise ValueError("Duplicate event IDs")
    if (events.groupby("series_id")["round"].nunique() != 1).any():
        raise ValueError("One series cannot belong to several rounds")
    if (events.groupby("series_id").label.nunique() != 1).any():
        raise ValueError("Conflicting labels within a series")
    if set(events.label) != {"central", "offcenter"} or events["round"].nunique() < 2:
        raise ValueError("Need both classes in at least two rounds")
    folds = []
    for value in sorted(events["round"].unique()):
        test = np.flatnonzero(events["round"].to_numpy() == value)
        train = np.flatnonzero(events["round"].to_numpy() != value)
        if any(set(events.iloc[ix].label) != {"central", "offcenter"} for ix in (train, test)):
            raise ValueError("Every train and test round must contain both classes")
        assert not set(events.iloc[train].series_id) & set(events.iloc[test].series_id)
        folds.append((train, test))
    return folds


def evaluate_series(events, family="resonance", kind="logistic", test_features=None):
    """Fixed settings, train-only preprocessing, held-out series metrics.

    Anomaly threshold is a descriptive training-only LOO fence, not calibration.
    Optional test_features changes held-out signals only (same IDs and order).
    """
    if kind not in ("logistic", "distance"):
        raise ValueError("Unknown method")
    columns = ["peak_dbfs"] if family == "loudness" else feature_columns(events, family)
    x = events[columns].to_numpy(float)
    if test_features is not None:
        if events.event_id.tolist() != test_features.event_id.tolist():
            raise ValueError("Test variants must preserve event order")
        xt = test_features[columns].to_numpy(float)
    else:
        xt = x
    if not np.isfinite(x).all() or not np.isfinite(xt).all():
        raise ValueError("Non-finite features")
    y = (events.label == "offcenter").to_numpy(int)
    rows, audit, metrics = [], [], []
    for fold, (train, test) in enumerate(paired_folds(events), 1):
        if kind == "logistic":
            model = make_classifier(family).fit(x[train], y[train])
            score = model.predict_proba(xt[test])[:, 1]
            threshold = 0.5
        else:
            reference = x[train][y[train] == 0]
            if len(reference) < 3:
                raise ValueError("Need at least three central reference hits")
            scaler = StandardScaler().fit(reference)
            z = scaler.transform(reference)
            d = cdist(z, z) / np.sqrt(x.shape[1])
            np.fill_diagonal(d, np.inf)
            threshold = max(float(np.quantile(d.min(axis=1), 0.95) * 1.5), 1e-6)
            score = cdist(scaler.transform(xt[test]), z).min(axis=1) / np.sqrt(x.shape[1])
        pred = score >= threshold
        metrics.append(
            dict(
                family=family,
                method=kind,
                fold=fold,
                test_round=int(events.iloc[test]["round"].iloc[0]),
                n_test=len(test),
                balanced_accuracy=float(balanced_accuracy_score(y[test], pred)),
                auc=float(roc_auc_score(y[test], score)),
                central_recall=float(np.mean(~pred[y[test] == 0])),
                offcenter_recall=float(np.mean(pred[y[test] == 1])),
                threshold=threshold,
                confusion_matrix=confusion_matrix(y[test], pred, labels=[0, 1]).tolist(),
            )
        )
        audit.append(
            dict(
                fold=fold,
                train_series=sorted(set(events.iloc[train].series_id)),
                test_series=sorted(set(events.iloc[test].series_id)),
            )
        )
        for j, index in enumerate(test):
            rows.append(
                dict(
                    event_id=events.iloc[index].event_id,
                    series_id=events.iloc[index].series_id,
                    label=events.iloc[index].label,
                    fold=fold,
                    family=family,
                    method=kind,
                    score=float(score[j]),
                    threshold=threshold,
                    prediction="offcenter" if pred[j] else "central",
                )
            )
    return pd.DataFrame(metrics), pd.DataFrame(rows), audit
