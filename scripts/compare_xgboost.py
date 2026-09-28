"""Fixed XGBoost versus logistic regression; entire rounds held out, no tuning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from xgboost import __version__ as xgboost_version

from pinglab.features import feature_columns
from pinglab.learning import make_classifier
from pinglab.series import paired_folds

PARAMS = dict(
    n_estimators=80,
    max_depth=2,
    learning_rate=0.05,
    min_child_weight=1,
    reg_lambda=5,
    reg_alpha=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2,
)


def compare(events, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, predictions, audit = [], [], []
    y = (events.label == "offcenter").to_numpy(int)
    for family in ["resonance", "physics", "mfcc", "mel"]:
        columns = feature_columns(events, family)
        x = events[columns].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite features")
        for train, test in paired_folds(events):
            round_id = int(events.iloc[test]["round"].iloc[0])
            audit.append(
                dict(
                    family=family,
                    test_round=round_id,
                    train_series=sorted(set(events.iloc[train].series_id)),
                    test_series=sorted(set(events.iloc[test].series_id)),
                )
            )
            for method in ["logistic", "xgboost"]:
                if method == "xgboost":
                    model = XGBClassifier(**PARAMS)
                    model.fit(x[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
                else:
                    model = make_classifier(family).fit(x[train], y[train])
                score = model.predict_proba(x[test])[:, 1]
                pred = score >= 0.5
                rows.append(
                    dict(
                        family=family,
                        method=method,
                        test_round=round_id,
                        n_train=len(train),
                        n_test=len(test),
                        train_ba=balanced_accuracy_score(y[train], model.predict(x[train])),
                        accuracy=accuracy_score(y[test], pred),
                        balanced_accuracy=balanced_accuracy_score(y[test], pred),
                        auc=roc_auc_score(y[test], score),
                        central_recall=float(np.mean(~pred[y[test] == 0])),
                        offcenter_recall=float(np.mean(pred[y[test] == 1])),
                    )
                )
                for i, s, p in zip(test, score, pred):
                    predictions.append(
                        dict(
                            event_id=events.iloc[i].event_id,
                            family=family,
                            method=method,
                            test_round=round_id,
                            score=float(s),
                            label=events.iloc[i].label,
                            prediction="offcenter" if p else "central",
                        )
                    )
    result = pd.DataFrame(rows)
    result.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                xgboost_version=xgboost_version,
                parameters=PARAMS,
                folds=audit,
                threshold=0.5,
                tuning=False,
                note="Exploratory: one racket, two correlated rounds.",
            ),
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(compare(pd.read_csv(args.features), args.output).round(3).to_string(index=False))
