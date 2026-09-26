"""Small fixed baselines with plate-disjoint folds and optional inner threshold selection."""

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .features import feature_columns
from .learning import NormalReference, make_classifier, plate_table

METHODS = {
    "A_distance": ("resonance", "spread", "distance"),
    "A_logistic": ("resonance", "spread", "logistic"),
    "B_mfcc_top3": ("mfcc", "taps", "logistic"),
}


def summarize(events, family, spread=False):
    table, cols = plate_table(events, family, "mean" if spread else "median")
    if spread:
        grouped = events.groupby("plate_id", sort=True)[cols]
        std = grouped.std(ddof=0).add_suffix("_spread")
        table = table.merge(std, left_on="plate_id", right_index=True)
        cols = cols + std.columns.tolist()
    return table, cols


def _fit_score(events, train_ids, test_ids, method):
    family, unit, learner = METHODS[method]
    tr = events[events.plate_id.isin(train_ids)]
    te = events[events.plate_id.isin(test_ids)]
    if unit == "taps":
        cols = feature_columns(events, family)
        weights = 1 / tr.groupby("plate_id").plate_id.transform("size").to_numpy()
        model = make_classifier(family).fit(
            tr[cols], tr.label.eq("nok").astype(int), logisticregression__sample_weight=weights
        )
        taps = te[["plate_id", "label", "hit_id"]].copy()
        taps["score"] = model.predict_proba(te[cols])[:, 1]
        scores = taps.groupby("plate_id").score.apply(lambda x: x.nlargest(min(3, len(x))).mean())
        return scores.reindex(test_ids).to_numpy(), taps
    train, cols = summarize(tr, family, unit == "spread")
    test, _ = summarize(te, family, unit == "spread")
    test = test.set_index("plate_id").loc[test_ids]
    if learner == "distance":
        model = NormalReference().fit(train.loc[train.label.eq("ok"), cols])
        scores = model.score(test[cols]) / model.threshold
    else:
        model = make_classifier(family).fit(train[cols], train.label.eq("nok").astype(int))
        scores = model.predict_proba(test[cols])[:, 1]
    return scores, None


def _summary(pred, bootstrap=300, seed=42):
    y = pred.label.eq("nok").to_numpy()
    decisions = pred.score.to_numpy() >= pred.threshold.to_numpy()
    tp = int(np.sum(y & decisions))
    fn = int(np.sum(y & ~decisions))
    fp = int(np.sum(~y & decisions))
    tn = int(np.sum(~y & ~decisions))
    n = tp + fn
    ci = [
        0.0 if tp == 0 else float(beta.ppf(0.025, tp, n - tp + 1)),
        1.0 if tp == n else float(beta.ppf(0.975, tp + 1, n - tp)),
    ]
    result = dict(
        n_plates=len(pred),
        recall_nok=tp / n,
        false_reject=fp / (fp + tn),
        missed_nok=fn,
        roc_auc=float(roc_auc_score(y, pred.score)),
        average_precision=float(average_precision_score(y, pred.score)),
        recall_cp95=ci,
        confusion_matrix=confusion_matrix(y, decisions, labels=[0, 1]).tolist(),
    )
    # Fixed OOF predictions: plate sampling only, NOT training or domain-shift uncertainty.
    rng = np.random.default_rng(seed)
    ids = [np.flatnonzero(y == value) for value in (False, True)]
    samples = []
    for _ in range(bootstrap):
        a, b = [rng.choice(i, len(i), replace=True) for i in ids]
        samples.append([float(decisions[b].mean()), float(decisions[a].mean())])
    if samples:
        result["recall_bootstrap95"], result["false_reject_bootstrap95"] = np.quantile(
            samples, [0.025, 0.975], axis=0
        ).T.tolist()
    return result


def benchmark(events, seed=42, bootstrap=300):
    """Outer fixed plate folds; inner thresholds never see an outer test plate.

    Methods are prespecified, not selected in inner CV. Model comparison remains exploratory.
    Unknown plates are excluded, not relabelled. Distance fits ONLY good training plates.
    """
    table, _ = plate_table(events, "resonance")
    table = table[table.label.isin(["ok", "nok"])].reset_index(drop=True)
    counts = table.label.value_counts()
    if min(counts.get("ok", 0), counts.get("nok", 0)) < 4:
        raise ValueError(
            "Porównanie metod: potrzeba >=4 OK i >=4 NOK. Przy 1–2 plikach zostajemy przy eksploracji."
        )
    ids = table.plate_id.to_numpy()
    y = table.label.eq("nok").to_numpy(int)
    splits = list(StratifiedKFold(min(5, int(counts.min())), shuffle=True, random_state=seed).split(ids, y))
    rows, audits, tap_rows = [], [], []
    for method in METHODS:
        for fold, (train, test) in enumerate(splits):
            threshold = 1.0 if method == "A_distance" else 0.5
            policy = "fixed"
            inner_audit = []
            # Need >=3 good references in every inner training fold.
            if min(np.bincount(y[train], minlength=2)) >= 5:
                inner_scores = np.full(len(train), np.nan)
                inner = StratifiedKFold(3, shuffle=True, random_state=seed + 1)
                for itr, iva in inner.split(train, y[train]):
                    fit_ids, cal_ids = ids[train[itr]], ids[train[iva]]
                    inner_scores[iva], _ = _fit_score(events, fit_ids, cal_ids, method)
                    inner_audit.append(dict(fit_plates=fit_ids.tolist(), calibration_plates=cal_ids.tolist()))
                # Largest threshold with zero misses on inner OOF NOK; NOT a guarantee on outer test.
                threshold = float(np.min(inner_scores[y[train] == 1]))
                policy = "inner_oof_zero_misses"
            scores, taps = _fit_score(events, ids[train], ids[test], method)
            if taps is not None:
                taps["fold"] = fold
                taps["threshold"] = threshold
                taps["method"] = method
                tap_rows.append(taps)
            for i, score in zip(test, scores):
                rows.append(
                    dict(
                        method=method,
                        plate_id=ids[i],
                        label=table.label.iloc[i],
                        score=float(score),
                        threshold=threshold,
                        threshold_policy=policy,
                        fold=fold,
                    )
                )
            audits.append(
                dict(
                    method=method,
                    fold=fold,
                    train_plates=ids[train].tolist(),
                    test_plates=ids[test].tolist(),
                    inner=inner_audit,
                    threshold=threshold,
                    threshold_policy=policy,
                )
            )
    predictions = pd.DataFrame(rows)
    leaderboard = pd.DataFrame(
        [
            dict(method=method, **_summary(group, bootstrap, seed))
            for method, group in predictions.groupby("method", sort=False)
        ]
    )
    taps = pd.concat(tap_rows, ignore_index=True) if tap_rows else pd.DataFrame()
    return leaderboard, predictions, audits, taps


def tap_count_curve(oof_taps, ks=(1, 3, 5, 10), repeats=100, seed=42):
    """Re-use held-out tap scores. Same eligible cohort for all k; fixed full-tap thresholds.

    Variation reflects tap choice only, not independent new plates or model retraining.
    """
    if oof_taps.empty:
        return pd.DataFrame()
    counts = oof_taps.groupby("plate_id").size()
    # Never fabricate extra independent taps by replacement.
    valid = [k for k in ks if k <= counts.min()]
    rng = np.random.default_rng(seed)
    rows = []
    for k in valid:
        for repeat in range(repeats):
            sampled = []
            for pid, group in oof_taps.groupby("plate_id"):
                chosen = group.iloc[rng.choice(len(group), k, replace=False)]
                sampled.append(
                    dict(
                        plate_id=pid,
                        label=group.label.iloc[0],
                        score=chosen.score.nlargest(min(3, k)).mean(),
                        threshold=group.threshold.iloc[0],
                    )
                )
            p = pd.DataFrame(sampled)
            decision = p.score.ge(p.threshold)
            rows.append(
                dict(
                    k=k,
                    repeat=repeat,
                    n_plates=len(p),
                    recall_nok=float(decision[p.label.eq("nok")].mean()),
                    false_reject=float(decision[p.label.eq("ok")].mean()),
                )
            )
    return pd.DataFrame(rows)
