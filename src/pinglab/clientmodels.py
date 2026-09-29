"""Fixed, group-held-out exploratory models. No tuning on these five pilot groups."""

import json
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .clientstudy import OUT_REL, find_root
from .features import feature_columns

SPECS = {
    "physics + LR": ("physics", "lr"),
    "MFCC + LR": ("mfcc", "lr"),
    "mel + LR": ("mel", "lr"),
    "physics + XGB": ("physics", "xgb"),
    "MFCC + XGB": ("mfcc", "xgb"),
    "AST + LR": ("ast", "lr"),
    "AST + XGB": ("ast", "xgb"),
    "QC only + LR": ("qc", "lr"),
    "QC no codec + LR": ("qc_no_codec", "lr"),
}
QC_COLS = [
    "file_peak_dbfs",
    "file_rms_p10_dbfs",
    "file_below_minus100_fraction",
    "file_duration_s",
    "file_is_aac",
]


def group_folds(frame):
    groups = frame.groupby("group").label.agg(["first", "nunique"])
    if (groups["nunique"] != 1).any():
        raise ValueError("Grupa ma sprzeczne etykiety.")
    good = groups.index[groups["first"] == 0].tolist()
    bad = groups.index[groups["first"] == 1].tolist()
    if len(good) < 2 or len(bad) < 2:
        raise ValueError("Potrzeba co najmniej dwóch grup każdej klasy do tego testu.")
    for b, g in product(bad, good):
        test = frame.group.isin([b, g]).to_numpy()
        yield f"{b}__{g}", np.flatnonzero(~test), np.flatnonzero(test)


def balanced_group_weights(frame):
    n = frame.groupby("group").group.transform("size").to_numpy()
    groups_per_class = frame.drop_duplicates("group").groupby("label").size().to_dict()
    w = 1 / n / frame.label.map(groups_per_class).to_numpy()
    return w / w.mean()


def fit_predict(train, test, family, kind="lr"):
    cols = (
        QC_COLS
        if family == "qc"
        else QC_COLS[:-1]
        if family == "qc_no_codec"
        else feature_columns(train, family)
    )
    x = train[cols].to_numpy(float)
    z = test[cols].to_numpy(float)
    w = balanced_group_weights(train)
    scaler = StandardScaler().fit(x, sample_weight=w)
    x = scaler.transform(x)
    z = scaler.transform(z)
    if kind == "lr":
        model = LogisticRegression(C=0.1, solver="liblinear", max_iter=2000, random_state=17)
    else:
        model = XGBClassifier(
            n_estimators=60,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=3,
            reg_lambda=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            random_state=17,
        )
    model.fit(x, train.label.to_numpy(), sample_weight=w)
    return model.predict_proba(z)[:, 1]


def metrics(frame):
    y = frame.label.to_numpy(int)
    p = frame.score.to_numpy(float)
    d = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return dict(
        n=len(y),
        n_nok=int(sum(y)),
        accuracy=accuracy_score(y, d),
        balanced_accuracy=balanced_accuracy_score(y, d),
        recall_nok=tp / max(tp + fn, 1),
        false_reject_ok=fp / max(tn + fp, 1),
        auc=roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        tp=int(tp),
        fn=int(fn),
        tn=int(tn),
        fp=int(fp),
    )


def evaluate(frame, specs=None):
    frame = frame.reset_index(drop=True)
    specs = specs or SPECS
    predictions = []
    audit = []
    for fold, tr, te in group_folds(frame):
        train, test = frame.iloc[tr], frame.iloc[te]
        assert not set(train.group) & set(test.group)
        assert not set(train.recording_id) & set(test.recording_id)
        assert not set(train.event_id) & set(test.event_id)
        audit.append(
            dict(
                fold=fold,
                train_groups=sorted(train.group.unique()),
                test_groups=sorted(test.group.unique()),
                train_recordings=sorted(train.recording_id.unique()),
                test_recordings=sorted(test.recording_id.unique()),
                train_taps=len(train),
                test_taps=len(test),
            )
        )
        for name, (family, kind) in specs.items():
            scores = fit_predict(train, test, family, kind)
            p = test[["event_id", "recording_id", "group", "label", "onset_s"]].copy()
            p["score"] = scores
            p["model"] = name
            p["fold"] = fold
            predictions.append(p)
    pred = pd.concat(predictions, ignore_index=True)
    # Fixed median of tap scores per recording; average the repeated held-out fits.
    filefold = pred.groupby(
        ["model", "fold", "recording_id", "group", "label"], as_index=False
    ).score.median()
    files = filefold.groupby(["model", "recording_id", "group", "label"], as_index=False).score.mean()
    groups = files.groupby(["model", "group", "label"], as_index=False).score.median()
    score_rows = []
    for level, table in [("recording", files), ("assumed_plate", groups)]:
        for model, g in table.groupby("model", sort=False):
            score_rows.append(dict(level=level, model=model, **metrics(g)))
        baseline = table[table.model == table.model.iloc[0]].copy()
        baseline["score"] = 0.0
        score_rows.append(dict(level=level, model="always OK", **metrics(baseline)))
    fold_scores = []
    for (model, fold), g in filefold.groupby(["model", "fold"]):
        fold_scores.append(dict(model=model, fold=fold, **metrics(g)))
    return pred, files, groups, pd.DataFrame(score_rows), pd.DataFrame(fold_scores), audit


def run(root=None):
    root = find_root(root)
    out = root / OUT_REL
    reps = pd.read_pickle(out / "representations.pkl")
    records = pd.read_csv(out / "manifest.csv")
    qc = records[
        ["recording_id", "peak_dbfs", "rms_p10_dbfs", "below_minus100_fraction", "duration_s", "codec"]
    ].copy()
    qc["is_aac"] = (qc.codec == "AAC").astype(int)
    qc = qc.drop(columns="codec").rename(
        columns={c: f"file_{c}" for c in qc if c not in ["recording_id", "codec"]}
    )
    reps = reps.merge(qc, on="recording_id", validate="many_to_one")
    core = reps[reps.scope == "core"].copy()
    core.to_pickle(out / "model_input.pkl")
    pred, files, groups, scores, folds, audit = evaluate(core)
    pred.to_csv(out / "oof_taps.csv", index=False)
    files.to_csv(out / "oof_recordings.csv", index=False)
    groups.to_csv(out / "oof_assumed_plates.csv", index=False)
    scores.to_csv(out / "model_comparison.csv", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2))
    # File-format rule: deliberately uses no acoustic information.
    control = files[files.model == "MFCC + LR"].copy()
    control["score"] = control.recording_id.map(records.set_index("recording_id").codec.eq("AAC")).astype(
        float
    )
    codec_metrics = metrics(control)
    (out / "codec_control.json").write_text(json.dumps(codec_metrics, indent=2))
    # Same support / no tool change: 4-point has both NOK and all three OK groups.
    matched = core[(core.support == "4-point") & ~core.tool_change & ~core.angle]
    _, mf, mg, ms, _, ma = evaluate(
        matched,
        {k: SPECS[k] for k in ["physics + LR", "MFCC + LR", "AST + LR", "QC only + LR", "QC no codec + LR"]},
    )
    ms.to_csv(out / "matched_support_metrics.csv", index=False)
    mf.to_csv(out / "matched_support_recordings.csv", index=False)
    (out / "matched_split_audit.json").write_text(json.dumps(ma, indent=2))
    # OLD cannot establish independent plates; descriptive external recording check only.
    old = reps[reps.scope == "old_unknown"].copy()
    old_results = []
    if len(old):
        for name in ["physics + LR", "MFCC + LR", "AST + LR"]:
            p = old[["recording_id", "label"]].copy()
            p["score"] = fit_predict(core, old, *SPECS[name])
            p["model"] = name
            old_results.append(p.groupby(["model", "recording_id", "label"], as_index=False).score.median())
        oldpred = pd.concat(old_results, ignore_index=True)
        oldpred.to_csv(out / "old_recordings_diagnostic.csv", index=False)
        pd.DataFrame([dict(model=m, **metrics(g)) for m, g in oldpred.groupby("model")]).to_csv(
            out / "old_metrics_diagnostic.csv", index=False
        )
    # Tap aggregation is evaluated using already held-out scores, never refitted to k.
    rng = np.random.default_rng(17)
    tap_rows = []
    for model in ["physics + LR", "MFCC + LR", "AST + LR"]:
        p = pred[pred.model == model]
        for k in [1, 3, 5, 10]:
            for repeat in range(100):
                fs = []
                for keys, g in p.groupby(["fold", "recording_id", "group", "label"]):
                    if len(g) < k:
                        continue
                    v = rng.choice(g.score.to_numpy(), k, replace=False)
                    fs.append(
                        dict(
                            fold=keys[0],
                            recording_id=keys[1],
                            group=keys[2],
                            label=keys[3],
                            score=float(np.median(v)),
                        )
                    )
                f = pd.DataFrame(fs).groupby(["recording_id", "group", "label"], as_index=False).score.mean()
                tap_rows.append(dict(model=model, k=k, repeat=repeat, **metrics(f)))
    pd.DataFrame(tap_rows).to_csv(out / "tap_count_resampling.csv", index=False)
    # Exact label permutations at GROUP level: only C(5,2)=10 allocations.
    group_ids = sorted(core.group.unique())
    n_bad = core.drop_duplicates("group").label.sum()
    perm = []
    for bad in combinations(group_ids, int(n_bad)):
        tmp = core.copy()
        tmp["label"] = tmp.group.isin(bad).astype(int)
        _, _, gg, _, _, _ = evaluate(tmp, {"MFCC + LR": SPECS["MFCC + LR"]})
        perm.append(dict(nok_groups=";".join(bad), **metrics(gg)))
    pd.DataFrame(perm).to_csv(out / "group_label_permutations.csv", index=False)
    (out / "model_settings.json").write_text(
        json.dumps(
            dict(
                positive_class="NOK crack",
                threshold=0.5,
                aggregation="median taps, mean held-out fits, median recordings per group",
                lr_C=0.1,
                xgb_trees=60,
                xgb_depth=2,
                xgb_lambda=10,
                seed=17,
                hyperparameter_search=False,
                primary="MFCC + LR",
                ast="frozen benchmark",
                group_status="user-authorized assumption of 3 distinct OK plates; not client-confirmed",
                excluded="OLD identities unknown; Bitten is a different defect; files under 2 seconds",
                weights="equal class weight, equal group weight within each class",
                confidence_intervals=False,
                caveat="2 NOK objects; repeated folds are not independent observations; codec and acquisition confounded",
            ),
            indent=2,
        )
    )
    print(scores.to_string(index=False), flush=True)
    return scores
