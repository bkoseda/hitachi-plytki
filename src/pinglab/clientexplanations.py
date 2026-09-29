"""Held-out feature importance and acquisition controls for the five-group pilot.

These diagnostics explain fitted models, not the physical cause of a crack.
No feature or hyperparameter is selected from these results for a claimed test.
"""

import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .clientmodels import evaluate, fit_model, group_folds, metrics
from .clientstudy import OUT_REL, decode, find_root, load
from .features import extract_features

MODELS = {
    "physics + LR": ("physics", "lr"),
    "physics + XGB": ("physics", "xgb"),
    "MFCC + LR": ("mfcc", "lr"),
    "MFCC + XGB": ("mfcc", "xgb"),
}


def file_scores(frame, score):
    result = frame[["recording_id", "group", "label"]].copy()
    result["score"] = score
    return result.groupby(["recording_id", "group", "label"], as_index=False).score.median()


def file_block_permutation(frame, rng):
    """Move full equal-size tap profiles between files, preserving within-file order."""
    blocks = list(frame.groupby("recording_id", sort=True).indices.values())
    if len({len(b) for b in blocks}) != 1:
        raise ValueError("Block permutation requires an equal number of taps per recording.")
    donor = rng.permutation(len(blocks))
    index = np.arange(len(frame))
    for dst, src in zip(blocks, donor):
        index[dst] = blocks[src]
    return index


def feature_groups(cols, family):
    if family == "mfcc":
        return {
            "ALL": cols,
            "ALL_means": [c for c in cols if c.endswith("_mean")],
            "ALL_stds": [c for c in cols if c.endswith("_std")],
        }
    return {
        "ALL": cols,
        "ALL_frequency": ["peak_hz", "centroid_hz", "bandwidth_hz", "rolloff85_hz"],
        "ALL_decay": ["decay10_s", "late_early_ratio"],
        "ALL_bands": [f"band_{i}" for i in range(1, 7)],
        "ALL_texture": ["flatness", "entropy", "peak_count"],
    }


def importances(root=None, repeats=30):
    root = find_root(root)
    out = root / OUT_REL / "explanations"
    out.mkdir(exist_ok=True)
    data = (
        pd.read_pickle(root / OUT_REL / "model_input.pkl")
        .sort_values(["recording_id", "onset_s"])
        .reset_index(drop=True)
    )
    raw, native, reference = [], [], []
    for fold, tr, te in group_folds(data):
        train, test = data.iloc[tr], data.iloc[te].reset_index(drop=True)
        for name, (family, kind) in MODELS.items():
            estimator, scaler, cols = fit_model(train, family, kind)
            x = scaler.transform(test[cols].to_numpy(float))
            score = estimator.predict_proba(x)[:, 1]
            baseline = file_scores(test, score)
            original = metrics(baseline)
            original_loss = log_loss(baseline.label, baseline.score, labels=[0, 1])
            reference.append(dict(model=name, fold=fold, logloss=original_loss, **original))
            coeff = estimator.coef_[0] if kind == "lr" else estimator.feature_importances_
            for c, value in zip(cols, coeff):
                native.append(dict(model=name, fold=fold, feature=c, value=float(value)))
            bundles = {c: [c] for c in cols} | feature_groups(cols, family)
            # Identical random donor permutations for each feature and model in a fold.
            for feature, bundle in bundles.items():
                rng = np.random.default_rng(1907)
                indices = [cols.index(c) for c in bundle]
                for repeat in range(repeats):
                    donor = file_block_permutation(test, rng)
                    changed = x.copy()
                    changed[:, indices] = x[donor][:, indices]
                    fs = file_scores(test, estimator.predict_proba(changed)[:, 1])
                    after = metrics(fs)
                    raw.append(
                        dict(
                            model=name,
                            fold=fold,
                            feature=feature,
                            repeat=repeat,
                            auc_drop=original["auc"] - after["auc"],
                            ba_drop=original["balanced_accuracy"] - after["balanced_accuracy"],
                            logloss_increase=log_loss(fs.label, fs.score, labels=[0, 1]) - original_loss,
                        )
                    )
        print("Importance:", fold, flush=True)
    raw = pd.DataFrame(raw)
    raw.to_csv(out / "permutation_raw.csv", index=False)
    pd.DataFrame(reference).to_csv(out / "importance_baselines.csv", index=False)
    native = pd.DataFrame(native)
    native.to_csv(out / "coefficients_and_gain_by_fold.csv", index=False)
    foldwise = raw.groupby(["model", "feature", "fold"], as_index=False)[
        ["auc_drop", "ba_drop", "logloss_increase"]
    ].mean()
    foldwise.to_csv(out / "importance_by_fold.csv", index=False)
    summary = (
        foldwise.groupby(["model", "feature"])
        .agg(
            auc_drop=("auc_drop", "mean"),
            ba_drop=("ba_drop", "mean"),
            logloss_increase=("logloss_increase", "mean"),
            min_fold_loss_increase=("logloss_increase", "min"),
            max_fold_loss_increase=("logloss_increase", "max"),
            positive_loss_folds=("logloss_increase", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    native["absolute"] = native.value.abs()
    ns = (
        native.groupby(["model", "feature"])
        .agg(
            mean_abs_weight_or_gain=("absolute", "mean"),
            mean_signed_weight_or_gain=("value", "mean"),
            positive_weight_folds=("value", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    summary = summary.merge(ns, how="left", on=["model", "feature"])
    summary.to_csv(out / "feature_importance.csv", index=False)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, name in zip(axs.flat, MODELS):
        top = (
            summary[(summary.model == name) & ~summary.feature.str.startswith("ALL")]
            .nlargest(7, "logloss_increase")
            .sort_values("logloss_increase")
        )
        ax.barh(top.feature, top.logloss_increase, color="#167f86")
        ax.axvline(0, color="gray", lw=0.7)
        ax.set(title=name, xlabel="Wzrost błędu log-loss po przetasowaniu cechy")
    fig.savefig(out / "importance.png", dpi=150)
    plt.close(fig)
    return summary


def mfcc_background(y, sr):
    """Same MFCC recipe, including silent windows as a lack-of-signal vector.

    Absolute energy is not a column of this model. Digital silence, quantization and
    noise spectral shape can nevertheless be recognizable. This is a negative control.
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    peak = np.max(np.abs(y))
    if peak < 1e-12:
        return {f"mfcc_{i:02d}_{s}": 0.0 for i in range(1, 14) for s in ["mean", "std"]}
    y /= peak
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        hop_length=sr // 200,
        n_mels=40,
        fmin=80,
        fmax=20000,
    )
    mf = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max, top_db=80), n_mfcc=14)
    return {f"mfcc_{i:02d}_{s}": float(getattr(mf[i], s)()) for i in range(1, 14) for s in ["mean", "std"]}


def quiet_windows(x, sr, onsets, length=0.1, guard_after=0.4, guard_before=0.1, max_windows=5):
    """Select low-RMS disjoint windows away from ALL detected candidates, label-blind.

    Undetected sounds may remain. These are background candidates, not verified clean
    ambient recordings. No replacement is made when a file has insufficient windows.
    """
    starts = np.arange(0.1, len(x) / sr - length - 0.1, length)
    onsets = np.asarray(onsets)
    choices = []
    for start in starts:
        end = start + length
        if np.any((start < onsets + guard_after) & (end > onsets - guard_before)):
            continue
        segment = x[round(start * sr) : round(end * sr)]
        choices.append((float(np.sqrt(np.mean(segment**2))), float(start), float(end)))
    return sorted(choices)[:max_windows]


def controls(root=None):
    root = find_root(root)
    out = root / OUT_REL / "explanations"
    out.mkdir(exist_ok=True)
    records, events, _ = load(root)
    core = pd.read_pickle(root / OUT_REL / "model_input.pkl")
    kinds = {k: [] for k in ["pre_hit_100ms", "quiet_100ms", "no_pre_150ms", "early_60ms", "late_90ms"]}
    audit = []
    for rid, file_rows in core.groupby("recording_id"):
        r = records.set_index("recording_id").loc[rid]
        a, sr = decode(root / r.path)
        x = a[:, int(r.selected_channel)].astype(float)
        for e in file_rows.itertuples():
            meta = dict(
                event_id=e.event_id, recording_id=rid, group=e.group, label=e.label, onset_s=e.onset_s
            )
            z = x[round((e.onset_s - 0.12) * sr) : round((e.onset_s - 0.02) * sr)]
            kinds["pre_hit_100ms"].append(meta | mfcc_background(z, sr))
            for kind, left, right in [
                ("no_pre_150ms", 0, 0.15),
                ("early_60ms", 0, 0.06),
                ("late_90ms", 0.06, 0.15),
            ]:
                z = x[round((e.onset_s + left) * sr) : round((e.onset_s + right) * sr)]
                kinds[kind].append(meta | extract_features(z, sr, onset_offset_s=0))
        onset = events[events.recording_id == rid].onset_s.to_numpy()
        selected = quiet_windows(x, sr, onset)
        for i, (rms, start, end) in enumerate(selected):
            z = x[round(start * sr) : round(end * sr)]
            meta = dict(
                event_id=f"{rid}_quiet_{i}", recording_id=rid, group=r.group, label=r.label, onset_s=start
            )
            kinds["quiet_100ms"].append(meta | mfcc_background(z, sr))
            audit.append(
                meta
                | dict(
                    start_s=start,
                    end_s=end,
                    rms_dbfs=20 * np.log10(max(rms, 1e-12)),
                    zero_fraction=float(np.mean(z == 0)),
                    filename=r.filename,
                )
            )
    rows = []
    for kind, data in kinds.items():
        frame = pd.DataFrame(data)
        frame.to_pickle(out / f"input_{kind}.pkl")
        _, files, _, result, folds, split = evaluate(
            frame, {"MFCC + LR": ("mfcc", "lr"), "MFCC + XGB": ("mfcc", "xgb")}
        )
        files.to_csv(out / f"predictions_{kind}.csv", index=False)
        folds.to_csv(out / f"folds_{kind}.csv", index=False)
        (out / f"splits_{kind}.json").write_text(json.dumps(split, indent=2))
        r = result[result.level == "recording"].copy()
        r["control"] = kind
        r["n_groups"] = frame.group.nunique()
        r["n_windows"] = len(frame)
        rows.append(r)
        print("Control:", kind, len(frame), "windows", frame.recording_id.nunique(), "files", flush=True)
    pd.DataFrame(audit).to_csv(out / "quiet_window_audit.csv", index=False)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out / "control_results.csv", index=False)
    # Background levels per file, not per tap. Quiet windows can be digitally gated.
    quiet = (
        pd.DataFrame(audit)
        .groupby(["recording_id", "group", "label"], as_index=False)
        .agg(
            rms_dbfs=("rms_dbfs", "median"),
            zero_fraction=("zero_fraction", "median"),
            windows=("event_id", "size"),
        )
    )
    quiet.to_csv(out / "quiet_by_file.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for group, g in quiet.groupby("group"):
        ax.scatter([group] * len(g), g.rms_dbfs, label=group)
        for e in g.itertuples():
            ax.annotate(e.recording_id, (group, e.rms_dbfs), fontsize=7)
    ax.set(
        ylabel="RMS [dBFS] · −240 = numeryczna podłoga dla zer",
        title="Najcichsze okna z dala od wykryć · każdy punkt to plik",
    )
    fig.savefig(out / "background_levels.png", dpi=150)
    plt.close(fig)
    return summary


def run(root=None, repeats=30):
    importances(root, repeats=repeats)
    return controls(root)
