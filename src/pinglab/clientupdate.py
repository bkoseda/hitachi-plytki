"""September 30 delivery: compare acquisition and test explicitly provisional identities."""

from __future__ import annotations

import hashlib
import json
from textwrap import fill

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch

from . import clientstudy as old
from .clientmodels import SPECS, evaluate, fit_model, metrics

OUT_REL = "reports/runs/client-update-20260930"
NAMES = [
    "physics + LR",
    "physics + XGB",
    "MFCC + LR",
    "MFCC + XGB",
    "MFCC + physics + LR",
    "MFCC + physics + XGB",
    "AST + LR",
    "AST + XGB",
    "QC no codec + LR",
]
SCENARIOS = {
    "same_BAD2": "Nowe BAD2 = stare Bad_2; nowe OK = OK_1. Stare Bad_1 pozostaje osobną płytką.",
    "new_BAD2_merge_old": "Nowe BAD2 jest nowe; stare Bad_1 i Bad_2 traktujemy jako jedną sztukę. Nowe OK = OK_1.",
    "new_BAD2_only_old_Bad1": "Nowe BAD2 jest nowe; tylko stare Bad_1 jest pewną klasą NOK. Stare Bad_2 wyłączone. Nowe OK = OK_1.",
    "new_BAD2_only_old_Bad2": "Nowe BAD2 jest nowe; tylko stare Bad_2 jest pewną klasą NOK. Stare Bad_1 wyłączone. Nowe OK = OK_1.",
}


def inventory(root=None):
    root = old.find_root(root)
    rows = []
    archive = json.loads((root / "config/client_archives_20260930.json").read_text())
    lookup = {r["path"]: r for r in archive}
    for batch, prefix in [("3-batch-enhancements-off", "E"), ("4-batch-lab-test", "L")]:
        for i, path in enumerate(sorted((root / "assets" / batch).rglob("*.wav")), 1):
            rel = str(path.relative_to(root))
            name = path.stem.lower()
            bg = "noise" in name
            bad = "BAD2" in path.parts
            rows.append(
                dict(
                    recording_id=f"{prefix}{i:02d}",
                    path=rel,
                    filename=str(path.relative_to(root / "assets" / batch)),
                    batch=batch,
                    group="BACKGROUND_NEW"
                    if bg
                    else "NOK_LAB_BAD2"
                    if bad
                    else "OK_ENH_OFF"
                    if prefix == "E"
                    else "OK_LAB",
                    scope="background" if bg else "core",
                    label=-1 if bg else int(bad),
                    group_status="physical identity unconfirmed; do not infer from take number",
                    support="unknown",
                    strike_location="fixed_by_name"
                    if ("one point" in name or "single point" in name)
                    else "unknown",
                    angle="angle" in name,
                    tool_change=False,
                    codec="PCM",
                    processing="off_by_folder_name" if prefix == "E" else "unknown_LAB_TEST",
                    acquisition="unknown; LAB may denote a different location"
                    if prefix == "L"
                    else "folder says enhancements off",
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    archive=lookup[rel]["archive"],
                )
            )
    return pd.DataFrame(rows)


def prepare(root=None):
    root = old.find_root(root)
    out = root / OUT_REL
    if not all((root / old.OUT_REL / name).exists() for name in ["manifest.csv", "features.csv"]):
        old.prepare(root)
    new, events, features = old.prepare(root, inventory_frame=inventory(root), out_rel=OUT_REL)
    previous = pd.read_csv(root / old.OUT_REL / "manifest.csv")
    previous["processing"] = "unknown_old"
    all_records = pd.concat([previous, new], ignore_index=True)
    all_records.to_csv(out / "all_recordings.csv", index=False)
    all_records[all_records.duplicated("sha256", keep=False)].to_csv(out / "duplicate_files.csv", index=False)
    all_records[all_records.duplicated("pcm_sha256", keep=False)].to_csv(
        out / "duplicate_pcm.csv", index=False
    )
    all_features = pd.concat([pd.read_csv(root / old.OUT_REL / "features.csv"), features], ignore_index=True)
    all_features.to_csv(out / "all_features.csv", index=False)
    summary = all_records.groupby("batch").agg(
        files=("recording_id", "size"),
        seconds=("duration_s", "sum"),
        files_zero100ms=("longest_zero_s", lambda s: int((s > 0.1).sum())),
        median_zero_fraction=("zero_fraction", "median"),
        median_floor_dbfs=("rms_p10_dbfs", "median"),
        median_peak_dbfs=("peak_dbfs", "median"),
        max_clipping=("clipping_fraction", "max"),
        accepted=("n_accepted", "sum"),
        median_gap_s=("median_gap_s", "median"),
    )
    summary.to_csv(out / "batch_quality.csv")
    return all_records, events, all_features


def representations(root=None):
    root = old.find_root(root)
    out = root / OUT_REL
    new = old.embeddings(root, out_rel=OUT_REL)
    if not (root / old.OUT_REL / "representations.pkl").exists():
        old.embeddings(root)
    previous = pd.read_pickle(root / old.OUT_REL / "representations.pkl")
    data = pd.concat([previous, new], ignore_index=True)
    records = pd.read_csv(out / "all_recordings.csv")
    qc = records[
        ["recording_id", "peak_dbfs", "rms_p10_dbfs", "below_minus100_fraction", "duration_s"]
    ].copy()
    qc = qc.rename(columns={c: "file_" + c for c in qc if c != "recording_id"})
    data = data.merge(qc, on="recording_id", validate="many_to_one")
    assert data.event_id.is_unique
    data.to_pickle(out / "all_representations.pkl")
    return data


def scenario_data(frame, name):
    """Alias possibly identical objects BEFORE splitting; no filename suffix becomes an ID."""
    d = frame[frame.scope.eq("core")].copy()
    d.loc[d.group.isin(["OK_ENH_OFF", "OK_LAB"]), "group"] = "OK_1"
    if name == "same_BAD2":
        d.loc[d.group.eq("NOK_LAB_BAD2"), "group"] = "Bad_2"
    elif name == "new_BAD2_merge_old":
        d.loc[d.group.isin(["Bad_1", "Bad_2"]), "group"] = "OLD_NOK_SHARED"
    elif name == "new_BAD2_only_old_Bad1":
        d = d[~d.group.eq("Bad_2")].copy()
    elif name == "new_BAD2_only_old_Bad2":
        d = d[~d.group.eq("Bad_1")].copy()
    else:
        raise ValueError(name)
    return d.reset_index(drop=True)


def file_predictions(model, scaler, cols, data):
    p = data[["recording_id", "group", "label"]].copy()
    p["score"] = model.predict_proba(scaler.transform(data[cols].to_numpy(float)))[:, 1]
    return p.groupby(["recording_id", "group", "label"], as_index=False, dropna=False).score.median()


def model(root=None):
    root = old.find_root(root)
    out = root / OUT_REL
    data = pd.read_pickle(out / "all_representations.pkl")
    previous = data[data.batch.isin(["1-batch-ping", "2-batch-ok"]) & data.scope.eq("core")]
    new = data[data.batch.isin(["3-batch-enhancements-off", "4-batch-lab-test"]) & data.scope.eq("core")]
    assert set(previous.recording_id).isdisjoint(new.recording_id)
    preds = []
    scores = []
    modeldir = root / "models/client-20260930-before-new-label-fit"
    modeldir.mkdir(exist_ok=True)
    for name in NAMES:
        est, scaler, cols = fit_model(previous, *SPECS[name])
        p = file_predictions(est, scaler, cols, new)
        p["model"] = name
        preds.append(p)
        scores.append(dict(experiment="old_to_new", model=name, **metrics(p)))
        joblib.dump(
            dict(
                model=est,
                scaler=scaler,
                columns=cols,
                threshold=0.5,
                trained_event_ids=previous.event_id.tolist(),
            ),
            modeldir / (name.replace(" ", "_") + ".joblib"),
        )
    predictions = pd.concat(preds, ignore_index=True)
    predictions = predictions.merge(
        new[["recording_id", "batch"]].drop_duplicates(), on="recording_id", validate="many_to_one"
    )
    predictions.to_csv(out / "old_to_new_predictions.csv", index=False)
    by_batch = []
    for (name, batch), p in predictions.groupby(["model", "batch"]):
        if p.label.nunique() == 1:
            vals = dict(
                n=len(p),
                n_nok=int(p.label.sum()),
                accuracy=float((p.score < 0.5).mean()),
                balanced_accuracy=np.nan,
                recall_nok=np.nan,
                false_reject_ok=float((p.score >= 0.5).mean()),
                auc=np.nan,
                tp=0,
                fn=0,
                tn=int((p.score < 0.5).sum()),
                fp=int((p.score >= 0.5).sum()),
            )
        else:
            vals = metrics(p)
        by_batch.append(dict(model=name, batch=batch, **vals))
    pd.DataFrame(by_batch).to_csv(out / "old_to_new_by_batch.csv", index=False)
    pd.DataFrame(scores).to_csv(out / "old_to_new_metrics.csv", index=False)
    (out / "old_to_new_training.json").write_text(
        json.dumps(
            dict(
                train_recordings=sorted(previous.recording_id.unique()),
                test_recordings=sorted(new.recording_id.unique()),
                threshold=0.5,
                hyperparameter_search=False,
                caveat="New recordings, not proven new plates. Old labels retained for reproducibility; identities disputed.",
            ),
            indent=2,
        )
    )
    combined = []
    allpred = []
    audits = {}
    for scenario in SCENARIOS:
        d = scenario_data(data, scenario)
        _, files, groups, result, folds, audit = evaluate(d, {n: SPECS[n] for n in NAMES})
        result["scenario"] = scenario
        combined.append(result)
        files["scenario"] = scenario
        allpred.append(files)
        audits[scenario] = dict(assumption=SCENARIOS[scenario], splits=audit)
        folds.to_csv(out / f"folds_{scenario}.csv", index=False)
        print("Evaluated identity scenario:", scenario, flush=True)
    combined = pd.concat(combined, ignore_index=True)
    combined.to_csv(out / "combined_metrics.csv", index=False)
    pd.concat(allpred, ignore_index=True).to_csv(out / "combined_predictions.csv", index=False)
    (out / "identity_scenarios.json").write_text(json.dumps(audits, indent=2))
    # Chipped and OLD stay out of crack training; inspect their responses without claiming crack recall.
    unknown = data[data.scope.isin(["chipped", "old_unknown"])]
    d = scenario_data(data, "same_BAD2")
    supplemental = []
    for name in ["MFCC + LR", "MFCC + XGB"]:
        est, scaler, cols = fit_model(d, *SPECS[name])
        p = file_predictions(est, scaler, cols, unknown)
        p["model"] = name
        supplemental.append(p)
    pd.concat(supplemental, ignore_index=True).to_csv(
        out / "excluded_diagnostic_predictions.csv", index=False
    )
    return pd.DataFrame(scores), combined


def audio_path(root, rid):
    return root / (OUT_REL if rid.startswith(("E", "L")) else old.OUT_REL) / "decoded" / f"{rid}.wav"


def figures(root=None):
    root = old.find_root(root)
    out = root / OUT_REL
    m = pd.read_csv(out / "all_recordings.csv")
    new = m[m.recording_id.str.startswith(("E", "L"))]
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    for batch, g in m.groupby("batch"):
        axs[0].scatter(g.recording_id, g.rms_p10_dbfs.clip(lower=-120), label=batch, s=20)
        axs[1].scatter(g.recording_id, g.longest_zero_s, label=batch, s=20)
    axs[0].set(
        ylabel="Dolny poziom RMS [dBFS]", title="RMS p10 całego pliku; cyfrowe zero pokazane na −120 dBFS"
    )
    axs[1].set(ylabel="Najdłuższy ciąg zer [s]")
    for ax in axs:
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    axs[0].legend(fontsize=8, ncol=2)
    fig.savefig(out / "figures/quality.png", dpi=130)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, rid in zip(axs.flat, ["R16", "E03", "L02", "L08"]):
        a, sr = old.decode(root / m.set_index("recording_id").loc[rid, "path"])
        x = a[:, 0]
        n = sr // 100
        rms = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1))
        ax.plot(np.arange(len(rms)) * 0.01, 20 * np.log10(np.maximum(rms, 1e-12)), lw=0.7)
        ax.set(
            title=fill(f"{rid} · " + m.set_index("recording_id").loc[rid, "filename"], 55),
            ylabel="RMS [dBFS]",
            xlabel="Czas [s]",
            ylim=(-145, 0),
        )
    for ax in axs.flat:
        ax.title.set_fontsize(10)
    fig.savefig(out / "figures/envelopes.png", dpi=130)
    plt.close(fig)
    # Onset and spectrogram atlas for every new file, downsample waveform display only.
    events = pd.read_csv(out / "candidate_audit.csv")
    for row in new.itertuples():
        x, sr = sf.read(audio_path(root, row.recording_id))
        step = max(1, len(x) // 18000)
        fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
        axs[0].plot(np.arange(0, len(x), step) / sr, x[::step], lw=0.5, color="#167f86")
        for e in events[events.recording_id.eq(row.recording_id)].itertuples():
            axs[0].axvline(e.onset_s, color="#167f86" if e.accepted else "#c78536", alpha=0.6, lw=0.6)
        with np.errstate(divide="ignore"):
            axs[1].specgram(x, NFFT=2048, Fs=sr, noverlap=1024, cmap="magma", vmin=-120, vmax=-20)
        axs[0].set(title=f"{row.recording_id} · {row.filename}", ylabel="Amplituda")
        axs[1].set(ylim=(0, 20000), xlabel="Czas [s]", ylabel="Hz")
        fig.savefig(out / f"figures/{row.recording_id}.png", dpi=120)
        plt.close(fig)
    # Frequency-domain comparison, same estimator and original sample rate.
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for rid in ["R16", "E03"]:
        x, sr = sf.read(audio_path(root, rid))
        f, p = welch(x, sr, nperseg=4096)
        ax.semilogx(f, 10 * np.log10(np.maximum(p, 1e-24)), label=rid)
    ax.set(
        xlim=(80, 20000),
        ylabel="PSD [dB/Hz]",
        xlabel="Hz",
        title="Całe pliki tła; inne sesje, nie kontrolowany test samego przełącznika",
    )
    ax.legend()
    fig.savefig(out / "figures/background.png", dpi=130)
    plt.close(fig)
    return m
