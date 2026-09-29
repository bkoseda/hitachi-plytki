"""Acquisition diagnostics and controlled perturbations for the client pilot."""

import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .clientmodels import SPECS
from .clientstudy import OUT_REL, decode, find_root, load
from .features import extract_features, feature_columns

COLORS = {"Bad_1": "#bf593d", "Bad_2": "#e69546", "OK_1": "#167f86", "OK_2": "#437fbb", "OK_3": "#73a991"}


def figures(root=None):
    root = find_root(root)
    out = root / OUT_REL
    m, e, f = load(root)
    r = pd.read_csv(out / "recording_features.csv")
    core = r[(r.scope == "core") & (r.n_accepted > 0)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    order = m.sort_values(["batch", "recording_id"])
    axs[0].barh(
        order.recording_id,
        order.duration_s,
        color=["#167f86" if b == "2-batch-ok" else "#bf593d" for b in order.batch],
    )
    axs[0].set(xlabel="Czas [s]", title="45 nagrań · tło też jest plikiem")
    axs[0].tick_params(axis="y", labelsize=7)
    axs[1].scatter(m.duration_s, m.n_accepted, c=np.where(m.batch == "2-batch-ok", "#167f86", "#bf593d"))
    for rr in m[m.duration_s < 2].itertuples():
        axs[1].annotate(rr.recording_id, (rr.duration_s, rr.n_accepted))
    axs[1].set(
        xlabel="Czas nagrania [s]",
        ylabel="Kandydaci użyci do cech",
        title="Trzy krótkie pliki nie wchodzą do modeli",
    )
    fig.savefig(out / "figures/inventory.png", dpi=140)
    plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    for ax, rid in zip(axs, ["R16", "G13", "G06"]):
        row = m.set_index("recording_id").loc[rid]
        a, sr = decode(root / row.path)
        x = a[:, int(row.selected_channel)].astype(float)
        n = sr // 100
        db = 10 * np.log10(np.maximum(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1), 1e-24))
        ax.plot((np.arange(len(db)) + 0.5) * 0.01, np.maximum(db, -140), lw=0.8)
        ax.axhline(-100, color="#bf593d", ls="--", lw=0.7)
        ax.set(title=f"{rid} · {row.filename}", ylim=(-145, 0), ylabel="RMS [dBFS]", xlabel="Czas [s]")
    fig.savefig(out / "figures/quiet_intervals.png", dpi=140)
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    review_ids = ["R02", "R06", "G06", "G14", "G20", "G11"]
    for ax, rid in zip(axs.flat, review_ids):
        x, sr = sf.read(out / "decoded" / f"{rid}.wav")
        start = 5 if len(x) / sr > 8 else 0
        end = min(len(x) / sr, start + 2)
        z = x[int(start * sr) : int(end * sr)]
        ax.plot(np.arange(len(z)) / sr + start, z, lw=0.5)
        for ee in e[(e.recording_id == rid) & e.onset_s.between(start, end)].itertuples():
            ax.axvline(ee.onset_s, color="#167f86" if ee.accepted else "#c18b30", alpha=0.6, lw=0.7)
        ax.set(title=rid, xlabel="Czas [s]", ylabel="Amplituda")
    fig.savefig(out / "figures/detector_review.png", dpi=150)
    plt.close(fig)
    # Median normalized power per file; each file gets equal visual weight.
    spec = []
    freqs = None
    for rid, g in f[f.scope == "core"].groupby("recording_id"):
        powers = []
        for row in g.iloc[
            np.unique(np.linspace(0, len(g) - 1, min(len(g), 10)).round().astype(int))
        ].itertuples():
            y, sr = sf.read(out / "clips" / f"{row.event_id}.wav")
            y = y / max(np.max(np.abs(y)), 1e-12)
            freqs, p = welch(y[round((row.onset_offset_s + 0.01) * sr) :], sr, nperseg=4096, nfft=4096)
            powers.append(p / max(p.sum(), 1e-12))
        spec.append((rid, np.median(powers, axis=0)))
    spectra = dict(spec)
    np.savez_compressed(out / "spectra_core.npz", frequency_hz=freqs, **spectra)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, subset, title in [
        (axs[0], core, "Wszystkie warunki"),
        (
            axs[1],
            core[(core.support == "4-point") & ~core.tool_change & ~core.angle],
            "4 podpory · bez zmiany narzędzia",
        ),
    ]:
        for group, g in subset.groupby("group"):
            p = np.median([spectra[rid] for rid in g.recording_id], axis=0)
            ax.semilogx(freqs, 10 * np.log10(np.maximum(p, 1e-12)), label=group, color=COLORS[group])
        ax.set(xlim=(100, 20000), ylim=(-90, 0), title=title, xlabel="Hz", ylabel="Względna energia [dB]")
        ax.legend(fontsize=8)
    fig.savefig(out / "figures/spectra.png", dpi=150)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for group, g in core.groupby("group"):
        for support, gg in g.groupby("support"):
            marker = {"4-point": "o", "1-point": "^", "unknown": "s"}[support]
            axs[0].scatter(
                gg.peak_hz, gg.centroid_hz, color=COLORS[group], marker=marker, label=f"{group} {support}"
            )
            for row in gg.itertuples():
                axs[0].annotate(row.recording_id, (row.peak_hz, row.centroid_hz), fontsize=7)
    axs[0].set(
        xlabel="Dominujący pik [Hz]", ylabel="Środek ciężkości widma [Hz]", title="Każdy punkt = nagranie"
    )
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].boxplot(
        [core[core.group == g].centroid_hz for g in COLORS], tick_labels=list(COLORS), showfliers=True
    )
    axs[1].set(ylabel="Środek ciężkości widma [Hz]", title="Zmienność nagrań tej samej grupy")
    fig.savefig(out / "figures/conditions.png", dpi=150)
    plt.close(fig)
    if (out / "representations.pkl").exists():
        reps = pd.read_pickle(out / "representations.pkl")
        reps = reps[reps.scope == "core"]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, fam in zip(axs, ["physics", "mfcc", "ast"]):
            cols = feature_columns(reps, fam)
            agg = reps.groupby("recording_id")[cols].median()
            x = agg.to_numpy()
            x = StandardScaler().fit_transform(x) if fam != "ast" else x
            pca = PCA(2).fit(x)
            pc = pca.transform(x)
            for rid, xy in zip(agg.index, pc):
                rr = m.set_index("recording_id").loc[rid]
                ax.scatter(*xy, color=COLORS[rr.group], marker="^" if rr.support == "1-point" else "o")
                ax.annotate(rid, xy, fontsize=6, xytext=(2, 2), textcoords="offset points")
            ax.set(
                title=f"{fam} · PCA tylko do oglądania",
                xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
                ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
            )
        fig.savefig(out / "figures/pca.png", dpi=150)
        plt.close(fig)
    # Simple measures to quantify recording-protocol problems, not infer specific filters.
    duplicate = m[m.duplicated("pcm_sha256", keep=False)][["recording_id", "path", "pcm_sha256"]]
    duplicate.to_csv(out / "exact_decoded_duplicates.csv", index=False)
    facts = dict(
        audio_files=len(m),
        duration_s=float(m.duration_s.sum()),
        accepted_candidates=len(f),
        core_files=int(((m.scope == "core") & (m.n_accepted > 0)).sum()),
        core_candidates=int((f.scope == "core").sum()),
        n_files_zero_run_100ms=int((m.longest_zero_s > 0.1).sum()),
        n_ok_files_zero_run_100ms=int(((m.batch == "2-batch-ok") & (m.longest_zero_s > 0.1)).sum()),
        all_dual_mono=bool(m.dual_mono.all()),
        max_clip_fraction=float(m.clipping_fraction.max()),
        median_file_gap_s=float(m[m.n_accepted > 0].median_gap_s.median()),
        background_seconds=float(m[m.scope == "background"].duration_s.sum()),
        short_files=m[m.duration_s < 2].recording_id.tolist(),
    )
    (out / "quality_facts.json").write_text(json.dumps(facts, indent=2))
    return facts


def perturbations(root=None):
    root = find_root(root)
    out = root / OUT_REL
    core = pd.read_pickle(out / "model_input.pkl")
    noise, sr = sf.read(out / "decoded/R16.wav")
    noise = noise[19 * sr : 30 * sr]
    from .embeddings import ASTEncoder

    # Single-thread PyTorch avoids an OpenMP pool conflict after XGBoost on macOS.
    encoder = ASTEncoder(root / "models/ast", offline=True, threads=1)
    cache = root / "data/cache/client-ast-perturbations"
    cache.mkdir(parents=True, exist_ok=True)
    names = [
        "physics + LR",
        "physics + XGB",
        "MFCC + LR",
        "MFCC + XGB",
        "MFCC + physics + LR",
        "MFCC + physics + XGB",
        "AST + LR",
        "AST + XGB",
    ]
    specs = {k: SPECS[k] for k in names}
    variants, audit, all_predictions = [], [], []
    for variant, snr in [("gain -12 dB", None), ("factory SNR 20 dB", 20), ("factory SNR 10 dB", 10)]:
        rows, vectors = [], []
        for i, r in enumerate(core.itertuples()):
            y, sr = sf.read(out / "clips" / f"{r.event_id}.wav")
            if snr is None:
                z = y * 10 ** (-12 / 20)
            else:
                rng = np.random.default_rng(1700 + i)
                a = int(rng.integers(0, len(noise) - len(y)))
                n = noise[a : a + len(y)].copy()
                n -= n.mean()
                n *= np.sqrt(np.mean(y * y)) / max(np.sqrt(np.mean(n * n)), 1e-12) / 10 ** (snr / 20)
                z = y + n
                measured = 20 * np.log10(np.sqrt(np.mean(y * y)) / np.sqrt(np.mean(n * n)))
                assert abs(measured - snr) < 1e-8
                audit.append(
                    dict(
                        event_id=r.event_id,
                        variant=variant,
                        requested_snr_db=snr,
                        measured_snr_db=float(measured),
                        noise_start_s=19 + a / sr,
                    )
                )
            rows.append(extract_features(z, sr, r.onset_offset_s))
            key = hashlib.sha256(
                z.astype(np.float32).tobytes() + str(sr).encode() + encoder.revision.encode() + b"peak095-v1"
            ).hexdigest()
            path = cache / f"{key}.npy"
            if path.exists():
                vector = np.load(path, allow_pickle=False)
            else:
                vector = encoder.encode(z, sr)
                np.save(path, vector)
            assert vector.shape == (768,) and np.isfinite(vector).all()
            vectors.append(vector)
            if (i + 1) % 30 == 0:
                print(f"{variant}: AST {i + 1}/{len(core)}", flush=True)
        changed = core.copy()
        for col in rows[0]:
            changed[col] = [row[col] for row in rows]
        changed.loc[:, feature_columns(changed, "ast")] = np.stack(vectors)
        from .clientmodels import fit_predict, group_folds, metrics

        predictions = []
        for fold, tr, te in group_folds(core):
            for name, (family, kind) in specs.items():
                test = changed.iloc[te]
                p = test[["recording_id", "group", "label"]].copy()
                p["score"] = fit_predict(core.iloc[tr], test, family, kind)
                p["model"] = name
                p["fold"] = fold
                predictions.append(p)
        p = (
            pd.concat(predictions)
            .groupby(["model", "fold", "recording_id", "group", "label"], as_index=False)
            .score.median()
        )
        p = p.groupby(["model", "recording_id", "group", "label"], as_index=False).score.mean()
        p["variant"] = variant
        all_predictions.append(p)
        for model, g in p.groupby("model"):
            variants.append(dict(variant=variant, model=model, **metrics(g)))
        print("Perturbation", variant, flush=True)
    baseline = pd.read_csv(out / "model_comparison.csv")
    baseline = baseline[(baseline.level == "recording") & baseline.model.isin(names)].drop(columns="level")
    baseline["variant"] = "original"
    result = pd.concat([baseline, pd.DataFrame(variants)], ignore_index=True)
    result.to_csv(out / "noise_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "noise_snr_audit.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "noise_recording_predictions.csv", index=False)
    return result
