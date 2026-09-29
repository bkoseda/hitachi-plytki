"""Descriptive nuisance/noise/embedding comparisons, not classifier validation."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinglab.embeddings import ASTEncoder  # noqa: E402 — local package path configured above
from pinglab.features import (  # noqa: E402 — local package path configured above
    extract_features,
    feature_columns,
)

OUT = ROOT / "reports/runs/client-20260929"


def run(run_ast=True):
    e = pd.read_csv(OUT / "features.csv")
    r = pd.read_csv(OUT / "manifest.csv").set_index("recording_id")
    spectra = np.load(OUT / "spectra.npz")
    f = spectra["frequency_hz"]
    groups = [
        ("Bad 1", ["R01", "R02", "R03", "R04"]),
        ("Bad 2", ["R05", "R06", "R07", "R08", "R09"]),
        ("OLD: OK", ["R20", "R21", "R22", "R23", "R24"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for ax, (title, ids) in zip(axes, groups):
        for rid in ids:
            p = spectra[rid]
            p = p / max(p.sum(), 1e-12)
            ax.semilogx(f, 10 * np.log10(np.maximum(p, 1e-15)), label=rid, lw=1)
        ax.set(
            title=title + " · różne nagrania / ustawienia",
            xlim=(100, 20000),
            ylim=(-90, -5),
            xlabel="Hz",
            ylabel="Widmo / suma [dB]",
        )
        ax.legend(ncol=5)
    fig.savefig(OUT / "figures" / "spectral_conditions.png", dpi=140)
    plt.close(fig)
    # Near-crack location is a filename annotation, not a confirmed per-hit label.
    loc = []
    for rid in ["R07", "R08"]:
        sub = e[e.recording_id == rid].copy()
        sub["region"] = np.where(sub.onset_s.between(14, 22), "14–22 s (opis pliku)", "pozostałe")
        for region, g in sub.groupby("region"):
            loc.append(
                dict(
                    recording_id=rid,
                    region=region,
                    n=len(g),
                    peak_hz_median=g.peak_hz.median(),
                    centroid_hz_median=g.centroid_hz.median(),
                    decay10_median=g.decay10_s.median(),
                )
            )
    pd.DataFrame(loc).to_csv(OUT / "near_crack.csv", index=False)
    # Representative events selected by time quantiles, independent of model separation.
    ids = []
    for _, g in e.groupby("recording_id", sort=True):
        ids.extend(
            g.sort_values("onset_s")
            .iloc[np.unique(np.linspace(0, len(g) - 1, 5).round().astype(int))]
            .event_id
        )
    selected = e.set_index("event_id").loc[ids].reset_index()
    selected.to_csv(OUT / "representative_events.csv", index=False)
    noise, sr = sf.read(OUT / "decoded/R16.wav", dtype="float32")
    # Do not synthesize with the 11–18 s muted interval. One contiguous segment per event.
    noise_pool = noise[round(19 * sr) : round(30 * sr)]
    scalers = {
        family: StandardScaler().fit(e[feature_columns(e, family)].groupby(e.recording_id).median())
        for family in ["physics", "mfcc", "mel"]
    }
    rows = []
    for i, row in enumerate(selected.itertuples()):
        y, sr = sf.read(OUT / "clips" / f"{row.event_id}.wav", dtype="float32")
        rng = np.random.default_rng(400 + i)
        a = rng.integers(0, len(noise_pool) - len(y))
        n = noise_pool[a : a + len(y)]
        base = extract_features(y, sr, row.onset_offset_s)
        for variant, snr in [
            ("gain -12 dB", None),
            ("factory SNR 20 dB", 20),
            ("factory SNR 10 dB", 10),
            ("factory SNR 5 dB", 5),
            ("factory SNR 0 dB", 0),
        ]:
            if snr is None:
                z = y * 10 ** (-12 / 20)
            else:
                n0 = n - n.mean()
                scale = np.sqrt(np.mean(y * y)) / (max(np.sqrt(np.mean(n0 * n0)), 1e-12) * 10 ** (snr / 20))
                z = y + n0 * scale
            features = extract_features(z, sr, row.onset_offset_s)
            for family, scaler in scalers.items():
                cols = feature_columns(e, family)
                shift = np.sqrt(
                    np.mean(
                        (
                            (np.array([features[c] for c in cols]) - np.array([base[c] for c in cols]))
                            / scaler.scale_
                        )
                        ** 2
                    )
                )
                rows.append(
                    dict(
                        event_id=row.event_id,
                        recording_id=row.recording_id,
                        family=family,
                        variant=variant,
                        scaled_feature_shift=shift,
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "noise_feature_shift.csv", index=False)
    pivot = df.groupby(["family", "variant"]).scaled_feature_shift.median().unstack()
    pivot.to_csv(OUT / "noise_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    pivot[
        ["gain -12 dB", "factory SNR 20 dB", "factory SNR 10 dB", "factory SNR 5 dB", "factory SNR 0 dB"]
    ].T.plot.bar(ax=ax)
    ax.set(
        title="Zmiana reprezentacji po dodaniu nagranego tła",
        ylabel="Mediana przesunięcia cech / skala nagrań",
        xlabel="",
    )
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(OUT / "figures" / "noise_features.png", dpi=140)
    plt.close(fig)
    print("Noise complete", len(selected), "events", flush=True)
    if run_ast:
        cache = ROOT / "data/cache/client-20260929-ast"
        cache.mkdir(parents=True, exist_ok=True)
        encoder = ASTEncoder(ROOT / "models/ast", offline=True)
        vectors = []
        for i, row in enumerate(selected.itertuples()):
            y, sr = sf.read(OUT / "clips" / f"{row.event_id}.wav", dtype="float32")
            key = hashlib.sha256(y.tobytes() + encoder.revision.encode()).hexdigest()
            file = cache / f"{key}.npy"
            if file.exists():
                v = np.load(file, allow_pickle=False)
            else:
                v = encoder.encode(y, sr)
                np.save(file, v)
            vectors.append(v)
            if i % 20 == 0:
                print("AST", i + 1, "/", len(selected), flush=True)
        a = pd.concat(
            [
                selected[["event_id", "recording_id", "scope"]],
                pd.DataFrame(vectors, columns=[f"ast_{i:03d}" for i in range(768)]),
            ],
            axis=1,
        )
        a.to_csv(OUT / "ast_representatives.csv", index=False)
        cols = feature_columns(a, "ast")
        g = a.groupby("recording_id")[cols].mean()
        z = g.to_numpy()
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        pca = PCA(2)
        pc = pca.fit_transform(z)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        colors = {"ok": "#137f83", "nok": "#c56838", "chipped": "#8c69ad"}
        for rid, xy in zip(g.index, pc):
            ax.scatter(
                *xy, color=colors[r.loc[rid, "scope"]], marker="^" if r.loc[rid, "folder"] == "OLD" else "o"
            )
            ax.annotate(rid, xy, xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set(
            title="AST · 5 zdarzeń / nagranie · opis, bez testu klasyfikacji",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
        )
        fig.savefig(OUT / "figures" / "pca_ast.png", dpi=140)
        plt.close(fig)
        (OUT / "ast_model.json").write_text(
            json.dumps(
                dict(
                    revision=encoder.revision,
                    n_events=len(selected),
                    per_recording=5,
                    selection="equally spaced event indices in time order",
                    fine_tuned=False,
                ),
                indent=2,
            )
        )
    return pivot


if __name__ == "__main__":
    run("--no-ast" not in sys.argv)
