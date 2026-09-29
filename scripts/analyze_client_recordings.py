"""Local first-pass client audio audit. No class-performance claims without plate IDs.

Usage: .venv/bin/python scripts/analyze_client_recordings.py [recordings_directory]
Uses the same pinglab decoding, impact detection and features as the notebooks.
Outputs are private (reports/runs); originals are never modified.
"""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinglab.audio import (  # noqa: E402 — local package path configured above
    Config,
    cut_hits,
    detect_hits,
    load_audio,
)
from pinglab.features import (  # noqa: E402 — local package path configured above
    decay_curve,
    extract_features,
    feature_columns,
)
from pinglab.recorder import segment_ringdown  # noqa: E402 — local package path configured above

OUT = ROOT / "reports/runs/client-20260929"
SR = 48000
COLORS = {"ok": "#137f83", "nok": "#c56838", "chipped": "#8c69ad", "background": "#697580"}


def native(path):
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)


def intervals(mask, hop=0.005, minimum=0.1):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (round(a * hop, 3), round(b * hop, 3))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])
        if (b - a) * hop >= minimum
    ]


def run(source):
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Brak katalogu nagrań: {source}")
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ["decoded", "figures", "clips"]:
        (OUT / sub).mkdir(exist_ok=True)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in [".wav", ".m4a", ".mp3", ".flac"])
    if not files:
        raise ValueError("Brak obsługiwanych plików audio w katalogu.")
    manifest = []
    events = []
    sensitivity = []
    spectra = {}
    decays = {}
    traces = {}
    for i, path in enumerate(files, 1):
        rid = f"R{i:02d}"
        rel = str(path.relative_to(source))
        folder = path.parent.name
        label = "background" if path.parent == source else "nok" if "nok" in path.name.lower() else "ok"
        scope = "chipped" if "Bitten" in folder else label
        plate = folder.replace(" ", "_") if folder.startswith("Bad ") else ""
        condition = path.stem.replace("NOK", "").replace("OK", "").strip()
        a = load_audio(path, Config(sr=SR, max_seconds=1800))
        x, nsr = native(path)
        raw = x[:, a.selected_channel]
        native_env = np.sqrt(
            np.maximum(uniform_filter1d(raw.astype(float) ** 2, max(2, round(0.01 * nsr))), 0)
        )[:: round(0.005 * nsr)]
        native_db = 20 * np.log10(np.maximum(native_env, 1e-12))
        sf.write(OUT / "decoded" / f"{rid}.wav", a.y, SR, subtype="FLOAT")
        ff = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
        )
        (OUT / f"{rid}_metadata.txt").write_text(ff.stderr)
        row = dict(
            recording_id=rid,
            path=rel,
            folder=folder,
            label=label,
            scope=scope,
            plate_id=plate,
            plate_id_status="folder-derived; confirm" if plate else "unknown",
            condition=condition,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            pcm_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            **a.qc,
            codec="AAC" if "aac (" in ff.stderr else "PCM" if "pcm_" in ff.stderr else "unknown",
            channel_max_difference=float(np.max(np.abs(x[:, 0] - x[:, 1]))) if x.shape[1] == 2 else np.nan,
            zero_fraction=float(np.mean(raw == 0)),
            below_minus100_fraction=float(np.mean(native_db < -100)),
            quiet_intervals=json.dumps(intervals(native_db < -100)),
            rms_p10_dbfs=float(np.quantile(native_db, 0.1)),
            rms_p90_dbfs=float(np.quantile(native_db, 0.9)),
            photo=str(next(path.parent.glob("*.jpg")).relative_to(source))
            if list(path.parent.glob("*.jpg"))
            else "",
        )
        onsets = detect_hits(a.y, SR, Config(min_gap_s=0.12))
        flux, _, _ = segment_ringdown(a.y, SR, min_gap_s=0.12, sensitivity=0.15)
        for gap in [0.12, 0.25, 0.4]:
            sensitivity.append(
                dict(
                    recording_id=rid,
                    min_gap_s=gap,
                    n_candidates=len(detect_hits(a.y, SR, Config(min_gap_s=gap))),
                )
            )
        row["n_candidates"] = len(onsets)
        row["n_flux_candidates"] = len(flux)
        row["median_gap_s"] = float(np.median(np.diff(onsets))) if len(onsets) > 1 else np.nan
        row["fraction_gap_under_350ms"] = (
            float(np.mean(np.diff(onsets) < 0.35)) if len(onsets) > 1 else np.nan
        )
        traces[rid] = (a.y, onsets, native_db)
        # Fixed 150 ms tail: rapid tapping prevents independent 350 ms windows.
        # Every candidate is retained in audit, even if rejected from feature exploration.
        clips = cut_hits(a.y, SR, onsets, Config(pre_s=0.02, tail_s=0.15, min_gap_s=0.12))
        powers = []
        tails = []
        for k, clip in enumerate(clips):
            t = clip["onset_s"]
            y = clip["y"]
            peak = float(np.max(np.abs(y)))
            before = a.y[max(0, round((t - 0.10) * SR)) : max(0, round((t - 0.02) * SR))]
            first = a.y[round(t * SR) : min(len(a.y), round((t + 0.025) * SR))]
            rise = float(
                20
                * np.log10(
                    max(np.sqrt(np.mean(first.astype(float) ** 2)), 1e-12)
                    / max(np.sqrt(np.mean(before.astype(float) ** 2)) if len(before) else 0, 1e-12)
                )
            )
            reasons = []
            if t < 0.75 or t > len(a.y) / SR - 0.4:
                reasons.append("edge/transient")
            if rise < 12:
                reasons.append("attack contrast <12 dB")
            if clip["truncated"]:
                reasons.append("next event/file before 150 ms")
            if peak < 1e-6:
                reasons.append("near silence")
            eid = f"{rid}_{k + 1:03d}"
            rec = dict(
                event_id=eid,
                recording_id=rid,
                plate_id=plate,
                label=label,
                scope=scope,
                onset_s=t,
                start_s=clip["start_s"],
                end_s=clip["end_s"],
                onset_offset_s=clip["onset_offset_s"],
                duration_s=len(y) / SR,
                attack_contrast_db=rise,
                peak_dbfs=20 * np.log10(max(peak, 1e-12)),
                flux_agreement=bool(len(flux) and np.min(abs(flux.onset_s.to_numpy() - t)) < 0.06),
                accepted=not reasons and label != "background",
                rejection="; ".join(reasons)
                if reasons
                else "background only"
                if label == "background"
                else "",
            )
            if rec["accepted"]:
                rec.update(extract_features(y, SR, clip["onset_offset_s"], fmax=20000))
                sf.write(OUT / "clips" / f"{eid}.wav", y, SR, subtype="PCM_24")
                norm = y / max(peak, 1e-12)
                ring = norm[round((clip["onset_offset_s"] + 0.01) * SR) :]
                f, p = welch(ring, fs=SR, nperseg=min(4096, len(ring)), nfft=4096)
                powers.append(p)
                tt, dd = decay_curve(norm, SR, clip["onset_offset_s"])
                tails.append(dd[:13])
            events.append(rec)
        row["n_feature_candidates"] = len(powers)
        manifest.append(row)
        if powers:
            spectra[rid] = np.median(powers, axis=0)
            decays[rid] = np.median(np.array(tails), axis=0)
        # Whole-file trace and spectrogram, no falsely precise ringdown threshold based on noisy intro.
        fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True, constrained_layout=True)
        axes[0].plot(np.arange(len(native_db)) * 0.005, native_db, lw=0.6, color="#366878")
        accepted = {e["onset_s"] for e in events if e["recording_id"] == rid and e["accepted"]}
        for t in onsets:
            axes[0].axvline(t, color="#137f83" if t in accepted else "#bda48c", lw=0.5, alpha=0.65)
        axes[0].set(title=f"{rid} · {rel}", ylabel="RMS [dBFS]", ylim=(-140, 0))
        hop = max(240, len(a.y) // 1300)
        S = np.abs(librosa.stft(a.y, n_fft=2048, hop_length=hop)) ** 2
        axes[1].imshow(
            librosa.power_to_db(S, ref=np.max, top_db=80),
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-80,
            vmax=0,
            extent=[0, len(a.y) / SR, 0, 24],
        )
        axes[1].set(ylabel="kHz", xlabel="Czas [s]", ylim=(0, 20))
        fig.savefig(OUT / "figures" / f"{rid}.png", dpi=120)
        plt.close(fig)
        print(rid, rel, "candidates", len(onsets), "features", len(powers), flush=True)
    records = pd.DataFrame(manifest)
    table = pd.DataFrame(events)
    good = table[table.accepted].copy()
    records.to_csv(OUT / "manifest.csv", index=False)
    table.to_csv(OUT / "candidate_audit.csv", index=False)
    good.to_csv(OUT / "features.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / "detector_sensitivity.csv", index=False)
    np.savez_compressed(OUT / "spectra.npz", frequency_hz=f, **spectra)
    np.savez_compressed(OUT / "decays.npz", **decays)
    # Per-recording, not per-plate: OK physical identity is unknown.
    fields = feature_columns(good, "physics")
    med = good.groupby("recording_id")[fields].median()
    summary = records.set_index("recording_id").join(med).reset_index()
    summary.to_csv(OUT / "recording_summary.csv", index=False)
    for family in ["physics", "mfcc", "mel"]:
        cols = feature_columns(good, family)
        grouped = good.groupby("recording_id")[cols].median()
        z = StandardScaler().fit_transform(grouped)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(z)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for rid, (u, v) in zip(grouped.index, coords):
            r = records.set_index("recording_id").loc[rid]
            ax.scatter(u, v, c=COLORS[r.scope], marker="^" if r.folder == "OLD" else "o", s=55)
            ax.annotate(rid, (u, v), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set(
            title=f"{family}: punkt = mediana nagrania (nie niezależna płytka)",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
        )
        fig.savefig(OUT / "figures" / f"pca_{family}.png", dpi=140)
        plt.close(fig)
        distances = []
        from scipy.spatial.distance import cdist

        D = cdist(z, z) / np.sqrt(len(cols))
        for j, rid in enumerate(grouped.index):
            for k, rjd in enumerate(grouped.index):
                if k <= j:
                    continue
                r1 = records.set_index("recording_id").loc[rid]
                r2 = records.set_index("recording_id").loc[rjd]
                category = "other"
                if r1.folder == r2.folder and r1.folder.startswith("Bad "):
                    category = "same plate, different recording/condition"
                elif {r1.folder, r2.folder} == {"Bad 1", "Bad 2"}:
                    category = "two cracked plates"
                elif {r1.scope, r2.scope} == {"ok", "nok"} and ("Bad " in r1.folder or "Bad " in r2.folder):
                    category = "OLD OK vs new crack (confounded)"
                distances.append(dict(family=family, a=rid, b=rjd, category=category, distance=D[j, k]))
        pd.DataFrame(distances).to_csv(OUT / f"distances_{family}.csv", index=False)
    # All noise-sensitive changes are compared at feature level, not fabricated model accuracy.
    (OUT / "analysis_settings.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                sr=SR,
                min_gap_s=0.12,
                tail_s=0.15,
                pre_s=0.02,
                min_attack_contrast_db=12,
                exclude_first_s=0.75,
                exclude_last_s=0.4,
                labels="filename/folder; physical OK IDs unconfirmed",
                performance_estimated=False,
            ),
            indent=2,
        )
    )
    return records, good


if __name__ == "__main__":
    run(Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/raw/client-20260929/Recordings")))
