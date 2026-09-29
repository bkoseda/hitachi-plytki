"""Reproducible two-batch pilot study. Group identities remain explicitly provisional."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from .audio import Config, cut_hits, detect_hits
from .features import extract_features, feature_columns

VERSION = "client-v2-170ms-01"
SR = 48000
OUT_REL = "reports/runs/client-combined-20260929"


def find_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "config/client_archive_manifest.json").is_file():
            return p
    raise FileNotFoundError("Otwórz folder ceramic-ping-lab w VS Code.")


def decode(path):
    """Native decoded float samples, without DC removal or amplitude normalization."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        r = subprocess.run(
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
        return sf.read(io.BytesIO(r.stdout), dtype="float32", always_2d=True)


def true_runs(mask, sr):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (float(a / sr), float(b / sr)) for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))
    ]


def inventory(root):
    root = Path(root)
    rows = []
    for batch, prefix in [("1-batch-ping", "R"), ("2-batch-ok", "G")]:
        paths = sorted(
            p for p in (root / "assets" / batch).rglob("*") if p.suffix.lower() in {".wav", ".m4a"}
        )
        for i, p in enumerate(paths, 1):
            rel = p.relative_to(root / "assets" / batch)
            folder = p.parent.name
            name = p.stem.lower()
            group, scope, label = "", "old_unknown", int(name.startswith("nok"))
            if batch == "2-batch-ok":
                group = {"Part 2": "OK_2", "Part 3": "OK_3"}.get(folder, "OK_1")
                scope, label = "core", 0
            elif folder in {"Bad 1", "Bad 2"}:
                group, scope, label = folder.replace(" ", "_"), "core", 1
            elif "Bitten" in folder:
                group, scope, label = folder.replace(" ", "_"), "chipped", 1
            elif "NOISE" in p.name:
                scope, label = "background", -1
            support = (
                "4-point"
                if "4 point" in name
                else "1-point"
                if ("1 point" in name or "one point" in name)
                else "unknown"
            )
            rows.append(
                dict(
                    recording_id=f"{prefix}{i:02d}",
                    path=str(p.relative_to(root)),
                    filename=str(rel),
                    batch=batch,
                    group=group,
                    scope=scope,
                    label=label,
                    group_status="folder hypothesis, not confirmed",
                    support=support,
                    angle=("angle" in name or "agnle" in name),
                    tool_change="tool change" in name,
                    codec="PCM" if p.suffix.lower() == ".wav" else "AAC",
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    table = pd.DataFrame(rows)
    overrides = root / "config/client_group_overrides.csv"
    if overrides.exists():
        ov = pd.read_csv(overrides).set_index("recording_id")
        for col in ["group", "group_status"]:
            if col in ov:
                table[col] = table.recording_id.map(ov[col]).fillna(table[col])
    return table


def prepare(root=None, force=False):
    root = find_root(root)
    out = root / OUT_REL
    for name in ["cache", "decoded", "clips", "figures"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    inventory_frame = inventory(root)
    all_qc, all_events, all_features = [], [], []
    for row in inventory_frame.to_dict("records"):
        rid = row["recording_id"]
        key = hashlib.sha256((row["sha256"] + VERSION).encode()).hexdigest()
        cache = out / "cache" / key
        if not force and cache.with_suffix(".json").exists():
            qc = json.loads(cache.with_suffix(".json").read_text())
            ev = pd.read_pickle(cache.with_suffix(".events"))
            ft = pd.read_pickle(cache.with_suffix(".features"))
        else:
            a, sr = decode(root / row["path"])
            if sr != SR:
                raise ValueError(f"{rid}: porównanie wymaga 48 kHz; źródło ma {sr}.")
            channel = int(np.argmax(np.mean(a.astype(float) ** 2, axis=0)))
            x = a[:, channel].astype(float)
            n = sr // 100
            rms = np.sqrt(np.mean(x[: len(x) // n * n].reshape(-1, n) ** 2, axis=1))
            db = 20 * np.log10(np.maximum(rms, 1e-12))
            zero_runs = true_runs(x == 0, sr)
            qc = dict(
                duration_s=len(x) / sr,
                sample_rate=sr,
                channels=a.shape[1],
                selected_channel=channel,
                dual_mono=bool(a.shape[1] == 2 and np.array_equal(a[:, 0], a[:, 1])),
                peak_dbfs=float(20 * np.log10(max(np.max(np.abs(x)), 1e-12))),
                clipping_fraction=float(np.mean(np.abs(x) >= 0.999)),
                zero_fraction=float(np.mean(x == 0)),
                longest_zero_s=max((b - a for a, b in zero_runs), default=0),
                below_minus100_fraction=float(np.mean(db < -100)),
                rms_p10_dbfs=float(np.quantile(db, 0.1)),
                rms_p90_dbfs=float(np.quantile(db, 0.9)),
                pcm_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            )
            y = (x - x.mean()).astype(np.float32)
            sf.write(out / "decoded" / f"{rid}.wav", y, sr, subtype="FLOAT")
            onsets = detect_hits(y, sr, Config(min_gap_s=0.12))
            qc.update(
                n_candidates=len(onsets),
                median_gap_s=float(np.median(np.diff(onsets))) if len(onsets) > 1 else None,
                gap_under_350ms=float(np.mean(np.diff(onsets) < 0.35)) if len(onsets) > 1 else None,
                n_gap250=len(detect_hits(y, sr, Config(min_gap_s=0.25))),
                n_gap400=len(detect_hits(y, sr, Config(min_gap_s=0.4))),
            )
            clips = cut_hits(y, sr, onsets, Config(pre_s=0.02, tail_s=0.15, min_gap_s=0.12))
            events, features = [], []
            for k, c in enumerate(clips):
                t, z = c["onset_s"], c["y"]
                pre = y[max(0, round((t - 0.10) * sr)) : max(0, round((t - 0.02) * sr))]
                attack = y[round(t * sr) : round((t + 0.025) * sr)]
                pre_rms = float(np.sqrt(np.mean(pre.astype(float) ** 2))) if len(pre) else 0.0
                attack_rms = float(np.sqrt(np.mean(attack.astype(float) ** 2))) if len(attack) else 0.0
                rise = 20 * np.log10(max(attack_rms, 1e-12) / max(pre_rms, 1e-12))
                reasons = []
                if qc["duration_s"] < 2:
                    reasons.append("file shorter than 2 s")
                if t < 0.75 or t > len(y) / sr - 0.4:
                    reasons.append("file edge")
                if rise < 12:
                    reasons.append("attack contrast <12 dB")
                if c["truncated"]:
                    reasons.append("next hit or edge before 150 ms")
                if np.max(np.abs(z)) < 1e-6:
                    reasons.append("near silence")
                if np.any(np.abs(z) >= 0.999):
                    reasons.append("clipping")
                if row["scope"] == "background":
                    reasons.append("background")
                eid = f"{rid}_{k + 1:03d}"
                e = dict(
                    event_id=eid,
                    recording_id=rid,
                    onset_s=t,
                    onset_offset_s=c["onset_offset_s"],
                    start_s=c["start_s"],
                    end_s=c["end_s"],
                    attack_contrast_db=float(rise),
                    pre_rms_dbfs=20 * np.log10(max(pre_rms, 1e-12)),
                    peak_dbfs=20 * np.log10(max(float(np.max(np.abs(z))), 1e-12)),
                    accepted=not reasons,
                    rejection="; ".join(reasons),
                )
                events.append(e)
                if not reasons:
                    f = extract_features(z, sr, c["onset_offset_s"])
                    features.append({**e, **f})
                    sf.write(out / "clips" / f"{eid}.wav", z, sr, subtype="FLOAT")
            ev = pd.DataFrame(
                events,
                columns=list(events[0])
                if events
                else ["event_id", "recording_id", "onset_s", "accepted", "rejection"],
            )
            ft = pd.DataFrame(features)
            qc["n_accepted"] = len(ft)
            cache.with_suffix(".json").write_text(json.dumps(qc, indent=2))
            ev.to_pickle(cache.with_suffix(".events"))
            ft.to_pickle(cache.with_suffix(".features"))
        all_qc.append({**row, **qc})
        all_events.append(ev)
        if len(ft):
            all_features.append(ft)
        print(f"{rid}: {qc['n_accepted']}/{qc['n_candidates']} kandydatów · {row['filename']}", flush=True)
    records = pd.DataFrame(all_qc)
    events = pd.concat([x for x in all_events if len(x)], ignore_index=True)
    features = pd.concat(all_features, ignore_index=True)
    meta = ["recording_id", "group", "scope", "label", "support", "angle", "tool_change", "codec", "batch"]
    features = features.merge(records[meta], on="recording_id", validate="many_to_one")
    records.to_csv(out / "manifest.csv", index=False)
    events.to_csv(out / "candidate_audit.csv", index=False)
    features.to_csv(out / "features.csv", index=False)
    columns = (
        feature_columns(features, "physics")
        + feature_columns(features, "mfcc")
        + feature_columns(features, "mel")
    )
    summary = records.merge(
        features.groupby("recording_id")[columns].median().reset_index(),
        on="recording_id",
        how="left",
        validate="one_to_one",
    )
    summary.to_csv(out / "recording_features.csv", index=False)
    (out / "settings.json").write_text(
        json.dumps(
            dict(
                version=VERSION,
                sr=SR,
                pre_s=0.02,
                tail_s=0.15,
                min_gap_s=0.12,
                contrast_db=12,
                representative_taps=10,
                group_ids="unconfirmed folder hypothesis",
            ),
            indent=2,
        )
    )
    return records, events, features


def representative_events(features, per_file=10):
    return pd.concat(
        [
            g.sort_values("onset_s").iloc[
                np.unique(np.linspace(0, len(g) - 1, min(per_file, len(g))).round().astype(int))
            ]
            for _, g in features.groupby("recording_id", sort=True)
        ],
        ignore_index=True,
    )


def embeddings(root=None, force=False):
    from .embeddings import ASTEncoder

    root = find_root(root)
    out = root / OUT_REL
    ft = pd.read_csv(out / "features.csv")
    selected = representative_events(ft)
    cache = root / "data/cache/client-combined-ast"
    cache.mkdir(parents=True, exist_ok=True)
    encoder = ASTEncoder(root / "models/ast", offline=True)
    vectors = []
    for i, r in enumerate(selected.itertuples()):
        y, sr = sf.read(out / "clips" / f"{r.event_id}.wav", dtype="float32")
        key = hashlib.sha256(
            y.tobytes() + str(sr).encode() + encoder.revision.encode() + b"peak095-v1"
        ).hexdigest()
        path = cache / f"{key}.npy"
        if path.exists() and not force:
            v = np.load(path, allow_pickle=False)
        else:
            v = encoder.encode(y, sr)
            np.save(path, v, allow_pickle=False)
        assert v.shape == (768,) and np.isfinite(v).all()
        vectors.append(v)
        if i % 25 == 0:
            print(f"AST: {i + 1}/{len(selected)}", flush=True)
    result = pd.concat(
        [selected, pd.DataFrame(vectors, columns=[f"ast_{i:03d}" for i in range(768)])], axis=1
    )
    result.to_pickle(out / "representations.pkl")
    selected[["event_id", "recording_id", "onset_s"]].to_csv(out / "representative_events.csv", index=False)
    (out / "ast.json").write_text(
        json.dumps(
            dict(
                revision=encoder.revision,
                events=len(result),
                sample_rate=16000,
                dimensions=768,
                frozen=True,
                normalized=True,
                per_recording=10,
                comment="170 ms clips padded by standard AST processor; no content above 8 kHz",
            ),
            indent=2,
        )
    )
    return result


def load(root=None):
    out = find_root(root) / OUT_REL
    return (
        pd.read_csv(out / "manifest.csv"),
        pd.read_csv(out / "candidate_audit.csv"),
        pd.read_csv(out / "features.csv"),
    )


def inspect_recording(recording_id="R02", start_s=0, duration_s=5, root=None):
    """Notebook widget callback; fixed waveform scale, audio not normalized."""
    import matplotlib.pyplot as plt
    from IPython.display import Audio, display

    root = find_root(root)
    records, events, _ = load(root)
    row = records.set_index("recording_id").loc[recording_id]
    x, sr = decode(root / row.path)
    y = x[:, int(row.selected_channel)]
    a = min(max(0, int(start_s * sr)), max(0, len(y) - 1))
    b = min(len(y), a + int(duration_s * sr))
    z = y[a:b]
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
    axs[0].plot(np.arange(a, b) / sr, z, lw=0.6, color="#177d83")
    for e in events[
        (events.recording_id == recording_id) & events.onset_s.between(a / sr, b / sr)
    ].itertuples():
        color = "#177d83" if e.accepted else "#d59b48"
        axs[0].axvline(e.onset_s, color=color, lw=0.8)
        if e.accepted:
            axs[0].axvspan(e.onset_s - 0.02, e.onset_s + 0.15, color=color, alpha=0.13)
    axs[0].set(
        title=f"{recording_id} · {row.filename} — zielone: okna cech; złote: odrzucone wykrycia",
        ylabel="Amplituda",
    )
    nfft = min(2048, 2 ** int(np.floor(np.log2(max(16, len(z))))))
    S = np.abs(librosa.stft(z, n_fft=nfft, hop_length=240)) ** 2
    axs[1].imshow(
        librosa.power_to_db(S, ref=np.max, top_db=80),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[a / sr, b / sr, 0, sr / 2000],
        vmin=-80,
        vmax=0,
    )
    axs[1].set(ylim=(0, 20), ylabel="Częstotliwość [kHz]", xlabel="Czas nagrania [s]")
    plt.show()
    display(Audio(z, rate=sr, normalize=False))
    display(
        events[(events.recording_id == recording_id) & events.onset_s.between(a / sr, b / sr)][
            ["event_id", "onset_s", "accepted", "rejection"]
        ].reset_index(drop=True)
    )


def recording_widget(root=None):
    import ipywidgets as w
    from IPython.display import display
    from IPython.utils.capture import capture_output

    root = find_root(root)
    records, _, _ = load(root)
    options = [(f"{r.recording_id} · {r.filename}", r.recording_id) for r in records.itertuples()]
    pick = w.Dropdown(options=options, value="R02", description="Nagranie", layout=w.Layout(width="90%"))
    start = w.FloatSlider(
        min=0, max=60, step=0.1, value=0, description="Początek [s]", continuous_update=False
    )
    length = w.FloatSlider(
        min=0.2, max=15, step=0.1, value=5, description="Długość [s]", continuous_update=False
    )
    output = w.Output()

    def render(change=None):
        start.max = max(0, float(records.set_index("recording_id").loc[pick.value, "duration_s"]) - 0.2)
        # Assign rendered MIME bundles directly. Output's IOPub interception can
        # deadlock nbclient with current ipykernel, though it works interactively.
        with capture_output() as captured:
            inspect_recording(pick.value, start.value, length.value, root)
        output.outputs = tuple(
            {"output_type": "display_data", "data": item.data, "metadata": item.metadata}
            for item in captured.outputs
        )

    for widget in [pick, start, length]:
        widget.observe(render, names="value")
    display(w.VBox([pick, start, length, output]))
    render()
    return pick, start, length, output
