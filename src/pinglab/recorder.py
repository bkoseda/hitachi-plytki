"""Local microphone recording and exploratory impact/ring-down boundaries."""

from pathlib import Path
from uuid import uuid4

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d


def input_devices():
    import sounddevice as sd

    return pd.DataFrame(
        [
            dict(
                device=i,
                name=d["name"],
                channels=d["max_input_channels"],
                default_sr=int(d["default_samplerate"]),
            )
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        ]
    )


def record_wav(folder, seconds=20, device=None, sr=None, countdown=3):
    """Explicit call opens the microphone. Preserve native input rate; never overwrite."""
    import time
    from datetime import datetime

    import sounddevice as sd

    if not 1 <= seconds <= 180:
        raise ValueError("Wybierz nagranie od 1 do 180 sekund.")
    info = sd.query_devices(device, "input")
    sr = int(sr or info["default_samplerate"])
    sd.check_input_settings(device=device, channels=1, samplerate=sr, dtype="float32")
    for remaining in range(countdown, 0, -1):
        print(f"Start za {remaining}…", flush=True)
        time.sleep(1)
    print(f"NAGRYWANIE · {seconds:g} s · najpierw 0.5–1 s ciszy", flush=True)
    try:
        y = sd.rec(round(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
        status = sd.wait()
    except BaseException:
        sd.stop()
        raise
    if status:
        raise RuntimeError(f"Przerwa w strumieniu audio ({status}). Powtórz nagranie.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ping_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}.wav"
    sf.write(path, y, sr, subtype="PCM_24")
    print("ZAPISANO:", path.name, flush=True)
    return path


def demo_sequence(sr=48000):
    """Known complete and overlapping decays; exercises only, not racket measurements."""
    from .demo import resonator

    rng = np.random.default_rng(78)
    y = rng.normal(0, 0.0001, 8 * sr)
    times = [1.0, 3.3, 3.65, 6.0]
    for i, t in enumerate(times):
        hit = resonator(sr, 1.8, f0=1700, tau=0.28, seed=i) * 0.5
        start = round(t * sr)
        y[start : start + len(hit)] += hit
    return y.astype("float32"), sr, np.array(times)


def segment_ringdown(
    y,
    sr,
    onsets=None,
    min_gap_s=0.12,
    sensitivity=0.10,
    noise_seconds=0.5,
    drop_db=30,
    quiet_hold_s=0.08,
    attack_s=0.03,
):
    """Heuristic observed decay, not source separation or physical damping measurement.

    Quiet is a sustained RMS level <= max(noise+6 dB, hit peak-drop_db).
    Repeated excitation cannot be unmixed from the previous ring-down with this rule.
    """
    y = np.asarray(y, float)
    if y.ndim != 1 or len(y) < sr * 0.2 or not np.isfinite(y).all():
        raise ValueError("Potrzebne co najmniej 0.2 s poprawnego sygnału mono.")
    if min_gap_s <= 0 or not 0 < sensitivity < 1 or drop_db <= 0 or quiet_hold_s <= 0 or attack_s <= 0:
        raise ValueError("Nieprawidłowe parametry detekcji.")
    if not 0 < noise_seconds < len(y) / sr:
        raise ValueError("Odcinek ciszy musi mieścić się w nagraniu.")
    hop = max(1, round(sr * 0.005))
    rms = np.sqrt(np.maximum(0, uniform_filter1d(y * y, max(2, round(sr * 0.01)))))[::hop]
    t = np.arange(len(rms)) * hop / sr
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    floor = float(np.quantile(db[t < noise_seconds], 0.9))
    if onsets is None:
        if np.max(np.abs(y)) < 1e-7:
            onsets = np.array([])
        else:
            novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=1024)
            onsets = librosa.onset.onset_detect(
                onset_envelope=novelty,
                sr=sr,
                hop_length=hop,
                units="time",
                backtrack=False,
                pre_max=3,
                post_max=3,
                pre_avg=20,
                post_avg=20,
                wait=max(1, round(min_gap_s * sr / hop)),
                delta=sensitivity,
            )
            # Onset strength peaks lag short attacks by approximately a frame.
            onsets = np.maximum(0, np.asarray(onsets) - 0.01)
    onsets = np.asarray(onsets, float)
    if (
        onsets.ndim != 1
        or not np.isfinite(onsets).all()
        or np.any(onsets < 0)
        or np.any(onsets >= len(y) / sr)
        or np.any(np.diff(onsets) <= 0)
    ):
        raise ValueError("Czasy muszą być rosnące, unikalne i wewnątrz nagrania.")
    rows = []
    hold = max(1, round(quiet_hold_s * sr / hop))
    duration = len(y) / sr
    for i, onset in enumerate(onsets):
        next_onset = onsets[i + 1] if i + 1 < len(onsets) else duration
        a = int(np.searchsorted(t, onset))
        b = int(np.searchsorted(t, next_onset))
        peak_end = min(b, max(a + 1, int(np.searchsorted(t, onset + attack_s))))
        peak = float(db[a:peak_end].max())
        threshold = max(floor + 6, peak - drop_db)
        begin = min(b, int(np.searchsorted(t, onset + attack_s)))
        # Keep 10 ms away from the next onset (RMS window straddles boundaries).
        stop = max(begin, b - 2)
        quiet = db[begin:stop] <= threshold
        end = None
        if len(quiet) >= hold:
            streak = np.convolve(quiet.astype(int), np.ones(hold, int), "valid")
            indices = np.flatnonzero(streak == hold)
            if len(indices):
                end = float(t[begin + indices[0]])
        status = (
            "zanik zaobserwowany"
            if end is not None
            else (
                "kolejne uderzenie przed zanikiem"
                if i + 1 < len(onsets)
                else "koniec nagrania przed zanikiem"
            )
        )
        end = next_onset if end is None else end
        rows.append(
            dict(
                ping=i + 1,
                onset_s=float(onset),
                attack_end_s=float(min(onset + attack_s, next_onset)),
                end_s=float(end),
                duration_s=float(end - onset),
                status=status,
                threshold_dbfs=threshold,
                noise_dbfs=floor,
                onset_in_noise_window=bool(onset < noise_seconds),
            )
        )
    columns = [
        "ping",
        "onset_s",
        "attack_end_s",
        "end_s",
        "duration_s",
        "status",
        "threshold_dbfs",
        "noise_dbfs",
        "onset_in_noise_window",
    ]
    return pd.DataFrame(rows, columns=columns), t, db


def ringdown_plot(y, sr, table, t, db):
    from .plots import theme

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Sygnał: czerwony = atak, zielony = zakończony ogon, pomarańczowy = ogon ucięty",
            "Obwiednia RMS i próg końca wybrzmienia",
            "Spektrogram",
        ),
    )
    step = max(1, len(y) // 4000)
    n = len(y) // step
    blocks = np.asarray(y[: n * step]).reshape(n, step)
    tx = np.arange(n) * step / sr
    for v in [blocks.min(axis=1), blocks.max(axis=1)]:
        fig.add_trace(
            go.Scatter(x=tx, y=v, mode="lines", line=dict(color="#327c8a", width=0.7), showlegend=False),
            row=1,
            col=1,
        )
    fig.add_trace(go.Scatter(x=t, y=db, name="RMS dBFS", line=dict(color="#327c8a")), row=2, col=1)
    hop = max(round(sr * 0.005), len(y) // 1500)
    power = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    spec = (
        librosa.power_to_db(power, ref=np.max, top_db=80) if power.max() > 1e-12 else np.full_like(power, -80)
    )
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = f <= min(20000, sr / 2)
    fig.add_trace(
        go.Heatmap(
            x=librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr, hop_length=hop),
            y=f[mask],
            z=spec[mask],
            colorscale="Magma",
            zmin=-80,
            zmax=0,
            colorbar=dict(title="dB rel.", len=0.25, y=0.13),
            showscale=True,
        ),
        row=3,
        col=1,
    )
    for row in table.itertuples():
        color = "#39a581" if row.status == "zanik zaobserwowany" else "#f2a844"
        fig.add_vrect(
            x0=row.onset_s, x1=row.attack_end_s, fillcolor="#d75851", opacity=0.30, line_width=0, row=1, col=1
        )
        fig.add_vrect(
            x0=row.attack_end_s, x1=row.end_s, fillcolor=color, opacity=0.22, line_width=0, row=1, col=1
        )
        fig.add_vline(x=row.onset_s, line_dash="dot", line_color="#d75851", row="all", col=1)
        fig.add_trace(
            go.Scatter(
                x=[row.onset_s, row.end_s],
                y=[row.threshold_dbfs] * 2,
                line=dict(dash="dash", color=color),
                name=f"Próg pingu {row.ping}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=row.onset_s, y=1, yref="y domain", text=str(row.ping), showarrow=False, row=1, col=1
        )
    fig.update_yaxes(title_text="Amplituda", row=1, col=1)
    fig.update_yaxes(title_text="dBFS", row=2, col=1)
    fig.update_yaxes(title_text="Hz", row=3, col=1)
    fig.update_xaxes(title_text="Czas [s]", row=3, col=1)
    return theme(fig, "Uderzenia i obserwowane wybrzmienia — przybliż fragment myszą", 850)


def export_pings(y, sr, table, folder):
    """Each file stops at observed quiet or next hit; never claim separation of mixed tails."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in table.itertuples():
        path = folder / f"ping_{row.ping:02}.wav"
        if path.exists():
            raise FileExistsError(f"Katalog eksportu już zawiera {path.name}; wybierz nowy.")
        start = max(0, round((row.onset_s - 0.015) * sr))
        sf.write(path, y[start : round(row.end_s * sr)], sr, subtype="PCM_24")
        paths.append(path)
    table.to_csv(folder / "segments.csv", index=False)
    return paths
