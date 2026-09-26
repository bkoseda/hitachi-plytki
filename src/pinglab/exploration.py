"""Portable notebook views: static PNG figures and embedded local audio."""

import html
import json

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Audio, Markdown, display

from .audio import detect_hits, load_audio


def audio_gallery(session):
    for row in session.records.itertuples():
        recording = load_audio(
            row.resolved_path, session.config, channel=int(row.channel) if row.channel else None
        )
        player = Audio(recording.y, rate=recording.sr, normalize=False)._repr_html_()
        title = html.escape(f"{row.plate_id} · {row.label} · {row.path}")
        display(
            HTML(f"<details><summary>{title} — odsłuch oryginalnej głośności</summary>{player}</details>")
        )


def spectrogram_gallery(session, per_page=4):
    annotations = json.loads(session.annotations.read_text()) if session.annotations.exists() else {}
    records = session.records[session.records.label != "background"].to_dict("records")
    for start in range(0, len(records), per_page):
        page = records[start : start + per_page]
        fig, axes = plt.subplots(
            len(page), 2, figsize=(12, 2.8 * len(page)), squeeze=False, constrained_layout=True
        )
        for ax, row in zip(axes, page):
            a = load_audio(
                row["resolved_path"], session.config, channel=int(row["channel"]) if row["channel"] else None
            )
            onsets = annotations.get(row["path"])
            if onsets is None:
                onsets = detect_hits(a.y, a.sr, session.config)
            # Min/max bin envelopes preserve short impacts even for long recordings.
            stride = max(1, len(a.y) // 2500)
            n = len(a.y) // stride
            bins = a.y[: n * stride].reshape(n, stride)
            t = np.arange(n) * stride / a.sr
            ax[0].fill_between(t, bins.min(axis=1), bins.max(axis=1), color="#287f8e")
            for onset in onsets:
                ax[0].axvline(onset, color="#d57936", alpha=0.65, linewidth=0.8)
            ax[0].set(
                title=f"{row['plate_id']} · {row['label']} · {len(onsets)} kandydatów",
                xlabel="Czas [s]",
                ylabel="Amplituda",
            )
            hop = max(session.config.hop, len(a.y) // 1800)
            power = np.abs(librosa.stft(a.y, n_fft=2048, hop_length=hop)) ** 2
            db = (
                librosa.power_to_db(power, ref=np.max, top_db=80)
                if power.max() > 1e-12
                else np.full_like(power, -80)
            )
            frequencies = librosa.fft_frequencies(sr=a.sr, n_fft=2048)
            keep = frequencies <= session.fmax
            ax[1].imshow(
                db[keep],
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=-80,
                vmax=0,
                extent=[0, len(a.y) / a.sr, 0, frequencies[keep][-1] / 1000],
            )
            ax[1].set(
                title="Spektrogram · −80…0 dB względem maksimum pliku",
                xlabel="Czas [s]",
                ylabel="Częstotliwość [kHz]",
            )
        display(fig)
        plt.close(fig)


def feature_atlas(events):
    if events.empty:
        display(Markdown("**Brak uderzeń:** popraw czasy w notebooku 02."))
        return
    features = ["peak_hz", "centroid_hz", "decay10_s"]
    units = ["Dominujący rezonans [Hz]", "Środek widma [Hz]", "Zanik o 10 dB [s]"]
    grouped = events.groupby(["plate_id", "label"], sort=True)[features].agg(["mean", "std"]).fillna(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, max(4, len(grouped) * 0.2)), constrained_layout=True)
    colors = {"ok": "#287f8e", "nok": "#c86635", "unknown": "#667085"}
    for ax, feature, title in zip(axes, features, units):
        for i, ((pid, label), row) in enumerate(grouped.iterrows()):
            ax.errorbar(
                row[feature]["mean"],
                i,
                xerr=row[feature]["std"],
                fmt="o",
                color=colors.get(label, "gray"),
                capsize=2,
                markersize=4,
            )
        ax.set(yticks=range(len(grouped)), yticklabels=[p for p, label in grouped.index], xlabel=title)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        "Każdy wiersz = fizyczna płyta. Punkt = średnia, kreska = zmienność uderzeń (nie przedział ufności)."
    )
    display(fig)
    plt.close(fig)


def metadata_audit(session):
    records = session.records[session.records.label != "background"]
    for col in ["session_id", "plate_type", "temperature"]:
        distinct = records[["plate_id", "label", col]].drop_duplicates()
        display(Markdown(f"**{col}** — liczba płyt; puste oznacza brak informacji"))
        display(pd.crosstab(distinct[col].replace("", "brak"), distinct.label))
    return records[["plate_id", "plate_type", "temperature", "photo", "notes"]].drop_duplicates()
