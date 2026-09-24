import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from plotly.offline import get_plotlyjs
from scipy.signal import welch

from pinglab.audio import cut_hits, detect_hits, load_audio
from pinglab.features import decay_curve, extract_features
from pinglab.learning import evaluate
from pinglab.plots import embedding_map
from pinglab.robustness import sensitivity_table

root = Path(__file__).resolve().parents[1]
assets = root / "assets"
assets.mkdir(exist_ok=True)


def arrays(value):
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), 5).tolist()
    if isinstance(value, dict):
        return {k: arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        return [arrays(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


samples = []
for name, filename in [("A", "SYN_001_ok.wav"), ("B", "SYN_021_nok.wav")]:
    audio = load_audio(root / "data/demo" / filename)
    times = detect_hits(audio.y, audio.sr)
    clip = cut_hits(audio.y, audio.sr, times)[0]
    y = clip["y"]
    sr = audio.sr
    sf.write(
        assets / f"demo-{name.lower()}.wav", y / max(np.max(np.abs(y)), 1e-8) * 0.85, sr, subtype="PCM_16"
    )
    tail = y[int((clip["onset_offset_s"] + 0.01) * sr) :]
    f, p = welch(tail, fs=sr, nperseg=4096)
    keep = (f >= 80) & (f <= 20000)
    dt, db = decay_curve(y, sr, clip["onset_offset_s"])
    s = np.abs(librosa.stft(y, n_fft=1024, hop_length=480))
    z = librosa.amplitude_to_db(s, ref=np.max, top_db=70)
    samples.append(
        dict(
            name=name,
            t=np.arange(0, len(y), 24) / sr,
            y=y[::24] / np.max(np.abs(y)),
            f=f[keep],
            spectrum=10 * np.log10(np.maximum(p[keep], 1e-20) / p[keep].max()),
            decay_t=dt,
            decay_db=db,
            spec_t=librosa.frames_to_time(np.arange(z.shape[1]), sr=sr, hop_length=480),
            spec_f=librosa.fft_frequencies(sr=sr, n_fft=1024),
            spec_z=z,
            features=extract_features(y, sr, clip["onset_offset_s"]),
        )
    )
    if name == "A":
        stats, variants = sensitivity_table(y, sr, offset=clip["onset_offset_s"])
        noise_data = []
        for i, key in enumerate(["original", "noise 20 dB", "noise 10 dB", "noise 0 dB"]):
            signal = variants[key]
            nt, ndb = decay_curve(signal, sr, clip["onset_offset_s"])
            row = stats[stats.variant == key].iloc[0]
            sf.write(
                assets / f"noise-{i}.wav",
                signal / max(np.max(np.abs(signal)), 1e-8) * 0.85,
                sr,
                subtype="PCM_16",
            )
            noise_data.append(
                dict(label=key, t=nt, db=ndb, centroid=float(row.centroid_hz), decay=float(row.decay10_s))
            )
        # min/max waveform envelope for whole-file view
        blocks = audio.y[: len(audio.y) // 240 * 240].reshape(-1, 240)
        long = dict(
            t=np.arange(len(blocks)) * 240 / sr, lo=blocks.min(axis=1), hi=blocks.max(axis=1), onsets=times
        )
f = pd.read_csv(root / "reports/runs/demo/features.csv")
pca = {}
for family in ["resonance", "physics", "mfcc", "mel"]:
    _, table = embedding_map(f, family)
    pca[family] = table[["plate_id", "label", "session_id", "PC1", "PC2"]].to_dict("records")

metrics, predictions, audit = evaluate(f, bootstrap=300)
data = dict(
    samples=samples,
    long=long,
    noise=noise_data,
    pca=pca,
    metrics=metrics,
    predictions=predictions.to_dict("records"),
)
(assets / "demo-data.js").write_text(
    "window.PING_DEMO = " + json.dumps(arrays(data), ensure_ascii=False) + ";\n"
)
(assets / "plotly.min.js").write_text(get_plotlyjs())
print("Guide assets ready")
