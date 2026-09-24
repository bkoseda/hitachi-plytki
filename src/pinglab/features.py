"""Compact, interpretable features. Loudness is retained separately as QC."""
import numpy as np
import librosa
from scipy.signal import welch, find_peaks

EPS = 1e-12


def decay_curve(y, sr, onset_offset_s=0.025):
    start = max(0, int((onset_offset_s + 0.01) * sr))
    tail = np.asarray(y[start:], dtype=float)
    hop = max(1, int(sr * 0.01))
    count = len(tail)//hop
    if count < 3:
        return np.array([]), np.array([])
    energy = np.mean(tail[:count*hop].reshape(count, hop)**2, axis=1)
    db = 10 * np.log10(np.maximum(energy, EPS)/max(float(energy.max()), EPS))
    return np.arange(count) * hop/sr, db


def extract_features(y, sr, onset_offset_s=0.025, fmax=20000):
    """Same fmax and sample rate must be used for all compared files."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < max(32, int(0.05 * sr)) or not np.isfinite(y).all():
        raise ValueError('Nieprawidłowe okno audio.')
    if np.max(np.abs(y)) < 1e-7:
        raise ValueError('Cisza nie jest uderzeniem; brak cech.')
    if not 500 <= fmax <= sr / 2:
        raise ValueError('fmax musi mieścić się w [500, sr/2].')
    y = y - y.mean()
    y = y / max(np.max(np.abs(y)), EPS)  # amplitude-invariant copy
    ring = y[min(int((onset_offset_s + .01)*sr), len(y)//2):]
    f, power = welch(ring, fs=sr, nperseg=min(4096, len(ring)))
    mask = (f >= 80) & (f <= fmax)
    f, power = f[mask], power[mask]
    q = power / max(float(power.sum()), EPS)
    centroid = float(np.sum(f*q))
    bandwidth = float(np.sqrt(np.sum((f-centroid)**2*q)))
    peak_ids, _ = find_peaks(power, prominence=max(power.max()*.04, EPS))
    peak_ids = sorted(peak_ids, key=lambda p: power[p], reverse=True)
    peak_hz = float(f[peak_ids[0]]) if peak_ids else float(f[np.argmax(power)])
    t, db = decay_curve(y, sr, onset_offset_s)
    below = np.flatnonzero(db <= -10)
    # This is a window-relative decay descriptor, not RT60 or a calibrated material constant.
    decay10 = float(t[below[0]]) if len(below) else float(t[-1]) if len(t) else 0.0
    n = max(1, len(ring)//4)
    late_ratio = float(np.mean(ring[-n:]**2) / max(np.mean(ring[:n]**2), EPS))
    basic = dict(peak_hz=peak_hz, centroid_hz=centroid, bandwidth_hz=bandwidth,
                 rolloff85_hz=float(f[min(np.searchsorted(np.cumsum(q), .85),len(f)-1)]),
                 flatness=float(np.exp(np.mean(np.log(power+EPS)))/(np.mean(power)+EPS)),
                 entropy=float(-np.sum(q*np.log(q+EPS))/np.log(max(len(q),2))),
                 decay10_s=decay10, late_early_ratio=late_ratio,
                 peak_count=float(len(peak_ids)))
    edges = np.geomspace(80, fmax, 7)
    for i in range(6):
        basic[f'band_{i+1}'] = float(q[(f>=edges[i]) & (f<=edges[i+1])].sum())
    fft = min(2048, 2**int(np.floor(np.log2(len(y)))))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=fft, hop_length=max(1, sr//200),
                                        n_mels=40, fmin=80, fmax=fmax)
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=14)
    # Exclude c0: mainly energy. Keep mean and spread of timbre coefficients.
    for i in range(1,14):
        basic[f'mfcc_{i:02d}_mean'] = float(mfcc[i].mean())
        basic[f'mfcc_{i:02d}_std'] = float(mfcc[i].std())
    for i, value in enumerate(logmel.mean(axis=1)):
        basic[f'mel_{i:02d}'] = float(value)
    return basic


def feature_columns(frame, family='physics'):
    physics = ['peak_hz','centroid_hz','bandwidth_hz','rolloff85_hz','flatness','entropy',
               'decay10_s','late_early_ratio','peak_count'] + [f'band_{i}' for i in range(1,7)]
    choices = {'physics': physics, 'mfcc': [c for c in frame if c.startswith('mfcc_')],
               'mel': [c for c in frame if c.startswith('mel_')],
               'ast': [c for c in frame if c.startswith('ast_')]}
    if family not in choices or not choices[family]:
        raise ValueError(f'Brak cech dla rodziny {family}.')
    return choices[family]
