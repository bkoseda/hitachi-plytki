"""Decode without modifying originals; quality measurements precede normalization."""
from dataclasses import dataclass
from pathlib import Path
import io
import subprocess
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class Config:
    sr: int = 48000
    n_fft: int = 2048
    hop: int = 240
    pre_s: float = 0.025
    tail_s: float = 1.0
    min_gap_s: float = 0.25
    sensitivity: float = 6.0
    max_seconds: float = 180.0

    def __post_init__(self):
        if self.sr < 8000 or self.n_fft < 128 or self.hop < 1:
            raise ValueError('Nieprawidłowe sr / FFT / hop.')
        if not (0 <= self.pre_s < self.tail_s <= 10):
            raise ValueError('Wymagane 0 <= pre_s < tail_s <= 10 s.')
        if self.min_gap_s <= 0 or self.sensitivity <= 0 or self.max_seconds <= 0:
            raise ValueError('Odstęp, czułość i limit czasu muszą być dodatnie.')


@dataclass
class Recording:
    path: Path
    y: np.ndarray
    sr: int
    native_sr: int
    channels: int
    selected_channel: int
    qc: dict


def load_audio(path, config=None, channel=None):
    config = config or Config()
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Brak pliku: {path}')
    try:
        info = sf.info(path)
        if info.duration > config.max_seconds:
            raise ValueError(f'Nagranie > {config.max_seconds:g} s. Podziel je lub zwiększ max_seconds.')
        x, native_sr = sf.read(path, dtype='float32', always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        import imageio_ffmpeg
        # Limit decoding too. An extra second lets us detect oversized compressed inputs.
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-i', str(path),
               '-t', str(config.max_seconds + 1), '-f', 'wav', '-acodec', 'pcm_f32le', 'pipe:1']
        r = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if r.returncode:
            raise ValueError(f'Nie można odczytać {path.name}: {r.stderr.decode(errors="replace")[:350]}')
        x, native_sr = sf.read(io.BytesIO(r.stdout), dtype='float32', always_2d=True)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Puste audio lub wartości NaN/Inf.')
    duration = len(x) / native_sr
    if duration > config.max_seconds:
        raise ValueError(f'Nagranie > {config.max_seconds:g} s. Podziel je lub zwiększ max_seconds.')
    if duration < 0.05:
        raise ValueError('Audio krótsze niż 50 ms: za mało do tej analizy.')
    rms_channels = np.sqrt(np.mean(x.astype(float) ** 2, axis=0))
    selected = int(np.argmax(rms_channels)) if channel is None else int(channel)
    if not 0 <= selected < x.shape[1]:
        raise ValueError('Indeks kanału poza zakresem (numeracja od 0).')
    raw = x[:, selected]
    peak = float(np.max(np.abs(raw)))
    rms = float(rms_channels[selected])
    clip = float(np.mean(np.abs(raw) >= 0.999))
    warnings = []
    if peak < 1e-6:
        warnings.append('CISZA: brak sygnału')
    elif 20 * np.log10(peak) < -30:
        warnings.append('Cichy szczyt: sprawdź gain / odległość')
    if clip > 0:
        warnings.append('Próbki blisko pełnej skali: sprawdź przesterowanie')
    if native_sr < config.sr:
        warnings.append(f'Pasmo źródłowe kończy się przy {native_sr / 2:g} Hz; resampling go nie odzyska')
    if x.shape[1] > 1:
        warnings.append(f'Wybrano kanał {selected}; sprawdź odsłuchem (bez sumowania kanałów)')
    qc = dict(duration_s=duration, native_sr=native_sr, channels=x.shape[1], selected_channel=selected,
              peak_dbfs=float(20 * np.log10(max(peak, 1e-12))), rms_dbfs=float(20 * np.log10(max(rms, 1e-12))),
              clipping_fraction=clip, dc_offset=float(raw.mean()), warnings=warnings)
    y = raw - raw.mean()
    if native_sr != config.sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=config.sr)
    return Recording(path, y.astype(np.float32), config.sr, native_sr, x.shape[1], selected, qc)


def detect_hits(y, sr, config=None):
    """RMS attack candidates, not a ceramic/non-ceramic event classifier."""
    config = config or Config(sr=sr)
    y = np.asarray(y, dtype=float)
    if len(y) < int(0.05 * sr) or np.max(np.abs(y), initial=0) < 1e-7:
        return np.array([], dtype=float)
    hop = max(1, int(sr * 0.005))
    window = max(2, int(sr * 0.01))
    env = np.sqrt(np.maximum(0, uniform_filter1d(y*y, window)))[::hop]
    floor = float(np.median(env))
    mad = float(np.median(np.abs(env - floor)))
    threshold = max(floor + config.sensitivity * max(mad, 1e-8), env.max() * 0.08)
    if env.max() < max(3 * floor, 1e-7):
        return np.array([], dtype=float)
    # Padding catches attacks close to the start of a file.
    peaks, _ = find_peaks(np.r_[0, env, 0], height=threshold,
                          prominence=max(3 * mad, env.max() * 0.08),
                          distance=max(1, round(config.min_gap_s * sr / hop)))
    peaks = peaks - 1
    times = []
    for p in peaks:
        start = p
        local_level = max(floor + 2 * mad, env[p] * 0.15)
        lower = max(0, p - round(0.08 * sr / hop))
        while start > lower and env[start - 1] > local_level:
            start -= 1
        times.append(max(0.0, start * hop / sr - 0.005))
    return np.unique(times)


def cut_hits(y, sr, onsets, config=None):
    config = config or Config(sr=sr)
    times = np.asarray(onsets, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Czasy uderzeń muszą być skończoną listą sekund.')
    if len(times) and (np.any(times < 0) or np.any(times >= len(y)/sr) or np.any(np.diff(times) <= 0)):
        raise ValueError('Czasy muszą rosnąć, być unikalne i leżeć wewnątrz nagrania.')
    result = []
    for i, onset in enumerate(times):
        start = max(0, int(round((onset - config.pre_s)*sr)))
        stop = min(len(y), int(round((onset + config.tail_s)*sr)))
        if i + 1 < len(times):
            stop = min(stop, int(round((times[i+1] - config.pre_s)*sr)))
        if stop - start < int(0.05*sr):
            raise ValueError('Uderzenia są zbyt blisko: okno krótsze niż 50 ms.')
        result.append(dict(y=y[start:stop].copy(), start_s=start/sr, end_s=stop/sr,
                           onset_s=float(onset), onset_offset_s=float(onset-start/sr),
                           truncated=bool(stop < int(round((onset+config.tail_s)*sr)))))
    return result


def listen(y, sr, normalize=False):
    from IPython.display import Audio
    # Explicit control: IPython normally normalizes each file independently.
    return Audio(np.asarray(y), rate=sr, normalize=normalize)
