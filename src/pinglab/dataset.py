"""Manifest paths are relative to the CSV; plate identity is mandatory."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import numpy as np
import pandas as pd
from .audio import Config, load_audio, detect_hits, cut_hits
from .features import extract_features

REQUIRED = ['path', 'plate_id', 'label', 'session_id']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f'Brak kolumn manifestu: {sorted(missing)}')
    if df.empty or any((df[c].str.strip() == '').any() for c in REQUIRED):
        raise ValueError('Manifest pusty lub brakuje ścieżki / ID / etykiety / sesji.')
    if not set(df.label) <= {'ok', 'nok', 'unknown', 'background'}:
        raise ValueError('Etykiety: ok, nok, unknown, background.')
    df['resolved_path'] = df.path.map(lambda x: str((path.parent / Path(x).expanduser()).resolve()))
    if df.resolved_path.duplicated().any():
        raise ValueError('Ten sam plik występuje dwukrotnie w manifeście.')
    for f in df.resolved_path:
        if not Path(f).is_file():
            raise FileNotFoundError(f)
    objects = df[df.label != 'background']
    if (objects.groupby('plate_id').label.nunique() > 1).any():
        raise ValueError('Jeden plate_id ma sprzeczne etykiety.')
    df['sha256'] = df.resolved_path.map(sha256)
    if df.sha256.duplicated().any():
        raise ValueError('Identyczna zawartość audio w kilku wierszach. Usuń kopie przed analizą.')
    for col in ['position', 'environment', 'notes', 'photo']:
        if col not in df:
            df[col] = ''
    return df


def analyze_manifest(manifest, output, config=None, fmax=20000, annotations=None):
    config = config or Config()
    df = read_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    annotations = annotations or {}
    if isinstance(annotations, (str, Path)):
        annotations = json.loads(Path(annotations).read_text())
    rows, quality = [], []
    for rec in df.to_dict('records'):
        audio = load_audio(rec['resolved_path'], config)
        q = {k: rec[k] for k in REQUIRED + ['sha256']}
        q.update(audio.qc)
        q['warnings'] = ' | '.join(q['warnings'])
        if fmax > audio.native_sr/2:
            raise ValueError(f"{rec['path']}: fmax={fmax} ponad pasmem źródła. Ustaw wspólne fmax <= {audio.native_sr/2:g}.")
        if rec['label'] == 'background':
            q.update(n_hits=0, segmentation='background', reviewed=False)
            quality.append(q)
            continue
        override = annotations.get(rec['path'])
        onsets = detect_hits(audio.y, audio.sr, config) if override is None else np.asarray(override, float)
        segments = cut_hits(audio.y, audio.sr, onsets, config)
        q.update(n_hits=len(segments), segmentation='auto' if override is None else 'manual',
                 reviewed=override is not None)
        if not segments:
            q['warnings'] += ' | Brak uderzeń: popraw czasy lub sprawdź plik'
        quality.append(q)
        for i, segment in enumerate(segments):
            feats = extract_features(segment['y'], audio.sr, segment['onset_offset_s'], fmax=fmax)
            row = {k: rec[k] for k in REQUIRED + ['position','environment','notes','photo','sha256']}
            row.update(hit_id=f"{rec['plate_id']}:{rec['sha256'][:10]}:{i+1}",
                       start_s=segment['start_s'], end_s=segment['end_s'], onset_s=segment['onset_s'],
                       onset_offset_s=segment['onset_offset_s'], truncated=segment['truncated'],
                       reviewed=override is not None, resolved_path=rec['resolved_path'])
            rows.append({**row, **feats})
    features = pd.DataFrame(rows)
    qc = pd.DataFrame(quality)
    features.to_csv(output/'features.csv', index=False)
    qc.to_csv(output/'quality.csv', index=False)
    metadata = dict(manifest=str(Path(manifest).resolve()), config=asdict(config), fmax=fmax,
                    manifest_sha256=sha256(manifest), files=df[['path','sha256']].to_dict('records'))
    (output/'run.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return features, qc


def event_audio(row, config=None):
    audio = load_audio(row['resolved_path'], config)
    return audio.y[round(row['start_s']*audio.sr):round(row['end_s']*audio.sr)], audio.sr
