from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = [
    (
        "md",
        """# 07 · Nagraj ping i zobacz wybrzmienie
**Pierwszy test z rakietą:** 1 s ciszy → dwa stuknięcia z długą przerwą → dwa szybsze → kilka sekund ciszy na koniec.
Trzymaj rakietę w podobny sposób; nie dotykaj mikrofonu. Sposób trzymania sam może zmieniać wybrzmienie.

**Jak nagrać:** kernel `.venv` → ustaw `RECORD = True` w komórce poniżej → **Run All**.
Po odliczaniu masz domyślnie 20 s. macOS może poprosić o dostęp do mikrofonu dla VS Code — zaakceptuj, jeśli chcesz nagrywać.
Przy `RECORD = False` możesz analizować wskazany WAV/MP3 albo obejrzeć jawnie oznaczone demo. Nic nie nagrywa się w tle.""",
    ),
    (
        "code",
        """from pathlib import Path
import sys
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Audio
ROOT = Path.cwd()
if not (ROOT/'pyproject.toml').exists(): ROOT = ROOT.parent
assert (ROOT/'pyproject.toml').exists(), 'Otwórz folder ceramic-ping-lab w VS Code.'
sys.path.insert(0,str(ROOT/'src'))
from pinglab.recorder import input_devices, record_wav, demo_sequence, segment_ringdown, ringdown_plot, export_pings
from pinglab.audio import load_audio, Config
print('Wejścia audio — DEVICE=None używa domyślnego mikrofonu systemu:')
try:
    display(input_devices())
except Exception as error:
    print('Nie można odczytać mikrofonów:',error)
    print('Możesz nadal analizować plik lub demo.')""",
    ),
    (
        "code",
        """# USTAW TUTAJ
RECORD = False             # True = nagraj teraz; False = plik / ostatnie nagranie w tym kernelu / demo
SECONDS = 20
DEVICE = None              # None = domyślny mikrofon; lub numer z tabeli powyżej
SAMPLE_RATE = None         # None = natywna częstotliwość urządzenia, zwykle 48000 Hz
AUDIO_FILE = None          # np. ROOT/'data/raw/recordings/ping_....wav'

# Zmień te parametry po odsłuchu, jeśli detektor myli się:
MANUAL_ONSETS = None       # np. [1.05, 3.2, 3.55]; sekundy. None = automat, [] = brak uderzeń
MIN_GAP = 0.12             # najmniejszy odstęp między kandydatami
SENSITIVITY = 0.10         # większa wartość = mniej kandydatów (0…1)
NOISE_SECONDS = 0.5        # pierwsze pół sekundy musi być bez stuknięć
DROP_DB = 30               # roboczy koniec ogona: spadek o 30 dB lub zejście do tła + 6 dB
QUIET_HOLD = 0.08          # jak długo poziom ma pozostać poniżej progu
""",
    ),
    (
        "code",
        """if RECORD:
    LAST_RECORDING = record_wav(ROOT/'data/raw/recordings',SECONDS,DEVICE,SAMPLE_RATE)
    source_path = LAST_RECORDING
else:
    source_path = Path(AUDIO_FILE).expanduser() if AUDIO_FILE else globals().get('LAST_RECORDING')
if source_path is None:
    y,sr,_ = demo_sequence()
    SOURCE = 'DEMO — tłumione sinusoidy, nie Twoja rakieta'
    display(Markdown('**'+SOURCE+'**'))
else:
    recording = load_audio(source_path,Config(max_seconds=180))
    y,sr = recording.y,recording.sr
    SOURCE = Path(source_path).name
    print('PLIK:',source_path)
    display(pd.Series(recording.qc))
display(Audio(y,rate=sr,normalize=False))
print('Analizujemy:',SOURCE)""",
    ),
    (
        "md",
        """## Uderzenia i ogony
**Czerwony:** umowna faza ataku (pierwsze 30 ms, nie pomiar długości kontaktu).
**Zielony:** obwiednia zeszła poniżej progu i pozostała tam przez zadany czas.
**Pomarańczowy:** wcześniej pojawił się kolejny ping albo skończyło się nagranie.

Próg w środkowym panelu to umowny koniec słyszalnego ogona. Zmiana progu zmienia granicę; to nie pomiar fizycznej stałej tłumienia. Przybliż wykres myszą i sprawdź, czy kreski trafiają w faktyczne ataki. Jeśli nie, wpisz poprawne sekundy w `MANUAL_ONSETS`. Mowa i szelest także mogą dać kandydatów.""",
    ),
    (
        "code",
        """segments,t,envelope_db = segment_ringdown(y,sr,MANUAL_ONSETS,
    min_gap_s=MIN_GAP,sensitivity=SENSITIVITY,noise_seconds=NOISE_SECONDS,
    drop_db=DROP_DB,quiet_hold_s=QUIET_HOLD)
display(ringdown_plot(y,sr,segments,t,envelope_db))
display(segments[['ping','onset_s','attack_end_s','end_s','duration_s','status']].round(3))
if segments.empty:
    print('Brak kandydatów. Sprawdź odsłuch, zmniejsz SENSITIVITY lub podaj czasy ręcznie.')
if not segments.empty and segments.onset_in_noise_window.any():
    print('UWAGA: uderzenie w odcinku użytym do oszacowania tła. Skróć NOISE_SECONDS lub nagraj sekundę ciszy na początku.')
print('Czas wybrzmienia urwanego przez następne uderzenie jest tylko dolną granicą, nie pełnym pomiarem.')""",
    ),
    (
        "md",
        """## Posłuchaj fragmentów
Każdy fragment kończy się przy zaobserwowanym zaniku albo przed następnym uderzeniem. Głośność jest oryginalna.
Jeżeli poprzedni ping jeszcze wybrzmiewał, **następny fragment zawiera również jego resztkę**. Dla czystego porównania barwy i zaniku zrób najpierw serię z dłuższymi przerwami.

Po nagraniu ustaw `RECORD=False`, aby zmieniać detekcję bez ponownego włączania mikrofonu. Ostatni plik jest pamiętany do restartu kernela; później wpisz jego ścieżkę w `AUDIO_FILE`.""",
    ),
    (
        "code",
        """for i,row in enumerate(segments.itertuples()):
    if i >= 12:
        print('Odsłuch ograniczony do 12 fragmentów; wykres i tabela obejmują wszystkie.')
        break
    previous_overlap = i > 0 and segments.iloc[i-1].status == 'kolejne uderzenie przed zanikiem'
    print(f'Ping {row.ping}: {row.status}' + (' · zawiera resztkę poprzedniego pingu' if previous_overlap else ''))
    start=max(0,round((row.onset_s-.015)*sr))
    display(Audio(y[start:round(row.end_s*sr)],rate=sr,normalize=False))""",
    ),
    (
        "md",
        "## Opcjonalny eksport\nOryginalny WAV już jest zapisany. Poniżej możesz dodatkowo zapisać osobne fragmenty i tabelę granic. Każdy eksport dostaje nowy katalog.",
    ),
    (
        "code",
        """SAVE_SEGMENTS = False
if SAVE_SEGMENTS:
    from datetime import datetime
    from uuid import uuid4
    out = ROOT/'data/raw/recordings'/f"segments_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    paths = export_pings(y,sr,segments,out)
    print('Zapisano:',out,'· fragmentów:',len(paths))
print('GOTOWE · nagrywanie i wybrzmienia ·',SOURCE)""",
    ),
    (
        "md",
        "Nagrywanie lokalne: [python-sounddevice](https://python-sounddevice.readthedocs.io/en/latest/usage.html). Mikrofon jest otwarty tylko podczas jawnego wywołania nagrania. Jeśli nagranie jest ciche/puste, sprawdź **Ustawienia systemowe → Prywatność i ochrona → Mikrofon → Visual Studio Code** oraz wybrane wejście dźwięku.",
    ),
]
nb = nbf.v4.new_notebook(
    cells=[nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
)
nb.metadata = dict(
    kernelspec=dict(display_name="Ping Lab (Python)", language="python", name="pinglab"),
    language_info=dict(name="python", version="3.12"),
)
nbf.write(nb, root / "notebooks/07_nagrywanie_i_wybrzmienia.ipynb")
