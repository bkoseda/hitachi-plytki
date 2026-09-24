# Ping Lab

Lokalny zestaw do szybkiej oceny nagrań z uderzeń w ceramiczne płytki: odsłuch, spektrogram,
rezonanse, wybrzmiewanie, cechy librosa, mały klasyfikator, dobre wzorce i gotowy model AST.

**Zacznij od `index.html`.** To siedem krótkich, interaktywnych kroków z dźwiękiem i wykresami.
Na tym Macu środowisko, dane demo i wagi AST są już przygotowane.

## Uruchomienie na tym Macu

Kliknij dwukrotnie **`Start.command`** w Finderze. Otworzy przewodnik i JupyterLab.
Alternatywnie w Terminalu:

```bash
cd /Users/karolsii/dev/ceramic-ping-lab
.venv/bin/jupyter lab
```

Wybierz `notebooks/01_pierwsza_para.ipynb`, kernel **Ping Lab (Python)**.
Notebooki mają już wyniki przykładowego wykonania. Aby użyć swoich danych, zmień komórkę
**KONFIGURACJA** i wybierz **Run → Run All Cells**. Gdy nie widzisz widgetów: uruchom komórkę ponownie.

| Notebook | Kiedy użyć |
|---|---|
| `01_pierwsza_para.ipynb` | Dwa pliki: upload lub ścieżki, odsłuch, jakość, spektrogram, porównanie pojedynczych uderzeń |
| `02_seria_i_uderzenia.ipynb` | Więcej plików: manifest, ręczna korekta czasów, eksport cech i jakości |
| `03_cechy_i_modele.ipynb` | PCA, odległości, regresja logistyczna, dobre wzorce, test nowych płytek i sesji, zapis modelu |
| `04_gotowy_model_audio.ipynb` | Zamrożony AST; embeddingi i porównanie z prostymi cechami |
| `05_odpornosc_i_protokol.ipynb` | Wpływ gainu i tła; protokół pomiarowy |

## Kiedy przyjdą pliki klienta

1. W notebooku 01 użyj przycisku upload lub skopiuj audio do `data/raw/`.
2. Wklej ścieżki do `FILE_A` i `FILE_B`. Nadaj opisy zgodne z oceną operatora.
3. Sprawdź QC i odsłuch. Dopiero potem interpretuj widmo oraz wybrzmiewanie.
4. Dla serii skopiuj `data/manifest.template.csv` do `data/manifest.csv` i uzupełnij wiersze.
5. Otwórz 02, ustaw manifest i własny plik adnotacji, np. `data/annotations.json`.
6. Zatwierdź czasy uderzeń i sprawdź raport. W 03/04 ustaw **ten sam manifest, adnotacje i parametry**.

`plate_id` jest ID fizycznego detalu, nie pliku. Wiele plików tego samego detalu ma to samo ID.
Etykiety: `ok`, `nok`, `unknown`, `background`. Ścieżki CSV liczone są od lokalizacji CSV.
Szczegóły: [parametry i ograniczenia](docs/METHODS.md).

**Pierwsze dwie próbki służą porównaniu i poprawie pomiaru.** Skuteczność oceniamy na różnych płytkach;
nie na losowo rozdzielonych uderzeniach tej samej płytki. Minimum programowe 3+3 nadal daje tylko wstępny test.
Mediana kilku miejsc może ukryć lokalną wadę — oznacz miejsce wskazane przez operatora.

## Raport bez notebooka

```bash
.venv/bin/pinglab analyze data/manifest.csv \
  --annotations data/annotations.json \
  --output reports/runs/client-v1 --evaluate
```

Wyniki: `report.html`, `features.csv`, `quality.csv`, `run.json`; gdy ocena jest możliwa także
`predictions.csv`, `metrics.json`, `folds.json`. Raport działa offline, zawiera interaktywne wykresy.
CLI nie nadpisuje katalogu istniejącej analizy — podaj nową nazwę. Notebook celowo aktualizuje swój `OUT`.
Jeśli źródło ma 16 kHz, użyj `--fmax 7500` / `FMAX=7500` dla całej porównywanej serii.

## Dane demonstracyjne

- `data/demo/`: **syntetyczne** 30 płytek, 90 uderzeń i tło. Służą do nauki i testów oprogramowania.
- `data/external/realimpact/`: 5 rzeczywistych nagrań demonstracyjnych RealImpact z różnych mikrofonów,
  **bez etykiet defektu**, wspólne ID obiektu. Dobre do sprawdzenia odsłuchu i przetwarzania.
- `reports/demo/report.html`: wynik na symulacji, jednoznacznie oznaczony.
- `reports/realimpact/report.html`: kontrola jakości rzeczywistych nagrań; bez pozornej oceny OK/NOK.

[Źródła, licencje i wybór metod](docs/RESEARCH.md). Żadne z tych danych nie jest walidacją dla klienta.
Pliki audio demo można odtworzyć generatorem; nagrania zewnętrzne, audio klienta, cache i wagi nie trafiają do Git.
Nie dodano zdalnego repo ani nie wysłano danych do usługi audio.

## Odtworzenie środowiska

Python 3.12 jest zalecany; skrypt wspiera 3.11–3.13. Na tym Macu jest już gotowa `.venv`.

```bash
bash scripts/setup.sh
# Dokładne wersje testowane na macOS / Apple Silicon / Python 3.12:
bash scripts/setup.sh --locked
# Opcjonalne małe publiczne demo:
.venv/bin/python scripts/download_realimpact.py
```

AST jest opcjonalny dla notebooków 01–03 i 05. Wersja lekka: `pip install -e .` w nowej `.venv`.
Pełna: `pip install -e '.[embeddings]'`, potem `python scripts/warm_model.py`.
Pierwsze pobranie modelu wymaga internetu (~346 MB wag); potem notebook 04 domyślnie pracuje offline.
CPU przetestowany; nie ma wymogu CUDA ani MPS. FFmpeg dostarcza pakiet `imageio-ffmpeg`.
Nie ma potrzeby instalowania Homebrew ani zmiany systemowego Pythona.

## Testy i odświeżenie materiałów

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_notebooks.py
.venv/bin/python scripts/build_guide_data.py
```

`reports/notebook_validation.json` zapisuje wykonanie wszystkich notebooków; `reports/ui_validation.json`
sprawdzenie interakcji HTML i układu. `docs/VALIDATION.md` opisuje zakres sprawdzenia.

Przewodnik otwiera się bez serwera. Do testu automatycznego UI użyto lokalnego serwera
`python -m http.server 8765 --bind 127.0.0.1` i skryptu `scripts/verify_ui.cjs` (Playwright + lokalny Edge).
Przewodnik i raporty nie ładują bibliotek ani fontów z CDN.
