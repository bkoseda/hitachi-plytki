# Ping Lab — notebooki na pierwszy dzień

Otwórz cały folder projektu w VS Code, wybierz lokalny kernel `.venv` / **Ping Lab (Python)** i uruchom **notebooks/00_eksploracja_do_przegladu.ipynb → Run All**.

**Jedna konfiguracja:** `project.json` wybiera manifest, adnotacje, katalog wyników i parametry audio dla wszystkich notebooków. Demo jest jawnie oznaczone jako syntetyczne. Gdy brakuje pliku klienta, analiza zgłasza błąd zamiast wracać do demo.

- [Instrukcja uruchomienia i podmiany plików](docs/START_NOTEBOOKI.md)
- [Plan i priorytety minimalnego PoC](docs/PLAN_POC.md)
- [Metody i ograniczenia](docs/METHODS.md)
- [Źródła danych i modeli](docs/RESEARCH.md)
- [Zapis walidacji](docs/VALIDATION.md)

| Notebook | Zastosowanie |
|---|---|
| 00 | Eksploracja do przeglądu: wszystkie pliki, QC, audio, spektrogramy, cechy, pierwsze modele |
| 01 | Szczegółowe porównanie dwóch plików i uderzeń |
| 02 | Kontrola segmentacji, ręczne czasy i eksport |
| 03 | PCA, medianowy baseline, test sesji, zapis modelu |
| 04 | Opcjonalny zamrożony AST; osobne wagi i zależności |
| 05 | Wrażliwość na gain i szum, protokół pomiarowy |
| 06 | Porównanie metod, przeoczenia i fałszywe alarmy, liczba uderzeń |

`index.html` pozostaje interaktywnym przewodnikiem po metodach. `Start.command` otwiera go i JupyterLab jako alternatywę dla VS Code.

**Nagrywanie mikrofonem:** `notebooks/07_nagrywanie_i_wybrzmienia.ipynb`. Ustaw `RECORD=True` i Run All; 20 s nagrania po odliczaniu. Domyślnie `False` pokazuje demo bez otwierania mikrofonu.

## Instalacja na innym komputerze

Python 3.11–3.13, zalecany 3.12. Podstawowy zestaw bez dużych modeli:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m ipykernel install --user --name pinglab --display-name 'Ping Lab (Python)'
```

Windows: użyj `.venv\Scripts\python`. W VS Code wystarczy wybór interpretera `.venv`.
Pełna instalacja z AST na macOS: `bash scripts/setup.sh`; wersje przypięte: `bash scripts/setup.sh --locked`.
AST można dodać osobno: `pip install -e '.[embeddings]'`, następnie `python scripts/warm_model.py`.

## Dane

`data/demo/` zawiera manifest 30 syntetycznych płyt i 90 uderzeń. Audio odtwarza się automatycznie przy pierwszym uruchomieniu demo.
`data/external/realimpact/` to lokalne rzeczywiste nagrania jednego obiektu z różnych mikrofonów, bez etykiet wad; pobieranie: `scripts/download_realimpact.py`.
Żadne z tych danych nie dowodzi wykrywania pęknięć klienta. Dane klienta, wagi, cache i pliki wyników roboczych są poza Git.

Dwie próbki wystarczą do przeglądu. Ocena modeli wymaga większej liczby niezależnych płyt obu klas. `plate_id` oznacza detal, nie plik ani uderzenie.

## Wyniki i wersja do udostępnienia

Po Run All wyniki są w katalogu wskazanym w `project.json`; kolejne wykonanie aktualizuje tę samą analizę. Dla nowej wersji wybierz nowy katalog.

`deliverables/Ping_Lab_przeglad.html` to samodzielny przegląd demo z osadzonym audio i wykresami. `deliverables/Ping_Lab_start.zip` zawiera projekt do ponownego uruchomienia. Budowanie: `python scripts/package_review.py` (wyłącznie z wykonanych notebooków demo).

## Sprawdzenie projektu

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_notebooks.py
.venv/bin/python scripts/verify_client_switch.py
```

Ostatni test wymaga pobranego RealImpact; uruchamia wszystkie notebooki na osobnym WAV i MP3 bez etykiet, bez zmiany `project.json`. Testy podziałów sprawdzają rozdzielenie fizycznych płyt i dopasowanie wzorców tylko na OK. Zapis kontroli VS Code jest w `reports/vscode_validation.json`.
