# Badminton: sesja z czterema seriami

Gotowe dodatkowe analizy 00–07 znajdziesz w [notebooks/badminton/START.md](../notebooks/badminton/START.md).
Każdy wariant działa samodzielnie i ma zapisane wyniki, w tym osadzone odsłuchy.
Wszystkie osiem przeszło wykonanie w świeżych kernelach; sprawdzono też renderowanie
interaktywnej segmentacji w VS Code. Ponowny test całego zestawu:
`.venv/bin/python scripts/verify_badminton_notebooks.py`.

Otwórz `reports/runs/badminton-four-series/Badminton_eksploracja.ipynb` w VS Code,
wybierz kernel projektu i kliknij **Run All**. Czytelną wersję z zapisanymi wynikami
znajdziesz obok: `Badminton_raport.html`.

Notebook wykorzystuje wspólne funkcje `pinglab`: dekodowanie, wykrywanie uderzeń,
segmentację wybrzmień, ekstrakcję cech, klasyfikator, mieszanie szumu i AST.
Walidacja jest osobna (`pinglab.series`), bo etykiety central/offcenter dotyczą
uderzeń tej samej rakiety. Nie omija wymogu niezależnych płytek w notebookach fabrycznych.
Nie zmienia `project.json` ani wyników demo.

## Wejście i powtórzenie analizy

- `data/raw/badminton/session.wav` — zachowana kopia pełnego eksportu Audacity;
- `data/raw/badminton/session.json` — granice serii, etykiety, parametry i niepewne zdarzenia;
- `scripts/run_badminton_session.py` — buduje notebook, uruchamia wszystkie komórki,
  zapisuje HTML i potwierdzenie wykonania;
- `reports/runs/badminton-four-series/` — WAV-y serii, CSV, model roboczy i interaktywne wykresy.

Uruchom z katalogu repozytorium:

```sh
.venv/bin/python scripts/run_badminton_session.py
```

Źródłowe pliki audio i notebook wynikowy w `reports/runs/` pozostają lokalne w katalogach ignorowanych przez Git.
Nie są dodawane do demonstracyjnej paczki do udostępnienia. Model AST korzysta z lokalnych
wag; ustaw `RUN_AST=False` w notebooku, jeśli wagi nie są dostępne.

## Co oznacza wynik

Trening obejmuje jedną parę serii (central/offcenter), a test drugą; potem zamieniamy
role. Nie dzielimy losowo uderzeń tej samej serii. Skalowanie i progi odległości
pochodzą tylko z treningu. Używamy 48 kHz do analizy, tak jak podstawowe narzędzia
projektu; źródło 44,1 kHz nie zyskuje przez to dodatkowego pasma.

Liczba wykryć nie jest wymuszana do dziesięciu. Etykiety są dziedziczone po serii;
pojedyncze nietrafione odbicie wymaga korekty. Progi zaniku są orientacyjne,
a flaga braku zaniku może wynikać również ze zmiany tła.

Dwie rundy tej samej sesji nie mierzą odporności na nowe rakiety, nowe stanowisko
ani uszkodzenia płyt. Wyniki poszczególnych uderzeń są skorelowane.
Nie stosujemy bootstrapu traktującego każde uderzenie jako niezależną próbkę.

Zapowiedzi zidentyfikowano pomocniczo lokalnym Whisper small, a ich kolejność
porównano z opisem użytkownika. Transkrypcja nie jest potrzebna przy ponownym
Run All; notebook korzysta z jawnych granic w JSON.
Źródło narzędzia: https://github.com/openai/whisper .

## Porównanie XGBoost

`pip install '.[boosting]'` (macOS: także `brew install libomp`).
`python scripts/compare_xgboost.py reports/runs/badminton-four-series/features.csv reports/runs/badminton-xgboost`

Ustalone 80 drzew, głębokość 2, learning rate 0.05, L2=5; bez strojenia na teście.
Podział zachowuje całe rundy, wagi klas liczone wyłącznie na treningu.
CSV zawiera accuracy, balanced accuracy, recall obu klas, AUC oraz treningowe BA.
Notebook w paczce `deliverables/Badminton_Ping_Lab/analiza.ipynb` ma sekcję 10
porównującą również wcześniejsze modele i AST, jeśli został uruchomiony.
Nowe porównanie nie zastępuje zapisanego modelu roboczego.
