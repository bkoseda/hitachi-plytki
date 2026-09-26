# Badminton: sesja z czterema seriami

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

Pliki audio i notebook wynikowy pozostają lokalne w katalogach ignorowanych przez Git.
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
