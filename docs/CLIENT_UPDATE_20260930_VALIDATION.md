# Walidacja rundy 30.09.2026

- 16 nowych WAV-ów odpowiada bajt w bajt wpisom w dwóch dostarczonych ZIP-ach.
  SHA256 archiwów i członków: `config/client_archives_20260930.json`.
- Sprawdzono również 49 starszych plików (45 audio + 4 zdjęcia): oryginały bez zmian.
- Razem 61 nagrań; brak identycznych plików i identycznych zdekodowanych sygnałów
  pomiędzy nimi. Wybrano 558 okien z pełnego zbioru, w tym 418 do głównego modelowania
  pęknięć. Nowe nagrania: 258 przyjętych kandydatów; 148 okien do porównań modeli.
- 13 testów `test_client_update.py`, `test_client_models.py`, `test_client_explanations.py`
  przeszło. Obejmują m.in. nierozdzielanie aliasów tej samej zakładanej płytki,
  brak mieszania nagrań między train/test i brak wpływu etykiet testowych na fit.
- `scripts/verify_client_update.py`: zgodność hashy, rozłączne podziały we wszystkich
  scenariuszach, skończone embeddingi, 27 wcześniejszych nagrań vs 15 nowych,
  poprawne NaN dla nieobecnej klasy, lokalne odwołania HTML i wykonanie notebooka.
- `05_nowe_paczki_20260930.ipynb`: wszystkie 8 komórek kodu przeszło na świeżym
  kernelu `pinglab` przez nbclient. Następnie końcowa wersja została uruchomiona
  przez przycisk Run All w VS Code na `.venv` 3.12.14 i zapisana (liczniki 17–24).
  Obejrzano końcowy wynik w edytorze. Sprawdzono lint i format zmienionych skryptów.
- Nowy notebook używa zwykłego odtwarzacza audio i parametrów wyboru fragmentu
  w komórce. Nie wymaga ładowania widgetów JavaScript z CDN, z czym to środowisko
  VS Code ma problem. Starszych, niezapisanych notebooków użytkownika nie zmieniano.
- Poprawiono wizualnie tytuły wykresów; sprawdzono zasoby HTML i ścieżki do audio.
  Automatyczny podgląd HTML przez narzędzie przeglądarki zablokował protokół file://;
  nie jest to błąd obliczeń ani dowód kontroli wyglądu całego HTML w przeglądarce.

Ograniczenia: ID fizycznych płytek i status ulepszeń LAB TEST niepotwierdzone.
Detekcja uderzeń jest automatyczna, nie zatwierdzono ręcznie każdego zdarzenia.
Nie przeprowadzono w tej rundzie nowych testów domieszki szumu ani pełnego zaniku.
Zachowano wspólne okna 170 ms dla porównania z poprzednią rundą. Trening i ocena
są eksploracyjne; wyników nie należy traktować jako pomiaru skuteczności produkcyjnej.

Odtworzenie na tym komputerze: Run All notebooka 05, następnie
`.venv/bin/python scripts/verify_client_update.py`.
Na nowym komputerze wymagane zależności projektu z dodatkami `boosting,embeddings`
i lokalne wagi AST (`scripts/warm_model.py`). Brakujące wyniki wcześniejszej
rundy są odtwarzane automatycznie z oryginałów.
