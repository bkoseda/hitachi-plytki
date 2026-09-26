# Start w VS Code

1. Otwórz **cały folder projektu** (File → Open Folder).
2. Na tym Macu środowisko `.venv` jest gotowe. Na innym komputerze: Python 3.11–3.13, `python3 -m venv .venv`, następnie `.venv/bin/python -m pip install -e .`. W Windows: `.venv\Scripts\python -m pip install -e .`.
3. Rozszerzenia VS Code: Python, Jupyter i Jupyter Notebook Renderers (rekomendowane w folderze `.vscode`). W notebooku wybierz kernel lokalnego `.venv`.
4. Otwórz `notebooks/00_eksploracja_do_przegladu.ipynb` i kliknij **Run All**. Do podstawowego przepływu nie są potrzebne AST, GPU ani przyciski widgetów.

## Podmiana na pliki klienta

- Utwórz `data/raw/client/`, skopiuj audio i `data/manifest.template.csv` jako `manifest.csv`.
- Wypełnij: `path` (względem manifestu), `plate_id` (fizyczny detal), `label` (`ok`, `nok`, `unknown`, `background`), `session_id`. Kilka plików tej samej płyty musi mieć wspólne ID.
- Dodatkowo: `plate_type`, `temperature`, `position`, `photo`, `notes`, `channel`. Jeżeli jeden plik ma kilka miejsc, zachowaj kolejność i opisz ją. Pusta wartość `channel` wybiera kanał o najwyższym RMS.
- Skopiuj zawartość `config/client.example.json` do `project.json`. Wskaż nowy katalog wyników. Wszystkie notebooki odczytają tę samą konfigurację po kolejnym **Run All**.
- W 02 sprawdź czasy i wpisz korekty do `CORRECTIONS`; przyciski edytora są opcjonalne. Brak `annotations.json` oznacza wykrywanie automatyczne. Plik adnotacji należy do konkretnej serii; nie przenoś go do innych nagrań i nie nadpisuj starych nagrań plikami o tej samej nazwie.
- Uruchom ponownie 00, potem 06. Przy dwóch plikach modelowanie zostanie świadomie pominięte. Błędy ścieżek nie powodują powrotu do demo.

`fmax` jest ograniczany do wspólnego pasma wszystkich źródeł. Plik 16 kHz ograniczy porównanie do 8 kHz. Parametry `tail_s`, `min_gap_s` i `sensitivity` zmieniasz wspólnie w `project.json`.

## Który notebook?

| Plik | Zastosowanie |
|---|---|
| 00 | Pełny pierwszy przegląd i wersja do omówienia |
| 01 | Dokładne porównanie dwóch plików/uderzeń |
| 02 | Ręczna kontrola segmentacji i eksport cech |
| 03 | PCA, medianowy baseline, test nowej sesji, zapis modelu |
| 04 | Opcjonalny AST; `pip install -e '.[embeddings]'`, potem `python scripts/warm_model.py` |
| 05 | Wrażliwość na szum; tło bierze z manifestu, bez niego używa białego szumu |
| 06 | Porównanie A/B, błędy płyt i liczba uderzeń |
| 07 | Nagraj mikrofonem, podziel na pingi i sprawdź nakładanie wybrzmień; niezależny od project.json |

## Wysyłka

`deliverables/Ping_Lab_przeglad.html` otwiera się bez Pythona i zawiera zapisane wykresy i dźwięk. `deliverables/Ping_Lab_start.zip` zawiera projekt potrzebny do uruchomienia notebooków; demo odtwarza się lokalnie. AST wymaga osobnego pobrania. Sam `.ipynb` wystarczy do odczytu zapisanych wyników, ale do Run All potrzebny jest kod z ZIP.

Paczka do leada zawiera wyłącznie demo. Nic nie zostało automatycznie wysłane. Eksport danych klienta może zawierać audio, etykiety i opis wad — udostępniaj go świadomie.
