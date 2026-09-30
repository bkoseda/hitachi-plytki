# Oryginały nagrań klienta

Wszystkie dostawy są tutaj, pod wspólnym `assets`. Nazwy wewnątrz paczek
pozostały bez zmian, łącznie z literówkami. Nie nadpisujemy oryginalnego audio.

| Katalog | Źródło | Audio | Identyfikatory analizy |
|---|---|---:|---|
| `1-batch-ping` | Recordings, wcześniejsza dostawa | 24 + 4 zdjęcia | R01–R24 |
| `2-batch-ok` | OK.zip | 21 | G01–G21 |
| `3-batch-enhancements-off` | OneDrive_2026-09-30.zip | 3 | E01–E03 |
| `4-batch-lab-test` | OneDrive_2_9-30-2026.zip | 13 | L01–L13 |

Suma: 61 plików audio. Dwa nagrania samego tła są poza klasyfikacją.
`demo-*.wav` i `noise-*.wav` obok katalogów nie należą do danych klienta.

Nowe `BAD/BAD2` zawiera 7 nagrań NOK, `OK` zawiera 6 nagrań OK.
`Mic Enhancements Off` zawiera 2 OK i tło. Deklaracja wyłączenia ulepszeń
pochodzi z nazwy folderu; nie przypisujemy jej automatycznie paczce LAB TEST.
Końcówki 2/3 są roboczo numerami powtórzeń, nie nowymi ID fizycznych płytek.
Nowe BAD2 może oznaczać wcześniejszą Bad_2 lub nową sztukę — wymaga wyjaśnienia.
Bitten pozostaje osobnym typem defektu, OLD ma niepewną tożsamość.

Sumy kontrolne oryginałów:
- `config/client_archive_manifest.json` — 49 starszych plików (w tym zdjęcia).
- `config/client_archives_20260930.json` — 16 nowych, także nazwy i SHA256 ZIP-ów.

Aktualny notebook: `notebooks/client/05_nowe_paczki_20260930.ipynb`.
Wyniki i dokładne przypisanie ID → ścieżka: `reports/runs/client-update-20260930/all_recordings.csv`.
Raport: `reports/runs/client-update-20260930/START.html`.
Katalog wyników i modele są lokalne, odtwarzane przez Run All; nie są wersjonowane.
