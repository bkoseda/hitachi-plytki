# Pierwsza paczka klienta — 29.09.2026

Raport do przeglądu: `reports/runs/client-20260929/START.html`.
Notebook z wykonanymi komórkami: `reports/runs/client-20260929/analiza.ipynb`.
Źródło notebooka bez prywatnych wyników: `notebooks/client/01_wstepna_analiza.ipynb`.
Uwagi do klienta: `reports/runs/client-20260929/UWAGI_DO_KLIENTA.md` (nie wysłano).

Prywatna kopia paczki: `data/raw/client-20260929/Recordings/`. Sprawdzono SHA256
wszystkich 24 plików audio względem początkowego folderu Downloads/Recordings.
Folder Downloads przestał być dostępny podczas analizy, więc kopię odtworzono
z pozostałego Recordings.zip. Wyniki i dane pozostają poza Git.

## Wykonanie

```sh
.venv/bin/python scripts/verify_client_recordings.py
```

Notebook uruchamia `analyze_client_recordings.py`, `client_recording_diagnostics.py`
i `build_client_review.publish()`. Korzysta z istniejących `pinglab.audio`,
`pinglab.features`, `pinglab.recorder`, `pinglab.plots` i `pinglab.embeddings`.
AST używa lokalnych wag i cache; pięć zdarzeń wybranych równomiernie po indeksach
czasowo uporządkowanych kandydatów z każdego z 23 plików uderzeń. Bez strojenia.
Raport HTML wymaga sąsiednich katalogów audio/figures i plotly.min.js.

## Ustalenia i ograniczenia

- 24 audio, 744,562 s, 48 kHz; 19 AAC i 5 PCM WAV. Kanały są identyczne.
- Bad 1–2: robocze ID z folderów, zgodne z opisem dwóch pękniętych płyt.
  Bitten osobno. Nieznane ID fizycznych sztuk OLD, brak rozstrzygnięcia duplikatów obiektów.
- 2395 kandydatów RMS (w tym tło), 2308 zaakceptowanych do cech. Nie jest to liczba
  ręcznie potwierdzonych uderzeń ani niezależnych płytek.
- Detektor: odstęp 120 ms, próg domyślny 6. Kandydat musi mieć kontrast ataku >=12 dB,
  pełne okno 20 ms przed +150 ms po, poza pierwszymi 0,75 s i ostatnimi 0,4 s.
  Tło nigdy nie trafia do cech klasyfikacyjnych. Każdy kandydat zostaje w audycie.
- Przegląd wykresów pokazał trafienia co około 200–300 ms. Poprzednie 250 ms pomijało część;
  okna 350 ms nakładałyby się na następne zdarzenie. Nie estymujemy fizycznego tłumienia.
- Nie trenujemy ani nie oceniamy klasyfikatora, dopóki nie znamy grup fizycznych płytek.
  Nie przedstawiamy PCA, dystansów ani perturbacji cech jako accuracy.
- Tło R16: 20,3241% próbek dokładnie zero, 22,9% ramek poniżej -100 dBFS.
  Podejrzenie przetwarzania, nie rozpoznanie konkretnej technologii. Do wyjaśnienia z klientem.
- Próba lokalnej transkrypcji Whisper dawała niespójne krótkie teksty w różnych językach.
  Nie użyto ich jako etykiet, treści rozmów ani podstawy segmentacji. Pełny odsłuch człowieka
  pozostaje do zatwierdzenia w atlasie. Wykresy i fotografie przejrzano wizualnie.
- Run All: 8 komórek kodu, bez błędów w świeżym kernelu. Kontrole: hashe źródeł,
  równe okna, skończone cechy/embeddingi, wyłączenie tła, działające lokalne odnośniki HTML.
  Automatyczny podgląd HTML w przeglądarce był zablokowany dla protokołu file://;
  sprawdzono strukturę i zasoby statycznie, bez deklaracji testu interakcji w przeglądarce.

Zmiana reguł segmentacji wymaga przeliczenia wszystkich reprezentacji i ponownego przeglądu.
Nie uznawać ręcznie wybranych parametrów pierwszej eksploracji za zamrożony protokół produkcyjny.
