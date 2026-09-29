# Weryfikacja obu notebooków — 29.09.2026

Sprawdzono na tym Macu, Python 3.12.14 ze środowiska `.venv` projektu.

- **Świeże kernele:** notebook jakości — 7 komórek kodu, 0 błędów;
  notebook modeli — 9 komórek, 0 błędów. Uruchomienie obejmuje przeliczenie
  modeli; ekstrakcja audio i AST korzystają z istniejącego cache.
- **VS Code:** rzeczywiście kliknięto Run All w obu notebookach i wybrano
  lokalny kernel `.venv`. Oba doszły do komunikatu GOTOWE. Wyniki zapisano.
  Jakość zakończona o 15:24:20, modele o 15:29:54 (Europe/Warsaw).
- **Interakcja:** zmiana R02 na G13 odświeżyła przebieg, spektrogram i tabelę;
  odtwarzacz G13 doszedł do 0:05; zmiana początku na 2 s odświeżyła okno.
  Sprawdzono wizualnie wykresy i tabele w VS Code.
- **Testy kodu:** 22 testy z `test_client_models.py`, `test_audio.py`
  i `test_pipeline.py` zakończone powodzeniem. Obejmują rozłączność
  płytek i nagrań, równoważenie wag i brak wpływu danych testowych na skalowanie.
- **Dane:** zweryfikowano SHA256 wszystkich 49 plików względem archiwów,
  45 nagrań, unikalność 3613 zdarzeń, wspólne okna 170 ms i 410 reprezentacji AST.
  Zmiana samego poziomu o −12 dB nie zmienia ocenianych wyników.
- **HTML:** sprawdzono istnienie wszystkich 45 ścieżek audio i lokalnych linków.
  Przeglądarka HTML nie była przedmiotem tego testu interakcji; interakcję
  sprawdzono w notebooku w VS Code.

Szczegółowe ślady: `reports/runs/client-combined-20260929/validation.json`
oraz `vscode_validation.json`. Powtórzenie testu programowego:

```sh
.venv/bin/python scripts/verify_combined_review.py
```

Ta weryfikacja potwierdza działanie narzędzi i odtwarzalność obliczeń.
Nie zastępuje potwierdzenia fizycznych ID płytek ani niezależnej walidacji
modelu na nowych sztukach i sesjach.

## Uzupełnienie: cechy i tło

`04_cechy_i_tlo.ipynb` wykonano w świeżym kernelu (7 komórek kodu,
0 błędów, około 41 s) oraz osobno przez Run All w VS Code, również do
komunikatu GOTOWE. Nie uruchamia AST. Sześć testów modeli i nowych kontroli
przeszło pomyślnie. Zweryfikowano 17 640 ocen permutacji (30 na cechę
i podział), rozdzielenie płytek we wszystkich pięciu wariantach kontroli
oraz skończone wartości wyników. Udostępnienie parametrów modelu do inspekcji
nie zmieniło wcześniejszych predykcji żadnego z czterech analizowanych modeli.

Wyniki i ślad wykonania: `reports/runs/client-combined-20260929/explanations/`.
Opis i ograniczenia: [cechy i tło](CLIENT_FEATURES_AND_BACKGROUND.md).


## Uzupełnienie: MFCC + physics oraz odporność AST

- Dodano wspólny wektor 41 cech dla LR i XGB, z tymi samymi sześcioma podziałami po płytkach.
- Notebook 03 wykonany w świeżym kernelu: 9 komórek, 0 błędów, około 25 s z cache.
- Notebook 04 wykonany w świeżym kernelu: 7 komórek, 0 błędów, około 73 s.
  Następnie ponownie wykonany przez Run All w VS Code i zapisany z wynikami.
- Ponowna kontrola 03 w interfejsie VS Code utknęła przy restarcie kernela.
  Zatrzymano wyłącznie nowy zawieszony proces tego kernela. W tej aktualizacji
  nie potwierdzono Run All dla 03 w interfejsie; pełne wykonanie programowe jest potwierdzone.
- 27 testów modeli, cech, audio i pipeline przeszło. Sprawdzono 41 unikalnych
  cech bez metadanych oraz rozdzielenie rodzin w teście ważności.
- Osiem modeli, cztery warunki: oryginał, −12 dB, tło SNR 20 i 10 dB.
  AST ponownie zakodował 810 zmienionych okien. Wyniki są lokalnie buforowane.
  Sprawdzono rzeczywisty stosunek RMS dla 540 mieszanek, 648 ocen nagrań
  oraz brak zmiany accuracy/AUC po ściszeniu każdego modelu.
- Na macOS domyślny AST używa jednego wątku CPU, aby uniknąć zaobserwowanego
  konfliktu puli OpenMP po uruchomieniu XGBoost w tym samym procesie.

Ślady: `validation_fusion_noise.json`, `fusion_noise_delivery_validation.json`,
`noise_snr_audit.csv`, `noise_recording_predictions.csv` w katalogu tej rundy.
