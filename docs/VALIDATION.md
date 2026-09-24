# Sprawdzenie na tym Macu

Wykonano 24.09.2026 na Apple Silicon, Python 3.12.14; obliczenia modeli na CPU.
Dokładne wersje bibliotek: `requirements-macos-py312.lock`.

## Potwierdzone działanie

- **28 testów pytest:** odczyt WAV, MP3, M4A/AAC, FLAC, OGG; cisza, za krótkie i zbyt długie nagranie,
  NaN, przesterowanie, stereo w przeciwfazie i wymuszenie kanału.
- Segmentacja znanych 3 uderzeń, uderzenie na początku pliku, brak nakładania okien, odrzucenie błędnych czasów.
- Manifest: brakujące i sprzeczne dane, powtórzone pliki, pasmo ograniczone częstotliwością próbkowania.
- 30 syntetycznych płytek / 90 uderzeń: ekstrakcja, rozdzielone płytki w CV, test oddzielnych sesji,
  odmowa oceny na dwóch płytkach, dobre wzorce bez sztucznego zerowego dystansu do samych siebie.
- Zapis modelu i użycie na innych płytkach; odmowa dla danych treningowych i niezgodnych parametrów.
- AST: rzeczywiste wagi, 768 elementów, praca offline, stabilność reprezentacji przy zmianie gainu.
- **5 notebooków wykonanych od początku do końca** bez błędów. Wyniki zachowano w plikach.
  Pierwszy przebieg notebooka AST na 90 uderzeniach trwał ok. 20–22 s; przebieg z cache ok. 7 s.
  To orientacyjny pomiar na tej maszynie, nie obietnica dla dowolnego pliku.
- JupyterLab w przeglądarce: Run All, wczytanie pliku w widżecie, zapis zatwierdzonych czasów.
- HTML: wszystkie 7 ekranów, przełączniki widoków, spektrogram A/B, PCA i sesje, fałszywy podział,
  suwak szumu, lista przygotowań, odczyt plików audio przez przeglądarkę.
- Układ przy 1440 px i 390 px, bez poziomego przepełnienia; brak błędów JavaScript.
- CLI tworzy kompletny raport i odmawia nadpisania istniejącej analizy. Kontrola zależności `pip check`, lint Ruff,
  składnia skryptów bash — poprawne.

## Dane rzeczywiste

Pobrano pięć plików publicznego demo RealImpact (48 kHz / 1.5 s). Dla każdego pipeline znalazł
jednego kandydata uderzenia, wyliczył cechy i utworzył raport. AST wyliczył macierz 5 × 768 również offline.
To test integracji na rzeczywistym audio z innej domeny. Nie ma etykiet wad; nie wyliczano skuteczności detektora pęknięć.

## Test szumu — ważne ograniczenie

`reports/stress_validation.json` zawiera test szerokiego zestawu 15 cech: zmiana gainu nie pogorszyła
wyniku na tej symulacji, lecz nawet dodatek białego szumu 20 dB sprowadził balanced accuracy do 0.5.
Nie ukrywamy tego wyniku. Pokazuje, dlaczego czysty demonstrator nie wystarcza do deklaracji odporności.
Po przetasowaniu etykiet całych płytek średni wynik 39 powtórzeń był ok. 0.47.

Notebook 05 udostępnia również mały zestaw 3 cech oraz trening z zaszumionymi wariantami tylko płytek
treningowych. Wynik dla każdej płytki testowej nadal pochodzi z modelu, który jej nie widział.
Ten wariant w kontrolowanej symulacji lepiej radził sobie z dodanym szumem, ale nie jest dowodem
odporności na rzeczywiste tło fabryki. Przy ocenie klienta należy użyć nowych płytek i nowego nagrania tła / sesji.

## Dowody w repo

- `reports/notebook_validation.json`: notebooki, liczby komórek i czasy.
- `reports/ui_validation.json`: interakcje HTML i dekodowanie audio.
- `reports/jupyter_validation.json`: zapis adnotacji przez działający JupyterLab.
- `reports/ast_smoke.json`: wersja modelu i wymiar reprezentacji.
- `reports/demo/`: jawnie syntetyczny raport, predykcje i rozdział płytek.
- `reports/realimpact/`: rzeczywiste audio bez sztucznych etykiet klas.
- `reports/ui/`: lokalne zrzuty kontrolne (pominięte w Git).

Nie zweryfikowano skuteczności na ceramice klienta — jej nagrań jeszcze nie ma. Nie testowano MPS/CUDA;
przygotowany i sprawdzony wariant używa CPU.
