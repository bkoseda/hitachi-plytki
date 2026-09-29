# Dwie paczki klienta

Otwórz cały folder `ceramic-ping-lab` w VS Code. W notebooku wybierz kernel
**Ping Lab (Python)** albo lokalny **.venv (Python 3.12)** i kliknij **Run All**.

1. [02_jakosc_obu_paczek.ipynb](02_jakosc_obu_paczek.ipynb) — zacznij tutaj:
   komplet danych, wyciszanie, odsłuch z wyborem pliku, segmentacja, spektrogramy,
   zmienność ustawienia i zalecenia dla operatora.
2. [03_modele_obu_paczek.ipynb](03_modele_obu_paczek.ipynb) — cechy i zamrożony AST,
   podział po całych płytkach, LR/XGBoost, accuracy/recall, kontrola kodeka i QC,
   MFCC + physics (41 cech) oraz próby z szumem obejmujące również AST.
   Nowe porównania: sekcje 4 i 6.
3. [04_cechy_i_tlo.ipynb](04_cechy_i_tlo.ipynb) — ważność cech regresji i XGBoost,
   wkład całej rodziny MFCC/physics w modelu łączonym (sekcje 2–3),
   modele kontrolne uczone na fragmentach przed pingiem i cichych fragmentach,
   porównanie ataku i ogona. Bez uruchamiania AST.

[Raport HTML](../../reports/runs/client-combined-20260929/START.html) zawiera wyniki
i odtwarzacz wszystkich 45 plików. Działa z dysku, bez serwera. Zachowaj obok
repo z katalogiem `assets`, bo HTML odwołuje się do oryginałów audio.

[Raport tekstowy](../../docs/CLIENT_COMBINED_REPORT.md).
[Krótka lista uwag dla klienta](../../docs/CLIENT_RECORDING_FEEDBACK.md).

Główny test przyjmuje trzy różne płytki OK na podstawie zaakceptowanej interpretacji
folderów. Klient musi jeszcze potwierdzić ID. Pliki OLD i wyszczerbione Bitten są
poza głównym testem pęknięć. Wysoki wynik w tej paczce nie jest pomiarem
skuteczności produkcyjnej.

Oryginały: `assets/1-batch-ping` oraz `assets/2-batch-ok`. Zgodność 49 plików
z obiema paczkami ZIP zapisano w `config/client_archive_manifest.json`.
Wyniki są w `reports/runs/client-combined-20260929`; przeliczanie korzysta
z cache zależnego od treści nagrań. Audio jest przetwarzane lokalnie.

Na innym komputerze trzeba zainstalować projekt z dodatkami `boosting,embeddings`
oraz pobrać wagi AST przez `scripts/warm_model.py`. Na tym Macu są już dostępne.
Weryfikacja programowa: `.venv/bin/python scripts/verify_combined_review.py`.
Kontrole danych: hashe oryginałów, rozłączne grupy train/test, długość okien,
skończone embeddingi i niezmienność wyniku po zmianie samego poziomu.
[Sprawdzone także przez Run All w VS Code](../../docs/CLIENT_COMBINED_VALIDATION.md).

`01_wstepna_analiza.ipynb` dotyczy wcześniejszej pierwszej paczki i pozostaje
jako zapis poprzedniej rundy. Do aktualnego przeglądu używaj 02 i 03.
