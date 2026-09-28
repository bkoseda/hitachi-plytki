# Badminton — notebooki do prezentacji

Każdy plik ma zapisane wyniki i działa niezależnie przez **Run All**, z kernelem projektu `.venv` / Ping Lab. Dane: jedna rakieta, cztery serie, dwie rundy. To test rozróżniania miejsca trafienia, nie detekcji pęknięć.

| Notebook | Co pokazać |
|---|---|
| [00 — Przegląd](00_eksploracja_do_przegladu.ipynb) | Cała sesja, odsłuchy, spektrogramy, cechy i pierwsze modele. |
| [01 — Para A/B](01_pierwsza_para.ipynb) | Dwa wybrane uderzenia, odsłuch oryginalny i wyrównany, interaktywne widma. |
| [02 — Segmentacja](02_seria_i_uderzenia.ipynb) | Cztery serie, korekta czasów uderzeń, powiększanie wykresu. |
| [03 — Cechy i modele](03_cechy_i_modele.ipynb) | PCA, mapa podobieństwa, pomyłki i odległość od wzorców. |
| [04 — AST](04_gotowy_model_audio.ipynb) | Embeddingi, mapa PCA i porównanie z klasycznymi cechami. |
| [05 — Odporność](05_odpornosc_i_protokol.ipynb) | Odsłuch szumu, wpływ głośności i szumu na cechy oraz predykcje. |
| [06 — Porównanie](06_porownanie_metod.ipynb) | Regresja, odległość, AST, XGBoost; recall, accuracy, BA, liczba uderzeń. |
| [07 — Wybrzmienia](07_nagrywanie_i_wybrzmienia.ipynb) | Interaktywny podział atak/ogon, eksport fragmentów; opcjonalny mikrofon. |

**Na krótką prezentację:** 01 → 07 → 06. Pełny przegląd: 00.

Wersje HTML są w `reports/runs/badminton-notebooks/` (otwórz `START.html`).

Modele sprawdzamy na odłożonej całej rundzie, nie na losowo wymieszanych uderzeniach. Nie uruchamiamy walidacji niezależnych płytek, bo mamy jedną rakietę. Nie wykonujemy testu permutacyjnego, który udawałby niezależność uderzeń, ani testu nowej sesji bez jej nagrania. Test szumu w 05 zmienia dane testowe, bez treningu z augmentacją.

Oryginalne notebooki 00–07 oraz `project.json` nadal służą ogólnemu przepływowi dla płytek. Warianty badmintonowe czytają `data/raw/badminton/session.json` i źródłowy WAV. Korekty z 02 są lokalne dla jego uruchomienia; nie zmieniają automatycznie innych notebooków.

AST korzysta z lokalnych wag (`python scripts/warm_model.py` na nowej instalacji). Porównanie XGBoost wymaga `pip install '.[boosting]'`, a na macOS `brew install libomp`; na tym komputerze wszystko jest już dostępne. 07 otwiera mikrofon wyłącznie po jawnym ustawieniu `RECORD=True`.
