# Plan minimalnego PoC — do omówienia

Nowe wiadomości potwierdzają zakres: 2–3 tygodnie, kilka prostych iteracji, odpowiedź na pytanie o wykonalność. Nie zakładamy wdrożeniowej skuteczności ani treningu sieci. Poniższa kolejność jest propozycją inżynierską, nie bezkrytycznym wykonaniem dokumentu AI.

| Priorytet | Co robimy | Gotowe narzędzie |
|---|---|---|
| 1 | Odsłuch wszystkich plików, jakość, spektrogramy, korekta uderzeń | 00, 01, 02 |
| 1 | Rezonanse, zanik i zmienność między uderzeniami | 00, 06 |
| 1 | Dystans od OK i regularizowana regresja na tych samych cechach/podziałach | 06 |
| 1 | MFCC + mały klasyfikator, decyzja dla całej płyty | 06 |
| 2 | Błędy pojedynczych płyt, typ/temperatura/sesja, szum i liczba uderzeń | 00, 03, 05, 06 |
| 3 | Zamrożony AST jako alternatywna reprezentacja | 04 — opcjonalny |
| Poza pierwszą iteracją | Pełny trening sieci, strojenie wielu modeli, SHAP, precyzyjne tłumienie każdego modu | Dopiero po obejrzeniu realnych danych |

## Co zmieniamy względem propozycji AI

- **Jednoklasowy nie ma automatycznie przewagi przy małej próbie.** Dystans i klasyfikator sprawdzamy obok siebie. Oba mogą nauczyć się warunków nagrania.
- **Próg bez przeoczeń dobieramy na wewnętrznych prognozach treningowych**, nie na płytach testowych. Zewnętrzny wynik może mieć przeoczenia. Jeśli danych nie wystarcza, raport jawnie używa progu stałego.
- **Średnia i odchylenie cech** uzupełniają wcześniejszą medianę. Dla MFCC oceniamy średnią z trzech najwyższych ocen uderzeń; to wariant do sprawdzenia, nie założony zwycięzca. Utrzymuj stałą liczbę i miejsca uderzeń.
- **Nie zamrażamy złego podziału na zawsze.** Najpierw korygujemy go po odsłuchu, potem zapisujemy wspólną konfigurację i przeliczamy całą serię.
- **Normalizacja amplitudy nie usuwa wszystkich skutków siły uderzenia.** Nie filtrujemy 50 Hz w ciemno; sprawdzamy widmo tła. Zanik jest prostym deskryptorem, nie laboratoryjnym współczynnikiem tłumienia.
- **Nie utożsamiamy wyniku 10/10 z gwarancją 100%.** Mała próba, niepewne etykiety i nowe warunki ograniczają wniosek. Cel biznesowy trzeba oddzielić od tego, co udowodni PoC.
- **Krzywa liczby uderzeń:** losowanie bez zwracania na tej samej grupie płyt, stały model i próg. Nie dodajemy sztucznych stuknięć, gdy brakuje ich w nagraniu.

## Co uzgodnić przy pierwszych plikach

ID fizycznej płyty, wiarygodność OK/NOK, typ płyty, temperatura, podparcie, miejsca uderzeń, ustawienie mikrofonu i osobne tło. Nieznana etykieta = `unknown`. Dwie próbki wystarczą do przeglądu technicznego, nie do oceny skuteczności. O liczbie wzorców decydują płyty, nie pliki ani uderzenia.

Na pierwszy dzień potrzebny jest wniosek: **nagrania użyteczne / co poprawić / jaki sygnał warto sprawdzić w większej serii**. Wysoka AUC na jednej małej serii sama nie uzasadnia pilota produkcyjnego.

Źródła zasad walidacji: [progi decyzji w scikit-learn](https://scikit-learn.org/stable/modules/classification_threshold.html), [novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html). Obsługa: [notebooki w VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).
