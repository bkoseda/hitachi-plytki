# Co wykorzystują modele i czy rozpoznają tło?

29.09.2026 · uzupełnienie analizy obu paczek

**Kierunek produkcyjny: MFCC + regresja logistyczna, porównywane z XGBoostem. AST pozostaje tylko punktem odniesienia.** Wyniki poniżej dotyczą pięciu zakładanych płytek, nie potwierdzonych mechanizmów wykrywania pęknięć.

## Połączenie MFCC + physics

Wspólny wektor ma 41 cech. W aktualnym porównaniu LR daje 27/27 poprawnych nagrań, XGB 26/27: tyle samo co samo MFCC. Physics nie poprawiło tutaj accuracy. W modelach łączonych przetasowanie całego MFCC obniża balanced accuracy średnio o około 41 punktów procentowych, a całego physics o około 5 pp (LR) lub 7 pp (XGB). Model korzysta więc z physics, ale główną rolę pełni MFCC. To diagnostyka sześciu współzależnych podziałów, nie dowód fizycznej przyczyny.

## Co dostaje model i jak się uczy

1. Z nagrania wykrywamy kandydatów na uderzenia. Każde zaakceptowane okno ma 170 ms: 20 ms przed początkiem i 150 ms po nim. Odejmujemy składową stałą i normalizujemy amplitudę.
2. Jedno uderzenie staje się jednym wierszem: **26 liczb MFCC** albo **15 cech physics**. MFCC to 13 współczynników opisujących kształt widma, każdy jako średnia i odchylenie standardowe po krótkich ramkach wewnątrz uderzenia. Nie używamy współczynnika c0, głównie opisującego energię.
3. Z każdego z 27 plików bierzemy 10 uderzeń rozłożonych w czasie: razem 270 wierszy. Etykieta pochodzi z płytki. Dwie klasy i poszczególne płytki mają zrównoważone wagi.
4. W każdej z sześciu prób uczymy na jednej NOK i dwóch OK; testujemy całą drugą NOK i jedną OK. Skaler uczy się tylko na treningu. Regresja: C=0.1. XGBoost: 60 drzew, głębokość 2, learning_rate=0.05, reg_lambda=10; parametry nie są optymalizowane na tym zbiorze.
5. Model ocenia **każde uderzenie osobno**. Regresja sumuje wystandaryzowane cechy pomnożone przez wagi i przepuszcza wynik przez sigmoidę. XGBoost sumuje wkład małych drzew z pytaniami o progi cech. Żaden nie analizuje kolejności tapów ani odstępów w serii.
6. Decyzja nagrania to mediana ocen jego 10 uderzeń, z progiem 0.5. W tabeli eksperymentu uśredniamy wyniki modeli z prób, w których plik był odłożony. Model wdrożeniowy trzeba później osobno dopasować i ocenić na nowej sesji; obecna ocena nie jest skalibrowanym prawdopodobieństwem pęknięcia.

## Które cechy są najistotniejsze

Na odłożonych płytkach podmieniamy jedną cechę między nagraniami, przenosząc cały profil 10 tapów. Pozostałe cechy pozostają bez zmian. Powtarzamy 30 razy na każdy podział. Jeśli model zaczyna popełniać większy błąd, ta cecha jest mu potrzebna. Ranking poniżej używa wzrostu **log-loss** — miary karzącej również błędną pewność. Podajemy też spadek balanced accuracy, czyli średniej skuteczności dla obu klas. Nie są to wyniki na danych treningowych.

**MFCC + regresja:** najważniejsze są `mfcc_02_mean`, `mfcc_13_mean`, `mfcc_10_std`, następnie `mfcc_09_mean` i `mfcc_10_mean`. **MFCC + XGBoost:** `mfcc_02_mean`, `mfcc_13_mean`, `mfcc_09_mean`, `mfcc_10_std`. Pierwsze cztery cechy XGB i pierwsze trzy LR zwiększały log-loss po zaburzeniu w każdym z sześciu podziałów. Podziały współdzielą płytki, więc nie są sześcioma niezależnymi dowodami.

`mfcc_02_mean` opisuje jeden z szerokich wzorców kształtu widma; `mfcc_13_mean` — drobniejsze zróżnicowanie tego kształtu. Numer MFCC nie odpowiada pojedynczej częstotliwości ani pasmu. `mfcc_10_std` mówi, jak bardzo dany opis zmienia się wewnątrz krótkiego dźwięku. Łącznie średnie MFCC są istotniejsze od ich zmienności w czasie. Nie można nazwać konkretnego MFCC „cechą pęknięcia”: mikrofon, podparcie, odszumianie i tło też zmieniają widmo.

**Physics + regresja:** płaskość widma (`flatness`), udział energii 3.18–7.97 kHz (`band_5`) i szerokość widma. **Physics + XGBoost:** udział 80–201 Hz (`band_1`), płaskość, udział 201–504 Hz (`band_2`) i 3.18–7.97 kHz (`band_5`). Płaskość jest większa, gdy energia jest bardziej równomiernie rozłożona, jak w szumie; mniejsza przy wyraźnych tonach. Pasma to udziały w energii analizowanego widma, nie surowa głośność. Niskie częstotliwości mogą również opisywać stanowisko lub odgłos samego uderzenia.

Cechy zaniku nie są obecnie głównym źródłem jakości XGB. To nie dowodzi, że tłumienie jest fizycznie nieistotne: krótkie okna, szybkie serie i możliwe wyciszanie ograniczają jego pomiar. Physics + LR ma słabszy wynik niż MFCC, więc jego ranking trzeba traktować szczególnie ostrożnie.

| Model | Cecha | Wzrost log-loss | Spadek BA | Dodatni wzrost w podziałach |
| --- | --- | --- | --- | --- |
| MFCC + LR | mfcc_02_mean | 0.116 | 12.1 p.p. | 6 |
| MFCC + LR | mfcc_13_mean | 0.057 | 4.7 p.p. | 6 |
| MFCC + LR | mfcc_10_std | 0.045 | 1.7 p.p. | 6 |
| MFCC + LR | mfcc_09_mean | 0.039 | 3.1 p.p. | 5 |
| MFCC + LR | mfcc_10_mean | 0.037 | 4.8 p.p. | 6 |
| MFCC + XGB | mfcc_02_mean | 0.131 | 20.8 p.p. | 6 |
| MFCC + XGB | mfcc_13_mean | 0.058 | 6.7 p.p. | 6 |
| MFCC + XGB | mfcc_09_mean | 0.040 | 6.6 p.p. | 6 |
| MFCC + XGB | mfcc_10_std | 0.031 | 5.5 p.p. | 6 |
| MFCC + XGB | mfcc_10_mean | 0.011 | 2.1 p.p. | 5 |
| MFCC + physics + LR | mfcc_02_mean | 0.059 | 4.1 p.p. | 6 |
| MFCC + physics + LR | mfcc_13_mean | 0.056 | 6.1 p.p. | 6 |
| MFCC + physics + LR | mfcc_10_std | 0.044 | 2.8 p.p. | 6 |
| MFCC + physics + LR | mfcc_10_mean | 0.038 | 5.4 p.p. | 6 |
| MFCC + physics + LR | mfcc_11_mean | 0.030 | 3.8 p.p. | 6 |
| MFCC + physics + XGB | mfcc_02_mean | 0.072 | 8.6 p.p. | 6 |
| MFCC + physics + XGB | mfcc_13_mean | 0.041 | 4.3 p.p. | 6 |
| MFCC + physics + XGB | mfcc_09_mean | 0.031 | 6.0 p.p. | 6 |
| MFCC + physics + XGB | flatness | 0.030 | 3.9 p.p. | 5 |
| MFCC + physics + XGB | band_1 | 0.020 | 4.1 p.p. | 5 |
| physics + LR | flatness | 0.102 | 4.5 p.p. | 6 |
| physics + LR | band_5 | 0.082 | 4.9 p.p. | 6 |
| physics + LR | bandwidth_hz | 0.045 | 1.2 p.p. | 6 |
| physics + LR | band_2 | 0.035 | 1.7 p.p. | 6 |
| physics + LR | band_6 | 0.024 | -0.1 p.p. | 5 |
| physics + XGB | band_1 | 0.108 | 9.3 p.p. | 6 |
| physics + XGB | flatness | 0.102 | 14.3 p.p. | 6 |
| physics + XGB | band_2 | 0.057 | 5.9 p.p. | 6 |
| physics + XGB | band_5 | 0.042 | 5.3 p.p. | 5 |
| physics + XGB | band_4 | 0.009 | -0.4 p.p. | 5 |

Cechy bywają skorelowane. Zmiana jednej może zostać zastąpiona informacją z innej; podmieniona kombinacja cech może też być nietypowa. Ranking nie jest ani udziałem procentowym w decyzji, ani dowodem przyczynowym. Nie wybieraliśmy na jego podstawie nowej listy cech i nie ogłaszamy nowej skuteczności po dostrojeniu.

## Czy istnieje różnica niezwiązana z uderzeniem?

**Tak, mamy wyraźne poszlaki.** W poprzedniej rundzie sama reguła AAC=NOK / WAV=OK dawała 25/27 trafień. Model używający tylko długości, poziomu i wyciszeń, bez kodeka, miał 24/27. Poniżej dodatkowe eksperymenty; to osobne modele kontrolne trenowane od początku na tych samych podziałach po płytkach, a nie podawanie ciszy modelowi wytrenowanemu na pingach.

| Fragment | Model | Pliki | Accuracy | Balanced accuracy | Recall NOK | Odrzucone OK |
| --- | --- | --- | --- | --- | --- | --- |
| Przed pingiem −120…−20 ms | MFCC + LR | 27 | 88.9% | 88.9% | 88.9% | 11.1% |
| Przed pingiem −120…−20 ms | MFCC + XGB | 27 | 92.6% | 88.9% | 77.8% | 0.0% |
| Ciche okna z dala od wykryć | MFCC + LR | 25 | 80.0% | 68.8% | 37.5% | 0.0% |
| Ciche okna z dala od wykryć | MFCC + XGB | 25 | 88.0% | 81.2% | 62.5% | 0.0% |
| Ping 0…150 ms | MFCC + LR | 27 | 100.0% | 100.0% | 100.0% | 0.0% |
| Ping 0…150 ms | MFCC + XGB | 27 | 96.3% | 94.4% | 88.9% | 0.0% |
| Ping 0…60 ms | MFCC + LR | 27 | 96.3% | 97.2% | 100.0% | 5.6% |
| Ping 0…60 ms | MFCC + XGB | 27 | 92.6% | 91.7% | 88.9% | 5.6% |
| Ogon 60…150 ms | MFCC + LR | 27 | 92.6% | 88.9% | 77.8% | 0.0% |
| Ogon 60…150 ms | MFCC + XGB | 27 | 92.6% | 91.7% | 88.9% | 5.6% |

- **100 ms przed uderzeniem** oznacza zakres od −120 do −20 ms. LR ma 24/27, XGB 25/27 trafień. Tło lub pozostałość poprzedniego pingu już zawiera informację o klasie. Przy odstępach około 0.28 s to nie jest czyste tło, więc nie przypisujemy całego wyniku zakłóceniom pomiaru.
- **Najcichsze 100 ms z dala od wykryć:** wybieramy do 5 rozłącznych okien na plik, co najmniej 400 ms po każdym kandydacie i 100 ms przed kolejnym. W 25 plikach znaleziono 118 okien; R03 i G01 nie mają odpowiednich fragmentów. XGB ma 22/25 trafień, balanced accuracy 81.25%, recall NOK 5/8. Zawsze-OK miałby 17/25, ale balanced accuracy tylko 50%. To mocniejszy sygnał różnic tła lub toru zapisu. Fragmenty wybrano automatycznie, mogą zawierać niewykryte dźwięki lub resztki wybrzmień; wymagają odsłuchu. Znaczna część najcichszych fragmentów OK jest cyfrowym zerem. Model nie dostaje osobnej kolumny głośności, ale może rozpoznać brak sygnału, kwantyzację czy kształt szumu.
- **Usunięcie 20 ms przed pingiem:** LR nadal ma 27/27. Samo ucięcie tej części nie usuwa problemu — tło, filtr i charakterystyka nagrywania są obecne także podczas uderzenia.
- **Tylko pierwsze 60 ms pingu:** LR ma 26/27. Krótki atak również przenosi użyteczną informację; nadal może ona dotyczyć siły, narzędzia lub filtracji. Nie jest to gotowy nowy wariant produkcyjny.



Nie ustalamy z tych prób „procentu decyzji wynikających z tła”. Pokazują, że zbiór ma skróty do etykiety i należy je wyeliminować w kolejnym pomiarze. Nowe nagrania oraz stare NOK należą do różnych paczek, a paczka, ustawienia, obiekt i klasa są częściowo splecione. Losowy podział tapów tylko ukryłby ten problem.

## Najkrótszy test, który pomoże rozstrzygnąć

1. Nagrać dostępne OK i NOK naprzemiennie, w jednej sesji, tym samym mikrofonem, aplikacją, gainem, WAV PCM i geometrią. Wyłączyć ulepszenia; potwierdzić faktyczne wejście mikrofonowe.
2. Dla każdej płytki: 10 s nieprzetworzonego tła, 10 tapów co 2–3 s w oznaczonych miejscach, 10 s tła. Zapisać plate_id i session_id. Zmienność tła i pozycje nie mogą pokrywać się z klasą.
3. Powtórzyć po ponownym ustawieniu stanowiska. Zamrozić model z pierwszej sesji i sprawdzić drugą bez dostrajania. To test odporności sesyjnej tych samych płytek; osobny test nowych fizycznych sztuk nadal jest konieczny.
4. Powtórzyć kontrolę samego tła. Jeżeli tło nadal przewiduje klasę, szukać różnicy toru albo sekwencji pomiaru. Można też domieszać to samo obce tło lub zamieniać tła między klasami i mierzyć zmianę decyzji, ale taki eksperyment nie odzyska fragmentów wyciętych przez bramkę szumów.

Nie naprawimy stratnego AAC przez zapis do WAV i nie udowodnimy braku różnic pomiaru samą normalizacją amplitudy. Obie klasy potrzebują wspólnych warunków. To teraz ważniejsze niż wybór regresja vs XGB.

Metoda interpretacji: [scikit-learn — permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [ograniczenia interpretacji współczynników](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html). Wszystkie podane liczby policzono lokalnie z obu paczek klienta.
