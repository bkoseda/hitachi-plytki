# Parametry i praktyczne ograniczenia

## Plik → uderzenie → cechy → płytka

- Oryginały nie są nadpisywane. WAV/FLAC/OGG/MP3 czyta libsndfile, pozostałe obsługiwane kodeki — lokalny FFmpeg.
- Domyślny limit pliku: 180 sekund. Dłuższy plik podziel lub świadomie zwiększ `Config(max_seconds=...)`.
- Analiza: mono 48 kHz. Dla wielu kanałów wybieramy kanał o najwyższym RMS, nie średnią. W notebooku 01 można wymusić kanał.
  W manifeście opcjonalna kolumna `channel` wymusza kanał (0, 1, …); puste pole = auto.
  Ten wybór jest zachowywany również przy obliczaniu AST. Automat może wybrać głośniejszy kanał szumowy.
- Kontrola jakości jest liczona przed odjęciem składowej stałej i resamplingiem: peak/RMS dBFS, DC, udział próbek >=0.999.
  Ten wskaźnik nie wykrywa każdego analogowego przesterowania ani nie mierzy SPL. Źródło float >1 może wymagać odsłuchu z normalizacją.
- `fmax=20000` Hz. Wszystkie pliki muszą mieć źródłowy Nyquist co najmniej równy fmax.
  Jeśli któryś plik ma 16 kHz, użyj wspólnego `FMAX=7500`; wyższy resampling nie odtworzy wysokich częstotliwości.
- Kandydaci uderzeń: obwiednia RMS 10 ms / krok 5 ms, próg oparty na medianie i MAD, min. odstęp 0.25 s.
  To heurystyka, nie rozpoznawanie rodzaju zdarzenia. Mowa, stół i narzędzia mogą dać fałszywy hit;
  stały głośny szum albo bardzo bliskie uderzenia mogą ukryć prawdziwy. Zawsze obejrzyj i odsłuchaj podział.
- Okno: 25 ms przed kandydatem + 1 s ogona. Koniec jest obcinany przed następnym uderzeniem lub końcem pliku.
  Kolumna `truncated` zaznacza takie przypadki. Zmiana długości okna wymaga przeliczenia całej serii.
- Ręczne czasy: JSON `{ "nazwa.wav": [0.4, 1.9] }`; klucz musi dokładnie pasować do `path` w manifeście.
  `[]` świadomie odrzuca plik. Po zmianie źródła popraw stare adnotacje; nie są przenośne pomiędzy różnymi nagraniami.

## Co liczymy

`resonance` (domyślny w CLI i notebooku 03) ma 3 cechy: dominujący pik, centroid i czas zaniku.
To mały, interpretowalny punkt startowy; nie gwarantuje odporności na hałas.
`physics` ma 15 cech: dominujący pik, centroid, szerokość widma, rolloff 85%, płaskość, entropia,
czas zejścia o 10 dB, stosunek energii późnej/wczesnej, liczba pików i sześć pasm logarytmicznych.
Welch analizuje ogon po pierwszych 10 ms za atakiem. To usuwa część wpływu kontaktu młotka, ale nie całą.
`mfcc` ma średnie i odchylenia 13 współczynników bez c0 (26 liczb). `mel` to średnie log-energie 40 pasm.
Widma są liczone na kopii z wyrównanym szczytem. Bez kalibrowanego mikrofonu nie porównujemy absolutnego SPL.

`decay10_s` to pierwsza ramka obwiedni <= -10 dB względem maksimum w analizowanym ogonie.
Jeśli spadek nie nastąpi, zwracany jest koniec obserwowanego ogona (pomiar ograniczony oknem).
Interferencja modów, szum i skrócone nagranie wpływają na ten wskaźnik. Nie jest RT60 ani współczynnikiem materiałowym.
Piki o małej separacji mogą zlewać się przy tej rozdzielczości FFT — przy podejrzeniu bliskich rezonansów
zwiększ FFT w spektrogramie i obejrzyj dłuższe, porównywalne okna.

## Ocena bez przecieku

Cechy uderzeń są agregowane medianą na fizyczną płytkę. Następnie stosujemy do 5 podziałów stratyfikowanych
na rozłącznych płytkach. Skalowanie, odrzucenie cech stałych i klasyfikator są uczone od nowa w każdym treningu.
Model: regresja logistyczna, C=0.1, class_weight=balanced. Próg roboczy 0.5.
Z dwóch płytek nie ma oceny; minimum programowe 3 na klasę także nie wystarcza do deklaracji wdrożeniowej.
Przedział bootstrap resampluje fizyczne płytki z już uzyskanych prognoz OOF; nie obejmuje wszystkich źródeł niepewności.
Gdy wszystkie prognozy są trafne, może wyjść pozornie wąski / zerowy przedział. Nie oznacza to braku ryzyka.

Pojedyncze miejsca uderzenia z NOK dziedziczą etykietę detalu. Nie musi to znaczyć, że każdy hit zdradza defekt.
Mediana może go wygładzić. Jeżeli protokół i miejsca są porównywalne, zrób osobną analizę tego samego miejsca
(np. przefiltruj tabelę uderzeń po wcześniej uzupełnionej kolumnie `position`). Nowej strategii „najgorszy hit”
nie oceniaj na zbiorze, na którym ją wymyślono. Liczba uderzeń musi być stała, inaczej maksimum jest obciążone.

Test sesji pozostawia całą sesję poza treningiem. Odmawia, gdy ta sama płytka występuje w wielu sesjach:
w takim przypadku potrzebny jest starannie zaprojektowany podział, który oddzieli jednocześnie płytki i warunki.
Zdjęcia, opisy wad, nazwy plików i ID nie trafiają do klasyfikatora.

## Dobre wzorce

Odległość euklidesowa do najbliższej dobrej płytki po standaryzacji cech, dzielona przez pierwiastek z wymiaru.
Dla samych wzorców pomijamy własny punkt, żeby nie pokazywać zawsze zera. Skalowanie opisuje wszystkie dobre
wzorce; te odległości nie są niezależną oceną OOF. Próg = 1.5 × 95. percentyl odległości między dobrymi wzorcami.
To umowna linia do triage, nie kalibracja odsetka fałszywych alarmów. Nowe dobre nagrania z nowej sesji są konieczne.

## Odpowiedzialna interpretacja

Dwie klasy mogą różnić się sesją, podparciem, geometrią, partią produkcyjną, operatorem lub gainem.
Przeplataj OK/NOK i dokumentuj te czynniki. Celem pierwszego dnia jest sprawdzenie wykonalności i poprawienie pomiaru.
Ani biblioteka, ani wytrenowany wcześniej model nie gwarantują odporności na dowolne audio.

## Test kontrolowanego szumu

Notebook 05 porównuje model uczony na czystych płytkach i na wariantach z dodanym tłem.
Wszystkie warianty płytki pozostają razem; test nigdy nie używa płytki treningowej. Łączna waga
wariantów danej płytki w funkcji kosztu pozostaje taka sama. Jest to test wrażliwości przy znanym
rozkładzie dodawanego szumu, nie niezależna ocena odporności na nową fabrykę. Użycie tego samego
nagrania tła w treningu i teście ogranicza wniosek do tego tła. Zostaw inną sesję tła do końcowego testu.

## Wspólna sesja i porównanie A/B (aktualizacja 26.09)

Notebooki od 00 do 06 czytają `project.json`; dawny opis ręcznych ustawień w każdym notebooku dotyczy tylko opcjonalnych nadpisań. Wspólne pasmo jest automatycznie obniżane do najniższej częstotliwości Nyquista źródeł. Typ danych (`synthetic`, `external`, `client`) jest jawny.

00/06 rozszerzają medianowy baseline z 03:
- A: średnia i odchylenie trzech cech rezonansu dla płyty. Porównujemy najbliższy wzorzec OK oraz tę samą regularizowaną regresję L2 co wcześniej. Nie stroimy kary na wyniku testowym.
- Dystans A jest dzielony przez roboczy próg wyliczony z dobrych płyt treningowych; próg 1 oznacza ten roboczy poziom. To normalizacja skali, nie kalibracja prawdopodobieństwa.
- B: MFCC każdego uderzenia, regresja i średnia do 3 najwyższych ocen. Każda płyta ma łączną wagę 1 w funkcji straty. Skalowanie korzysta tylko z treningowych uderzeń, ale płyty z większą liczbą uderzeń bardziej wpływają na jego statystyki; utrzymuj stały protokół.
- Wspólny zewnętrzny podział stratyfikowany do 5 foldów; żadna fizyczna płyta nie przecieka między treningiem a testem. Przy >=5 płyt każdej klasy w treningu: wewnętrzne 3 foldy dobierają największy próg z zerem przeoczeń NOK. Następnie model jest dopasowany na całym treningu. Przeniesienie progu po reficie może zmienić jego zachowanie; sprawdza to zewnętrzny test.
- Przy zbyt małym treningu próg jest stały, co zapisujemy w `threshold_policy`. Minimum programu 4+4 nie oznacza wystarczającej próby naukowej. Odległość uczy się wyłącznie z OK także wewnątrz zagnieżdżonej oceny.
- Average precision, ROC-AUC, recall, false reject, macierz pomyłek i dokładny dwustronny 95% przedział Cloppera–Pearsona. Bootstrap resampluje ustalone prognozy OOF, a nie cały trening. Porównanie metod jest eksploracyjne; ostateczny wybór wymaga nowej serii.
- Krzywa liczby uderzeń używa wyłącznie ocen OOF modelu B, stałego progu i tej samej grupy płyt. Pomija k większe niż liczba uderzeń na dowolnej płycie; nie zastępuje brakujących stuknięć losowaniem ze zwracaniem.

Galeria 00 używa PNG i osadzonego audio, więc zapisane wyniki działają także bez widgetów. Ścieżki zdjęć są metadanymi do przeglądu, nie cechą modelu. Automatycznie wykryte uderzenia wymagają przeglądu; kod nie potwierdza automatycznie jakości etykiet, podwójnych uderzeń ani przesterowania analogowego.
