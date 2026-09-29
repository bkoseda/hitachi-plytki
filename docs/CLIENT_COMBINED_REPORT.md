# Ocena nagrań płytek OK i NOK

29.09.2026 · dwie paczki klienta · analiza lokalna

## Najważniejsza decyzja

**W tej paczce jest mocny sygnał pozwalający rozróżnić OK i NOK. Sposób zbierania danych wymaga jednak korekty przed oceną skuteczności na nowych płytkach.**
MFCC z prostym klasyfikatorem wystarcza do pierwszej separacji. Transformer nie poprawił tego wyniku.
Jednocześnie sam format i parametry nagrania pozwalają zgadywać klasę. Nie umiemy jeszcze oddzielić wpływu pęknięcia od wpływu sposobu nagrywania.

## Co przeanalizowaliśmy

- 45 plików, 19.74 minuty dźwięku, 48 kHz. Nowa paczka: 21 WAV-ów.
- 3613 kandydatów na uderzenia po filtrach jakości; to nie 3613 niezależnych przykładów materiału.
- Główny test: 27 nagrań, 2621 kandydatów, **2 płytki NOK i roboczo 3 płytki OK**.
- Folder główny OK, Part 2 i Part 3 traktujemy jako różne sztuki na podstawie założenia zaakceptowanego przez użytkownika. Potrzebne potwierdzenie klienta.
- OLD ma nieznane powiązania fizycznych płytek; sprawdzono go dodatkowo, poza głównym wynikiem. Bitten ma inny typ uszkodzenia i nie jest klasą pęknięć.
- G11, G12 i G16 trwają 0,47–1,20 s i nie weszły do modeli. Detekcje pozostają w audycie.

## Czy jakość pomiarów wystarcza

**Do eksploracji: tak. Do wiarygodnego pomiaru tłumienia i deklaracji skuteczności na nowych sztukach: jeszcze nie.**

1. **Wyciszanie sygnału.** W 17 z 45 plików występuje ciąg dokładnych zer dłuższy niż 100 ms, w tym w 10 nowych WAV-ach OK. W G13 trwa aż około 1,40 s. Nie jest to tylko artefakt stratnego AAC. Podejrzenie obejmuje bramkowanie, odszumianie, edycję lub przerwy toru. Nie rozpoznajemy konkretnego mechanizmu i nie dowodzimy utraty początku uderzenia. Cichy ogon jest szczególnie zagrożony.
2. **Zbyt szybka seria do badania pełnego wybrzmienia.** Typowy odstęp to około 0,28 s. Dlatego używamy wspólnego okna 20 ms przed + 150 ms po początku. Wzrost liczby uderzeń nie zastępuje nagrania dłuższego ogona. Na wykresie niebiesko-zielone pola oznaczają okno cech, a nie zmierzony fizyczny czas tłumienia.
3. **Warunki zmieniają się razem z klasą.** Nowe OK są w PCM WAV; 7 z 9 głównych nagrań NOK jest w AAC. Zmieniają się również podpory, narzędzie, kąt, długość nagrania i tło. Model może korzystać z tych różnic.
4. **Brakuje identyfikatorów i pełnego protokołu.** Potrzebne: ID płytki, typ/wymiary, temperatura, miejsce uderzenia, narzędzie, podparcie, odległość i orientacja mikrofonu, gain, aplikacja i ustawienia przetwarzania. OK również różnią się widmem; sprawność nie oznacza identycznego dźwięku.
5. **Tło jest za krótkie i częściowo wyciszone.** Otrzymaliśmy około 31,6 s zamiast ~10 min przewidzianych w minimalnej propozycji.

Pozytywy: wspólne 48 kHz, brak próbek osiągających przyjęty próg przesterowania 0,999 i liczne czytelne impulsy. Kanały stereo są identyczne we wszystkich 45 plikach, więc analizujemy jeden. Brak twardego clippingu nie dowodzi braku limitera lub AGC.

## Jak sprawdziliśmy predykcję

Z każdego użytecznego nagrania wybrano 10 uderzeń równomiernie w czasie, bez patrzenia na wynik modelu. Wszystkie reprezentacje porównujemy na tych samych zdarzeniach. Cechy klasyczne są dostępne także dla wszystkich kandydatów.

W każdej z 6 prób odkładamy **całą jedną płytkę NOK i całą jedną OK**. Trenujemy na pozostałych grupach. Skalowanie uczy się wyłącznie na treningu. Wagi równoważą klasy i grupy; długa seria nie dominuje uczenia. Nie stroiliśmy parametrów, progu ani wyboru cech pod ten test.

Wynik nagrania to mediana ocen 10 uderzeń. Tabela uśrednia oceny z prób, w których nagranie było odłożone; próg wynosi 0,5. To mały eksperyment eksploracyjny, nie niezależny końcowy test produkcyjny. Oceny modelu nie są skalibrowanym prawdopodobieństwem pęknięcia.

MFCC opisuje kształt widma, czyli w przybliżeniu barwę dźwięku. Physics to 15 cech opisujących częstotliwości i zanik. LR to prosty klasyfikator liniowy. XGB to zespół małych drzew. AST jest zamrożonym transformerem AudioSet, który tworzy 768 liczb opisujących dźwięk; dopiero LR lub XGB uczy się naszych etykiet. AST pracuje przy 16 kHz, traci pasmo powyżej 8 kHz i standardowo dopełnia krótkie uderzenia do długiego wejścia. Nie dostrajaliśmy transformera.

## Wyniki i ich granice

MFCC + LR: po uśrednieniu odłożonych predykcji 27/27 nagrań poprawnych, 9/9 nagrań NOK wykrytych. **W jednej pojedynczej próbie odrzucił jedno nagranie OK**; balanced accuracy poszczególnych prób wynosi 90%–100%. Wyniku 100% z tabeli nie należy opisywać jako bezbłędności każdej próby.
AST + LR: 25/27 poprawnych, 8/9 NOK, jedno fałszywe odrzucenie OK. AST + XGB: 24/27. Prostsze MFCC jest obecnie wystarczającym punktem odniesienia.

Po zebraniu ocen wszystkich nagrań danej grupy MFCC i AST poprawnie klasyfikują 5/5 zakładanych płytek. **To nadal tylko dwie uszkodzone sztuki.** Powtarzane podziały ani tysiące uderzeń nie zwiększają tej liczby. Nie podajemy sztucznie wąskich przedziałów ufności liczonych po uderzeniach.

Najważniejszy test kontrolny: reguła „AAC = NOK, WAV = OK” daje 92.6% accuracy bez słuchania uderzeń. Model QC bez informacji o kodeku, używający tylko długości, poziomu i ciszy, daje 88,9%. To nie dowodzi, że MFCC korzysta wyłącznie z tych skrótów, ale uniemożliwia przypisanie całego wyniku pęknięciom.

Po ograniczeniu do 4 podpór bez zmiany narzędzia MFCC i AST rozdzielają 10/10 nagrań. Jednak oba NOK w tym podzbiorze nadal są AAC, więc kontrola formatu też jest idealna. Samo dopasowanie podpór nie usuwa problemu.

Dwa nagrania NOK w WAV (R07, R08) również są rozpoznane przez MFCC, gdy cała płytka Bad_2 jest odłożona. To pozytywna poszlaka, że nie chodzi wyłącznie o rozróżnienie kodeka. Nadal pozostają inne różnice między sesjami i obiektami.

Dokładne przetasowanie etykiet między pięcioma grupami ma tylko 10 możliwych przydziałów dwóch NOK. Rzeczywisty przydział ma najwyższe AUC, ale rozdzielczość tego testu to p=0.10. To nie jest mocny dowód statystyczny; wymienność grup dodatkowo ograniczają różnice zbierania danych.

W OLD, wyłączonym z uczenia, MFCC poprawnie rozpoznaje 6/8 plików, AST 7/8. Nie traktujemy tego jako niezależnego testu płytek: mogą to być nagrania tych samych sztuk.

## Szum i liczba uderzeń

Ściszenie o 12 dB nie zmienia wyników, zgodnie z normalizacją amplitudy. Po dodaniu aktywnego fragmentu nagranego tła do odłożonych uderzeń MFCC nadal wykrywa 9/9 nagrań NOK, ale fałszywie odrzuca 5/18 OK przy SNR 20 dB i 6/18 przy 10 dB. Niższy SNR oznacza mocniejsze tło. To kontrolowana domieszka do okien już wykrytych, nie test detektora na zaszumionej całej sesji. AST nie przeszedł w tej rundzie testu szumu.

Wykres 1/3/5/10 uderzeń pokazuje stabilność mediany ocen na tych samych odłożonych nagraniach. Pasmo oznacza rozrzut 100 losowań, nie przedział ufności skuteczności na nowych płytkach. Nie wyznaczamy jeszcze minimalnej liczby uderzeń do produkcji ani nie porównujemy „master tap” bez pozycji trafień.

## Co zmienić przy następnej paczce

**Najpierw krótki test ustawień.** Jedna OK i jedna NOK, ta sama geometria, narzędzie i gain. Dla każdej wariant obecny oraz z wyłączonymi ulepszeniami wejścia mikrofonowego, odszumianiem, AGC i usuwaniem echa. 10 s tła, 10 uderzeń z przerwami 2–3 s, 10 s tła. Zapisać bezpośrednio WAV PCM 48 kHz, najlepiej 24 bit, bez efektów. Nie konwertować starego AAC jako substytutu nowego pomiaru. Potwierdzić fizycznie wybrane wejście mikrofonowe.

**Następnie powtarzalna seria wszystkich dostępnych płytek.** Jeden ustalony schemat podpór i narzędzie; 5 oznaczonych miejsc po 2 uderzenia. Te same miejsca dla OK i NOK. Nagrywać kolejno OK, NOK, OK, NOK, aby zmiana tła w czasie nie pokrywała się z klasą. Powtórzyć po zdjęciu i ponownym ułożeniu płytek, zapisać session_id. Siła możliwie podobna, bez dociskania płyty podczas wybrzmienia. Zmiany narzędzia i podpór dopiero jako osobny, jawnie oznaczony eksperyment dla obu klas.

**Więcej różnych sztuk, nie setki tapów tej samej.** Zachować co najmniej 30 OK i 10 NOK jako cel danych ze slajdu minimalnego PoC. Potwierdzić etykiety i zdjęcia, oddzielić pęknięcia od wyszczerbień. Nagrać około 10 min nieprzetworzonego tła w typowych warunkach.

miniDSP wprost zaleca wyłączenie ulepszeń Windows dla UMIK-1. Mikrofon jest w propozycji przykładem; model faktycznie użyty wymaga potwierdzenia. Kalibracja charakterystyki mikrofonu i ręczny gain nie są tym samym co adaptacyjne odszumianie.

## Zakres i odtwarzanie

To realizuje etap QA, spektrogramów, klasycznych cech i prób rozróżniania z minimalnego PoC. AST jest dodatkiem porównawczym, nie zmianą zobowiązania klienta. Przejrzano wykresy reprezentatywnych fragmentów, nie odsłuchano ręcznie wszystkich 3613 kandydatów. Notebook zawiera odtwarzacz i przegląd do zatwierdzenia segmentacji.

Notebooki: `notebooks/client/02_jakosc_obu_paczek.ipynb` i `03_modele_obu_paczek.ipynb`. Wybierz kernel `.venv` i Run All. Pierwszy sprawdza dane i pozwala klikać nagrania. Drugi przelicza modele, podziały, testy kontrolne, szum i liczbę uderzeń. Wagi AST są już lokalnie dostępne na tym Macu; nie wysyłamy audio do usługi zewnętrznej.

Źródła: slajd 1 `docs/Hitachi PoC proposals 2.pptx`, roboczy `docs/Hitachi DQC - Methods Guideline (PoC Sprint).docx`; producent miniDSP; model MIT AST; dokumentacja walidacji grupowej scikit-learn.
